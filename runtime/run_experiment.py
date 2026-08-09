#!/usr/bin/env python3
"""Run one reproducible DSpark evaluation inside a dedicated result directory."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers


REPO_ROOT = Path("/data/home/wly/dLLM/DeepSpec")
DATASET_ROOT = REPO_ROOT / "eval_datasets"
RESULT_ROOT = Path("/data/home/wly/dLLM/DeepSpec-results/qwen3_8b")
DEFAULT_TARGET = Path("/data1/linyewei/models/Qwen3-8B")
DEFAULT_DRAFT = Path("/data1/linyewei/models/dspark_qwen3_8b_block7")
# Keep this path identical to all observation launchers so baseline and
# observation jobs participate in the same cross-process port lease protocol.
PORT_LEASE_ROOT = Path("/tmp/deepspec_conditional_confidence_ports")
DATASET_CAPS = OrderedDict(
    [
        ("gsm8k", 500),
        ("math500", 500),
        ("aime25", 30),
        ("humaneval", 164),
        ("mbpp", 256),
        ("livecodebench", 500),
        ("mt-bench", 80),
        ("alpaca", 500),
        ("arena-hard-v2", 500),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DSpark on all nine datasets or one selected dataset."
    )
    parser.add_argument("dataset", choices=["all", *DATASET_CAPS])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--dist-timeout-minutes", type=int, default=24 * 60)
    parser.add_argument("--dist-backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument(
        "--master-addr",
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Explicit distributed port. If omitted, reserve a currently free local port.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override the configured cap; mainly useful for a smoke test.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_manifest(path: Path, manifest: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def gpu_names(count: int) -> list[str]:
    names = []
    for index in range(count):
        try:
            names.append(torch.cuda.get_device_name(index))
        except Exception as exc:  # pragma: no cover - hardware-specific fallback
            names.append(f"unavailable: {exc!r}")
    return names


def build_tasks(dataset: str, max_samples: int | None) -> list[tuple[str, int]]:
    selected = list(DATASET_CAPS.items()) if dataset == "all" else [(dataset, DATASET_CAPS[dataset])]
    if max_samples is not None:
        selected = [(name, min(cap, max_samples)) for name, cap in selected]
    return selected


def _port_is_bindable(master_addr: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            sock.bind((master_addr, port))
        return True
    except OSError:
        return False


def reserve_master_port(
    master_addr: str,
    requested_port: int | None,
    run_dir: Path,
) -> tuple[int, Path, str]:
    """Reserve a free distributed port across concurrent DeepSpec launches."""

    PORT_LEASE_ROOT.mkdir(parents=True, exist_ok=True)
    attempts = 1 if requested_port is not None else 100
    for _ in range(attempts):
        if requested_port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((master_addr, 0))
                port = int(probe.getsockname()[1])
        else:
            port = int(requested_port)
        if port < 1 or port > 65535:
            raise ValueError("--master-port must be in [1, 65535]")
        if not _port_is_bindable(master_addr, port):
            if requested_port is not None:
                raise RuntimeError(
                    "Requested distributed port is already in use: "
                    f"{master_addr}:{port}"
                )
            continue
        lease_path = PORT_LEASE_ROOT / f"{port}.json"
        try:
            descriptor = os.open(
                lease_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            if requested_port is not None:
                raise RuntimeError(
                    "Requested distributed port is leased by another DeepSpec run: "
                    f"{port}"
                )
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "master_addr": master_addr,
                    "master_port": port,
                    "run_dir": str(run_dir),
                    "created_at": now_iso(),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
        return port, lease_path, "explicit" if requested_port is not None else "auto_reserved"
    raise RuntimeError("Could not reserve a free distributed port after 100 attempts")


def release_port_lease(lease_path: Path) -> None:
    lease_path.unlink(missing_ok=True)


def main() -> None:
    cli = parse_args()
    if cli.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not math.isfinite(cli.temperature) or cli.temperature < 0.0:
        raise ValueError("--temperature must be finite and non-negative")
    if cli.confidence_threshold < 0:
        raise ValueError("--confidence-threshold must be non-negative")
    if cli.max_samples is not None and cli.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if cli.dist_timeout_minutes <= 0:
        raise ValueError("--dist-timeout-minutes must be positive")

    run_dir = cli.run_dir.expanduser().resolve()
    if run_dir.parent != RESULT_ROOT.resolve():
        raise ValueError(f"--run-dir must be a direct child of {RESULT_ROOT}")
    if re.fullmatch(r"\d{8}_\d{6}_.+", run_dir.name) is None:
        raise ValueError(
            "--run-dir name must follow YYYYMMDD_HHMMSS_<task-label>"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "experiment_manifest.json"
    tensorboard_dir = run_dir / "tensorboard"
    artifact_dir = tensorboard_dir / "artifacts" / f"step_{cli.step}"
    progress_dir = run_dir / "progress"
    dataset_results_path = run_dir / "dataset_results.jsonl"
    if manifest_path.exists() or tensorboard_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse an existing experiment directory: {run_dir}"
        )

    target = cli.target.expanduser().resolve()
    draft = cli.draft.expanduser().resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Target model directory not found: {target}")
    if not draft.is_dir():
        raise FileNotFoundError(f"Draft model directory not found: {draft}")

    tasks = build_tasks(cli.dataset, cli.max_samples)
    dataset_records = []
    for name, cap in tasks:
        path = DATASET_ROOT / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        total_lines = line_count(path)
        dataset_records.append(
            {
                "name": name,
                "path": str(path),
                "configured_max_samples": cap,
                "file_line_count": total_lines,
                "effective_max_samples": min(cap, total_lines),
                "sha256": sha256_file(path),
                "status": "pending",
                "phase": "pending",
                "started_at": None,
                "completed_at": None,
                "result": None,
            }
        )

    cuda_device_count = torch.cuda.device_count()
    if cuda_device_count < 1:
        raise RuntimeError("No visible CUDA device; check CUDA_VISIBLE_DEVICES and PyTorch CUDA support")

    master_port, port_lease_path, port_source = reserve_master_port(
        cli.master_addr,
        cli.master_port,
        run_dir,
    )
    os.environ["MASTER_ADDR"] = cli.master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    # Also cover errors before the spawn try/finally (for example artifact
    # creation or evaluator import failures). Normal completion unregisters it.
    atexit.register(release_port_lease, port_lease_path)

    start_wall = time.time()
    manifest = {
        "schema_version": 2,
        "status": "running",
        "start_time": now_iso(),
        "last_update_time": now_iso(),
        "end_time": None,
        "elapsed_seconds": None,
        "completed_dataset_count": 0,
        "run_dir": str(run_dir),
        "mode": cli.dataset,
        "datasets": dataset_records,
        "models": {"target": str(target), "draft": str(draft)},
        "hyperparameters": {
            "max_new_tokens": cli.max_new_tokens,
            "temperature": cli.temperature,
            "temperature_mode": "greedy" if cli.temperature < 1e-5 else "sampling",
            "confidence_threshold": cli.confidence_threshold,
            "seed": cli.seed,
            "step": cli.step,
            "max_samples_override": cli.max_samples,
            "enable_thinking": False,
            "draft_block_size": 7,
            "batch_size_per_process": 1,
            "attention_implementation": "sdpa",
            "dist_timeout_minutes": cli.dist_timeout_minutes,
        },
        "distributed": {
            "strategy": "one full target+draft replica per visible GPU; samples are rank-sharded",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_count": cuda_device_count,
            "visible_gpu_names": gpu_names(cuda_device_count),
            "master_addr": cli.master_addr,
            "master_port": master_port,
            "master_port_source": port_source,
            "port_lease_path": str(port_lease_path),
            "input_rank": os.environ.get("RANK"),
            "input_world_size": os.environ.get("WORLD_SIZE"),
            "backend": cli.dist_backend,
        },
        "environment": {
            "cwd": os.getcwd(),
            "hostname": socket.gethostname(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "torch_cuda_version": torch.version.cuda,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "repository": {
            "path": str(REPO_ROOT),
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output("status", "--short"),
        },
        "invocation": shlex.join([sys.executable, *sys.argv]),
        "shell_logging": f"stdout+stderr are expected in {run_dir / 'eval.log'} via tee with pipefail",
        "outputs": {
            "manifest": str(manifest_path),
            "combined_log": str(run_dir / "eval.log"),
            "dataset_results_jsonl": str(dataset_results_path),
            "progress_dir": str(progress_dir),
            "tensorboard_dir": str(tensorboard_dir),
            "artifact_dir": str(artifact_dir),
        },
        "error": None,
    }
    artifact_dir.mkdir(parents=True, exist_ok=False)
    progress_dir.mkdir(parents=True, exist_ok=False)
    dataset_results_path.touch(exist_ok=False)
    write_manifest(manifest_path, manifest)
    print(f"Experiment directory: {run_dir}", flush=True)
    print(f"Experiment manifest: {manifest_path}", flush=True)
    print(f"Incremental dataset results: {dataset_results_path}", flush=True)
    print(
        f"Reserved distributed endpoint: {cli.master_addr}:{master_port} "
        f"({port_source})",
        flush=True,
    )

    worker_args = SimpleNamespace(
        target_name_or_path=str(target),
        draft_name_or_path=str(draft),
        max_new_tokens=cli.max_new_tokens,
        temperature=cli.temperature,
        confidence_threshold=cli.confidence_threshold,
        tensorboard_dir=str(tensorboard_dir),
        step=cli.step,
        seed=cli.seed,
        tasks=tasks,
        experiment_manifest_path=str(manifest_path),
        dataset_results_path=str(dataset_results_path),
        progress_dir=str(progress_dir),
        dist_timeout_minutes=cli.dist_timeout_minutes,
        dist_backend=cli.dist_backend,
    )

    from eval import main as evaluation_worker

    try:
        torch.multiprocessing.spawn(
            evaluation_worker,
            args=(worker_args,),
            nprocs=cuda_device_count,
            join=True,
        )
    except BaseException as exc:
        final_status = "failed"
        final_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    else:
        final_status = "completed"
        final_error = None
        print(f"Experiment completed: {run_dir}", flush=True)
    finally:
        try:
            final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            final_manifest = manifest
        final_manifest["status"] = final_status
        final_manifest["error"] = final_error
        final_manifest["last_update_time"] = now_iso()
        final_manifest["end_time"] = now_iso()
        final_manifest["elapsed_seconds"] = round(time.time() - start_wall, 3)
        if final_status == "failed":
            for dataset_record in final_manifest.get("datasets", []):
                if dataset_record.get("status") == "running":
                    dataset_record["status"] = "failed"
                    dataset_record["phase"] = "failed"
                    dataset_record["failed_at"] = now_iso()
        try:
            write_manifest(manifest_path, final_manifest)
        finally:
            release_port_lease(port_lease_path)
            atexit.unregister(release_port_lease)


if __name__ == "__main__":
    main()
