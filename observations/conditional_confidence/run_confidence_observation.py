#!/usr/bin/env python3
"""Launch the isolated DSpark conditional-confidence observation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
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
PORT_LEASE_ROOT = Path("/tmp/deepspec_conditional_confidence_ports")
DATASET_CAPS = OrderedDict(
    (
        ("gsm8k", 500),
        ("math500", 500),
        ("aime25", 30),
        ("humaneval", 164),
        ("mbpp", 256),
        ("livecodebench", 500),
        ("mt-bench", 80),
        ("alpaca", 500),
        ("arena-hard-v2", 500),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DSpark while observing raw conditional confidence and the "
            "correction token's true rank in the full draft distribution."
        )
    )
    parser.add_argument("dataset", choices=("all", *DATASET_CAPS))
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
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
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
        help="Override each selected dataset cap; intended primarily for smoke tests.",
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


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_settings_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_local_config(model_dir: Path) -> tuple[dict, str]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Local checkpoint config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8")), sha256_file(config_path)


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


def reserve_master_port(master_addr: str, requested_port: int | None, run_dir: Path) -> tuple[int, Path, str]:
    """Reserve a free port across concurrent launches of this experiment."""
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
                raise RuntimeError(f"Requested distributed port is already in use: {master_addr}:{port}")
            continue
        lease_path = PORT_LEASE_ROOT / f"{port}.json"
        try:
            descriptor = os.open(lease_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if requested_port is not None:
                raise RuntimeError(f"Requested distributed port is leased by another observation run: {port}")
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


def validate_run_directory(run_dir: Path) -> None:
    if run_dir.parent != RESULT_ROOT.resolve():
        raise ValueError(f"--run-dir must be a direct child of {RESULT_ROOT}")
    if re.fullmatch(r"\d{8}_\d{6}_.+", run_dir.name) is None:
        raise ValueError("--run-dir name must follow YYYYMMDD_HHMMSS_<task-label>")
    run_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(path.name for path in run_dir.iterdir() if path.name != "eval.log")
    if unexpected:
        raise FileExistsError(
            f"Refusing to reuse a non-empty experiment directory {run_dir}; found {unexpected}"
        )


def main() -> None:
    cli = parse_args()
    if cli.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if cli.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0.0 <= cli.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1]")
    if cli.max_samples is not None and cli.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if cli.dist_timeout_minutes <= 0:
        raise ValueError("--dist-timeout-minutes must be positive")

    run_dir = cli.run_dir.expanduser().resolve()
    target = cli.target.expanduser().resolve()
    draft = cli.draft.expanduser().resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Target model directory not found: {target}")
    if not draft.is_dir():
        raise FileNotFoundError(f"Draft model directory not found: {draft}")

    target_config, target_config_sha = read_local_config(target)
    draft_config, draft_config_sha = read_local_config(draft)
    architecture = (draft_config.get("architectures") or [None])[0]
    if architecture != "Qwen3DSparkModel":
        raise ValueError(f"This experiment requires Qwen3DSparkModel, got {architecture!r}")
    if not bool(draft_config.get("enable_confidence_head", False)):
        raise ValueError("Draft checkpoint config has enable_confidence_head=false")
    if not bool(draft_config.get("confidence_head_with_markov", False)):
        raise ValueError("Draft checkpoint does not enable the Markov-corrected confidence head")
    if int(draft_config.get("markov_rank", 0)) <= 0:
        raise ValueError("Draft checkpoint does not provide a Markov correction head")

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
    validate_run_directory(run_dir)
    master_port, port_lease_path, port_source = reserve_master_port(
        cli.master_addr,
        cli.master_port,
        run_dir,
    )
    os.environ["MASTER_ADDR"] = cli.master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    settings_path = run_dir / "settings.json"
    manifest_path = run_dir / "experiment_manifest.json"
    dataset_results_path = run_dir / "dataset_results.jsonl"
    progress_dir = run_dir / "progress"
    tensorboard_dir = run_dir / "tensorboard"
    baseline_artifact_dir = tensorboard_dir / "artifacts" / f"step_{cli.step}"
    observation_artifact_root = run_dir / "observations" / "conditional_confidence"

    settings = {
        "schema_version": 1,
        "experiment": "dspark_qwen3_8b_conditional_confidence_and_true_draft_rank",
        "created_at": now_iso(),
        "run_dir": str(run_dir),
        "purpose": {
            "accepted_distribution": "Raw conditional confidence of every accepted draft position.",
            "rejected_distribution": "Raw conditional confidence of each first rejected/replaced position.",
            "signed_gap": "mean(s_k for accepted draft positions in the round) - s_rejected; sign is retained.",
            "signed_relative_gap": "signed_gap / mean(s_k for accepted draft positions in the round); reported as ratio and percent.",
            "true_draft_rank": "1 + count(q_k[v] > q_k[correction_token]) over the complete Markov-corrected q_k; categories 1..10,other.",
            "cdf_bin_width": 0.05,
        },
        "scope": {
            "dataset_selection": cli.dataset,
            "tasks": [list(task) for task in tasks],
            "enable_thinking": False,
            "batch_size_per_process": 1,
            "attention_implementation": "sdpa",
            "not_measured": ["task accuracy", "pass@1", "judge score", "end-to-end speed", "TPS"],
        },
        "models": {
            "target": str(target),
            "draft": str(draft),
            "target_config_sha256": target_config_sha,
            "draft_config_sha256": draft_config_sha,
            "target_architectures": target_config.get("architectures"),
            "draft_architectures": draft_config.get("architectures"),
            "draft_block_size": draft_config.get("block_size"),
            "draft_target_layer_ids": draft_config.get("target_layer_ids"),
            "draft_enable_confidence_head": draft_config.get("enable_confidence_head"),
            "draft_confidence_head_with_markov": draft_config.get("confidence_head_with_markov"),
            "draft_markov_rank": draft_config.get("markov_rank"),
            "draft_markov_head_type": draft_config.get("markov_head_type"),
        },
        "hyperparameters": {
            "max_new_tokens": cli.max_new_tokens,
            "temperature": cli.temperature,
            "confidence_threshold": cli.confidence_threshold,
            "seed": cli.seed,
            "step": cli.step,
            "max_samples_override": cli.max_samples,
            "dist_timeout_minutes": cli.dist_timeout_minutes,
            "dist_backend": cli.dist_backend,
        },
        "datasets": dataset_records,
        "distributed": {
            "strategy": "one target+draft replica per visible GPU; dataset samples are rank-sharded",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_count": cuda_device_count,
            "visible_gpu_names": gpu_names(cuda_device_count),
            "master_addr": cli.master_addr,
            "master_port": master_port,
            "master_port_source": port_source,
            "port_lease_path": str(port_lease_path),
            "input_rank": os.environ.get("RANK"),
            "input_world_size": os.environ.get("WORLD_SIZE"),
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
        "outputs": {
            "settings": str(settings_path),
            "manifest": str(manifest_path),
            "combined_log": str(run_dir / "eval.log"),
            "dataset_results_jsonl": str(dataset_results_path),
            "progress_dir": str(progress_dir),
            "tensorboard_dir": str(tensorboard_dir),
            "existing_confidence_artifact_dir": str(baseline_artifact_dir),
            "conditional_confidence_artifact_root": str(observation_artifact_root),
        },
    }

    # This is deliberately the first experiment metadata written after the run
    # directory is accepted, and it is never rewritten by this launcher.
    try:
        write_settings_exclusive(settings_path, settings)
    except BaseException:
        port_lease_path.unlink(missing_ok=True)
        raise

    start_wall = time.time()
    manifest = {
        "schema_version": 3,
        "experiment": settings["experiment"],
        "settings_path": str(settings_path),
        "status": "running",
        "start_time": now_iso(),
        "last_update_time": now_iso(),
        "end_time": None,
        "elapsed_seconds": None,
        "completed_dataset_count": 0,
        "run_dir": str(run_dir),
        "mode": cli.dataset,
        "datasets": dataset_records,
        "models": settings["models"],
        "hyperparameters": settings["hyperparameters"],
        "distributed": settings["distributed"],
        "repository": settings["repository"],
        "outputs": settings["outputs"],
        "error": None,
    }

    final_status = "failed"
    final_error = None
    try:
        baseline_artifact_dir.mkdir(parents=True, exist_ok=False)
        observation_artifact_root.mkdir(parents=True, exist_ok=False)
        progress_dir.mkdir(parents=True, exist_ok=False)
        dataset_results_path.touch(exist_ok=False)
        write_json_atomic(manifest_path, manifest)
        print(f"Experiment directory: {run_dir}", flush=True)
        print(f"Immutable settings: {settings_path}", flush=True)
        print(f"Reserved distributed endpoint: {cli.master_addr}:{master_port}", flush=True)

        worker_args = SimpleNamespace(
            target_name_or_path=str(target),
            draft_name_or_path=str(draft),
            max_new_tokens=cli.max_new_tokens,
            temperature=cli.temperature,
            confidence_threshold=cli.confidence_threshold,
            tensorboard_dir=str(tensorboard_dir),
            observation_artifact_root=str(observation_artifact_root),
            step=cli.step,
            seed=cli.seed,
            tasks=tasks,
            experiment_manifest_path=str(manifest_path),
            dataset_results_path=str(dataset_results_path),
            progress_dir=str(progress_dir),
            settings_path=str(settings_path),
            dist_timeout_minutes=cli.dist_timeout_minutes,
            dist_backend=cli.dist_backend,
        )

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from observations.conditional_confidence.confidence_observation import evaluation_worker

        os.chdir(REPO_ROOT)
        torch.multiprocessing.spawn(
            evaluation_worker,
            args=(worker_args,),
            nprocs=cuda_device_count,
            join=True,
        )
        final_status = "completed"
        print(f"Experiment completed: {run_dir}", flush=True)
    except BaseException as exc:
        final_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        port_lease_path.unlink(missing_ok=True)
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
        write_json_atomic(manifest_path, final_manifest)


if __name__ == "__main__":
    main()
