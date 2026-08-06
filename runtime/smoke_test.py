from types import SimpleNamespace

from eval import main

args = SimpleNamespace(
    target_name_or_path="/data1/linyewei/models/Qwen3-8B",
    draft_name_or_path="/data1/linyewei/models/dspark_qwen3_8b_block7",
    max_new_tokens=64,
    temperature=1.0,
    confidence_threshold=0.0,
    tensorboard_dir=None,
    step=None,
    seed=980406,
    tasks=[("gsm8k", 2)],
)
main(0, args)