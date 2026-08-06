from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

target = Path("/data1/linyewei/models/Qwen3-8B")
draft = Path("/data1/linyewei/models/dspark_qwen3_8b_block7")

tcfg = AutoConfig.from_pretrained(target, local_files_only=True)
dcfg = AutoConfig.from_pretrained(draft, local_files_only=True)
tok = AutoTokenizer.from_pretrained(target, local_files_only=True)

assert tcfg.model_type == "qwen3"
assert tcfg.hidden_size == 4096
assert tcfg.num_hidden_layers == 36
assert dcfg.architectures[0] == "Qwen3DSparkModel"
assert dcfg.block_size == 7
assert dcfg.num_hidden_layers == 5
assert dcfg.target_layer_ids == [1, 9, 17, 25, 33]
assert dcfg.markov_rank == 256
assert dcfg.markov_head_type == "vanilla"
assert dcfg.enable_confidence_head is True
assert dcfg.vocab_size == tcfg.vocab_size == 151936
assert 0 <= dcfg.mask_token_id < dcfg.vocab_size

with safe_open(draft / "model.safetensors", framework="pt", device="cpu") as handle:
    keys = set(handle.keys())

required = {
    "embed_tokens.weight",
    "lm_head.weight",
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
    "confidence_head.proj.weight",
}
assert required <= keys, required - keys

print("tokenizer length:", len(tok))
print("target/draft config and safetensors keys OK")