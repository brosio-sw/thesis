"""
LLaDA model & tokenizer loading utilities.

Thin wrapper around HuggingFace AutoModel/AutoTokenizer.
We use 'GSAI-ML/LLaDA-8B-Instruct' by default (the instruction-tuned 8B).

Reference
---------
LLaDA (arXiv:2502.09992) – Large Language Diffusion with mAsking.
https://github.com/ML-GSAI/LLaDA
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer


LLADA_DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
LLADA_MASK_ID = 126336   # vocabulary id of the [MASK] token in LLaDA


def load_model(
    model_name: str = LLADA_DEFAULT_MODEL,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """
    Load the LLaDA model onto `device`.

    Args:
        model_name: HuggingFace model repo name / local path.
        device:     Target device ('cuda', 'cpu', …).
        dtype:      Torch dtype.  bfloat16 is recommended for VRAM efficiency
                    on RTX-class GPUs.

    Returns:
        Model in eval mode.
    """
    print(f"[model] Loading {model_name} …")
    # Load on CPU first — avoids GPU OOM during transformers 5.x loading.
    # The new core_model_loading.py places every tensor on GPU before any
    # bitsandbytes quantization can help, so we load on CPU then dispatch.
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=dtype,
    ).eval()

    if not torch.cuda.is_available() or str(device) == "cpu":
        return model

    # Use accelerate dispatch to split layers between GPU and CPU RAM.
    from accelerate import dispatch_model, infer_auto_device_map

    # Leave ~1.5 GB of VRAM headroom for activations during generation.
    gpu_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_for_weights = int(gpu_bytes - 1.5 * 1024 ** 3)
    gpu_limit = f"{gpu_for_weights // (1024 ** 3)}GiB"
    print(f"[model] Dispatching to GPU+CPU (GPU weight budget: {gpu_limit}) …")

    device_map = infer_auto_device_map(
        model,
        max_memory={0: gpu_limit, "cpu": "60GiB"},
        no_split_module_classes=[
            "LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock"
        ],
    )
    n_gpu = sum(1 for v in device_map.values() if str(v) not in ("cpu", "disk"))
    n_cpu = sum(1 for v in device_map.values() if str(v) == "cpu")
    print(f"[model] {n_gpu} modules on GPU, {n_cpu} on CPU")

    return dispatch_model(model, device_map=device_map)


def load_tokenizer(model_name: str = LLADA_DEFAULT_MODEL) -> AutoTokenizer:
    """
    Load the matching tokenizer.

    LLaDA sampling code requires *left-padding* so that the prompt occupies
    the left side of the tensor (the code appends generation positions on the
    right).  We enforce this here.

    Also verifies that the pad token id ≠ mask id (a hard requirement of the
    generate function from the original repo).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    # LLaDA's generate() is written for left-padding
    tokenizer.padding_side = "left"

    assert tokenizer.pad_token_id != LLADA_MASK_ID, (
        "Pad token id must differ from the mask id (126336). "
        "This is a requirement of the LLaDA sampling code."
    )

    return tokenizer
