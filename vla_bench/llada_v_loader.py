"""
LLaDA-V wrapper for the VLABench VLM sanity grid.

Mirrors the surface of ``vla_bench.model_loader.Qwen2_VL_Local``:

  * ``self.name``                                    : str
  * ``self.evaluate_with_ti_list(ti_list, with_CoT)``: dict with
        ``"origin_output"`` and one of
        ``"skill_sequence"`` / ``"format_error"``.

so that ``run_vlm_sanity_grid.py`` is fully model-agnostic above this
interface. The official LLaDA-V demo (`LLaDA-V/train/generate_demo.py`) is
the reference for how to load and call the model — we mirror it as closely
as possible. No steering, no custom commit policy.

Memory layout
-------------
LLaDA-V's full weights are ~16 GB in float16; the host GPU is a 12 GB
RTX 4070 Ti. We pass ``device_map="auto"`` with ``max_memory`` so accelerate
splits the model across GPU and CPU (the same offload pattern used by
``llada/model.py``).
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
LLADA_V_REPO_DIR = REPO_ROOT / "vla_bench" / "LLaDA-V"
LLADA_V_TRAIN_DIR = LLADA_V_REPO_DIR / "train"


def _ensure_llava_on_path() -> None:
    """Make ``import llava.*`` resolve to the cloned LLaDA-V/train tree."""
    if not LLADA_V_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"LLaDA-V clone not found at {LLADA_V_REPO_DIR}. "
            "Run: git clone https://github.com/ML-GSAI/LLaDA-V.git "
            f"{LLADA_V_REPO_DIR}"
        )
    if str(LLADA_V_TRAIN_DIR) not in sys.path:
        sys.path.insert(0, str(LLADA_V_TRAIN_DIR))


_SIGLIP_PATCH_APPLIED = False


def _patch_siglip_load_to_construct_from_config() -> None:
    """
    Replace ``SigLipVisionTower.load_model`` so it builds a fresh empty
    ``SigLipVisionModel(SigLipVisionConfig())`` instead of calling
    ``SigLipVisionModel.from_pretrained(...)``.

    The LLaDA-V 16 GB checkpoint contains the full vision tower, so the
    state-dict loader of the outer ``LlavaLLaDAModelLM.from_pretrained`` will
    fill the empty submodule. Avoiding ``from_pretrained`` here sidesteps
    the transformers-5.x meta-device guard and the missing
    ``SigLipVisionConfig._set_token_in_kwargs`` symbol — both of which
    only exist because the bundled SigLIP code was written against an
    older transformers release.
    """
    global _SIGLIP_PATCH_APPLIED
    if _SIGLIP_PATCH_APPLIED:
        return

    import torch
    import torch.nn as nn
    from llava.model.multimodal_encoder import siglip_encoder as _sm

    def _patched_load_model(self, device_map=None):  # noqa: ARG001
        if self.is_loaded:
            return
        # Construct on CPU explicitly. Without this, when called inside the
        # meta-device window opened by ``low_cpu_mem_usage=True``, the new
        # parameters are created on the meta device and the LLaDA-V state-
        # dict loader does *not* repopulate them with real tensors (the
        # transformers/accelerate path treats them as offloaded to disk
        # without writing the offload file), so a later forward fails with
        # "Cannot copy out of meta tensor; no data!".
        with torch.device("cpu"):
            empty_vt = _sm.SigLipVisionModel(_sm.SigLipVisionConfig())
            # Mirror the post-processing done by the original load_model so
            # the submodule structure matches the LLaDA-V checkpoint.
            del empty_vt.vision_model.encoder.layers[-1:]
            empty_vt.vision_model.head = nn.Identity()
        empty_vt.requires_grad_(False)
        self.vision_tower = empty_vt
        self.is_loaded = True

    _sm.SigLipVisionTower.load_model = _patched_load_model
    _SIGLIP_PATCH_APPLIED = True


# Default decoding hyperparameters for the first VLABench baseline.
# Per user spec: gen_length=256, steps=256, block_length=128.
# Two 128-token blocks, 128 denoising steps per block, ~1 token committed
# per step under low_confidence remasking.
DEFAULT_GEN_KWARGS = {
    "steps": 256,
    "gen_length": 256,
    "block_length": 128,
    "temperature": 0.0,
    "cfg_scale": 0.0,
    "remasking": "low_confidence",
}

DEFAULT_MODEL_NAME = "GSAI-ML/LLaDA-V"


class Llada_V_Local:
    """LLaDA-V wrapper compatible with ``run_vlm_sanity_grid.py``.

    The interface deliberately matches ``Qwen2_VL_Local``: a ``name`` attribute
    and a single ``evaluate_with_ti_list(ti_list, with_CoT=False) -> dict``
    method whose return shape is ``{"origin_output": ..., ("skill_sequence" |
    "format_error"): ...}``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        attn_implementation: str = "sdpa",
        device_map: str | dict | None = None,
        max_memory: dict | None = None,
        torch_dtype: str = "float16",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        generation_kwargs: dict | None = None,
        stopping_criteria: tuple[str, ...] = ("<|eot_id|>",),
        # Optional Fast-dLLM / dLLM-Cache speedups (off by default).
        use_fast_dllm: bool = False,
        use_dllm_cache: bool = False,
        # Optional Fast-dLLM kwargs threaded to model.generate(...). Off by
        # default to match the user's "baseline only, no speed tricks" spec.
        prefix_refresh_interval: int | None = None,
        threshold: int | None = None,
    ) -> None:
        # 1) Compat shims, then sys.path, then import llava.*.
        from vla_bench.llada_v_compat import ensure_compat

        ensure_compat()
        _ensure_llava_on_path()

        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: WPS433
        from llava.conversation import conv_templates  # noqa: WPS433
        from llava.mm_utils import process_images, tokenizer_image_token  # noqa: WPS433

        # Patch the upstream SigLIP loader to skip its HuggingFace download.
        # The LLaDA-V 16 GB checkpoint *already contains* every
        # ``model.vision_tower.*`` weight, so the vision tower only needs to
        # be constructed empty from its config — the outer model construction
        # below provides the right submodule shape, and our checkpoint loader
        # fills the weights. Going through ``SigLipVisionModel.from_pretrained``
        # instead trips over two transformers-5.x-only failure modes (meta-
        # device guard, ``_set_token_in_kwargs`` removed) and would just
        # redundantly download SigLIP weights that are about to be
        # overwritten.
        _patch_siglip_load_to_construct_from_config()

        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._conv_templates = conv_templates
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token

        self.name = "LLaDA_V"
        self.model_name = model_name
        self.stopping_criteria = list(stopping_criteria)
        self.use_fast_dllm = bool(use_fast_dllm)
        self.use_dllm_cache = bool(use_dllm_cache)

        merged_gen_kwargs = dict(DEFAULT_GEN_KWARGS)
        if generation_kwargs is not None:
            merged_gen_kwargs.update(generation_kwargs)
        self.generation_kwargs = merged_gen_kwargs
        self._extra_generate_kwargs: dict[str, Any] = {}
        if prefix_refresh_interval is not None:
            self._extra_generate_kwargs["prefix_refresh_interval"] = prefix_refresh_interval
        if threshold is not None:
            self._extra_generate_kwargs["threshold"] = threshold

        # 2) Load model + processor + tokenizer manually.
        #
        # We bypass ``llava.model.builder.load_pretrained_model`` because:
        #   * it hard-codes ``low_cpu_mem_usage=True``, which combined with
        #     transformers 5.x ``device_map="auto"`` dispatch left some of
        #     the (patched) SigLIP vision tower on the meta device with no
        #     offload backing — every forward then crashed with
        #     "Cannot copy out of meta tensor; no data!".
        #   * the LLaDA-V 16 GB checkpoint does not fit in 12 GB of fp16
        #     VRAM. We load it in 4-bit (NF4 via bitsandbytes) which
        #     compresses the LLM weights to ~5 GB, so the whole model
        #     (LLM + SigLIP + projector) fits on the GPU and there is no
        #     offload ambiguity at all. The vision tower runs in fp16.
        torch_dtype_obj = (
            torch.float16 if torch_dtype == "float16"
            else torch.bfloat16 if torch_dtype == "bfloat16"
            else torch.float32
        )
        self.tokenizer, self.model, self.image_processor = self._load_model(
            model_name=model_name,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype_obj,
            device_map=device_map,
            max_memory=max_memory,
            load_in_4bit=bool(load_in_4bit),
            load_in_8bit=bool(load_in_8bit) and not bool(load_in_4bit),
        )
        self.max_length = getattr(self.model.config, "max_position_embeddings", 16384)
        self.model.eval()

        # ``self.device`` is the input embedding's device. With device_map="auto"
        # this is wherever accelerate placed the embedding layer (usually GPU).
        try:
            embed = self.model.get_input_embeddings()
            self.device = embed.weight.device
        except Exception:
            self.device = next(self.model.parameters()).device

        # 3) Optional speed hooks (off by default).
        if self.use_fast_dllm:
            from llava.hooks.fast_dllm_hook import register_fast_dllm_hook
            register_fast_dllm_hook(self.model)
        elif self.use_dllm_cache:
            from llava.cache import dLLMCache, dLLMCacheConfig
            from llava.hooks import register_cache_LLaDA_V
            from dataclasses import asdict
            dLLMCache.new_instance(
                **asdict(dLLMCacheConfig(
                    prompt_interval_steps=25,
                    gen_interval_steps=7,
                    transfer_ratio=0.25,
                ))
            )
            register_cache_LLaDA_V(self.model, "model.layers")

    def get_name(self) -> str:
        return self.name

    @staticmethod
    def _reseed_siglip_position_ids(vision_tower) -> None:
        inner = getattr(vision_tower, "vision_tower", None)
        if inner is None:
            return
        try:
            embeddings = inner.vision_model.embeddings
        except AttributeError:
            return
        num_positions = int(embeddings.num_positions)
        device = embeddings.position_embedding.weight.device
        embeddings.position_ids = torch.arange(
            num_positions, device=device
        ).expand((1, -1))

    # ------------------------------------------------------------------
    # Manual checkpoint loader (bypasses load_pretrained_model)
    # ------------------------------------------------------------------
    @staticmethod
    def _load_model(
        *,
        model_name: str,
        attn_implementation: str,
        torch_dtype: torch.dtype,
        device_map: str | dict,
        max_memory: dict | None,
        load_in_4bit: bool,
        load_in_8bit: bool = False,
    ):
        """
        Load tokenizer, image processor, and the full ``LlavaLLaDAModelLM``.

        With ``load_in_4bit=True`` the LLM weights are NF4-quantised by
        bitsandbytes (the vision tower stays in fp16) and the whole model
        is placed on the GPU — no offloading.

        Otherwise the model is loaded on CPU and accelerate's
        ``dispatch_model`` is used to split parts to GPU/CPU.
        """
        from transformers import AutoTokenizer
        from llava.constants import (  # noqa: WPS433
            DEFAULT_IMAGE_PATCH_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.model.language_model.llava_llada import (  # noqa: WPS433
            LlavaLLaDAConfig,
            LlavaLLaDAModelLM,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        cfg = LlavaLLaDAConfig.from_pretrained(model_name)
        # Reset rope_scaling: transformers 5.x auto-populates it with a
        # new-format dict that the bundled ``_init_rope`` does not recognise.
        cfg.rope_scaling = None
        # Force ``use_cache=False``. The bundled LLaDA-V forward asserts
        # ``past_key_values is None and not use_cache`` (it's a masked
        # diffusion model — no KV cache), but it then tries to call
        # ``DynamicCache.from_legacy_cache`` which transformers 5.x removed.
        # Disabling ``use_cache`` keeps the diffusion path purely cache-free.
        cfg.use_cache = False

        from_pretrained_kwargs: dict[str, Any] = dict(
            config=cfg,
            attn_implementation=attn_implementation,
        )

        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig
            if load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=[
                        "vision_tower",
                        "vision_resampler",
                        "mm_projector",
                    ],
                )
            else:
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=[
                        "vision_tower",
                        "vision_resampler",
                        "mm_projector",
                    ],
                )
            from_pretrained_kwargs["quantization_config"] = quant_config
            from_pretrained_kwargs["device_map"] = (
                device_map if device_map is not None else "auto"
            )
            if max_memory is not None:
                from_pretrained_kwargs["max_memory"] = max_memory
            else:
                # Force the LLM weights through CPU first so we never need
                # the full fp16 checkpoint resident on the GPU (which is
                # smaller than the unquantised model).
                from_pretrained_kwargs["max_memory"] = {
                    0: "10GiB",
                    "cpu": "60GiB",
                }
        else:
            from_pretrained_kwargs["torch_dtype"] = torch_dtype
            # Pass device_map + max_memory through to from_pretrained so
            # transformers' loader streams shards directly to the chosen
            # devices instead of materialising the full fp16 model in CPU
            # RAM first.
            if device_map is not None:
                from_pretrained_kwargs["device_map"] = device_map
                if max_memory is not None:
                    from_pretrained_kwargs["max_memory"] = max_memory
                else:
                    from_pretrained_kwargs["max_memory"] = {
                        0: "9GiB",
                        "cpu": "60GiB",
                    }

        model = LlavaLLaDAModelLM.from_pretrained(
            model_name, **from_pretrained_kwargs
        ).eval()

        # Mirror the post-construct steps that ``load_pretrained_model``
        # would have done (image-patch tokens, embedding resize, vision
        # tower load).
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True,
            )
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        # Vision tower goes wherever the rest of the model went.
        if device_map is None:
            vision_tower.to(dtype=torch_dtype)
        elif torch.cuda.is_available():
            vision_tower.to(device="cuda:0", dtype=torch_dtype)
        else:
            vision_tower.to(dtype=torch_dtype)
        image_processor = vision_tower.image_processor

        # transformers 5.x's state-dict loader leaves non-persistent buffers
        # holding uninitialised memory (the bundled SigLIP code registers
        # ``position_ids`` with ``persistent=False``). Reseed them with the
        # values the SigLIP forward expects, otherwise the embedding lookup
        # crashes with "index out of range in self" on the first forward.
        Llada_V_Local._reseed_siglip_position_ids(vision_tower)

        # transformers' ``from_pretrained`` already dispatched the model
        # across GPU/CPU per ``device_map`` + ``max_memory`` for the fp16
        # path (see ``from_pretrained_kwargs`` above).

        return tokenizer, model, image_processor


    # ------------------------------------------------------------------
    # ti_list -> chat-template prompt + image tensor
    # ------------------------------------------------------------------
    def _build_prompt_from_ti_list(
        self, ti_list: list[list[str]]
    ) -> tuple[str, list[str]]:
        """
        Flatten ``ti_list`` into the user-message string LLaDA-V expects.

        Each ``("text", str)`` becomes a line of the user message; each
        ``("image", path)`` becomes an inline ``<image>`` placeholder. We
        accumulate paths in order — they will be processed by
        ``process_images`` and bound back to the placeholders by
        ``tokenizer_image_token``.
        """
        parts: list[str] = []
        image_paths: list[str] = []
        for kind, value in ti_list:
            if kind == "text":
                parts.append(str(value))
            elif kind == "image":
                parts.append(self._DEFAULT_IMAGE_TOKEN)
                image_paths.append(str(value))
            else:
                raise ValueError(f"Unknown ti_list kind={kind!r}")
        user_message = "\n".join(parts)
        return user_message, image_paths

    def _format_chat_prompt(self, user_message: str) -> str:
        conv = copy.deepcopy(self._conv_templates["llava_llada"])
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _prepare_inputs(self, ti_list):
        user_message, image_paths = self._build_prompt_from_ti_list(ti_list)
        prompt_text = self._format_chat_prompt(user_message)

        input_ids = self._tokenizer_image_token(
            prompt_text,
            self.tokenizer,
            self._IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        if image_paths:
            pil_images = [Image.open(p).convert("RGB") for p in image_paths]
            image_tensor = self._process_images(
                pil_images, self.image_processor, self.model.config
            )
            if isinstance(image_tensor, list):
                image_tensor = [
                    t.to(dtype=torch.float16, device=self.device)
                    for t in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(
                    dtype=torch.float16, device=self.device
                )
            image_sizes = [img.size for img in pil_images]
        else:
            image_tensor = None
            image_sizes = None

        return {
            "input_ids": input_ids,
            "image_tensor": image_tensor,
            "image_sizes": image_sizes,
            "prompt_text": prompt_text,
        }

    # ------------------------------------------------------------------
    # Generation + JSON-from-fence parsing
    # ------------------------------------------------------------------
    def evaluate_with_ti_list(self, ti_list, with_CoT: bool = False) -> dict:  # noqa: N803
        prepared = self._prepare_inputs(ti_list)

        gen_kwargs = dict(self.generation_kwargs)
        gen_length = int(gen_kwargs["gen_length"])

        with torch.inference_mode():
            output_ids = self.model.generate(
                prepared["input_ids"],
                images=prepared["image_tensor"],
                image_sizes=prepared["image_sizes"],
                tokenizer=self.tokenizer,
                stopping_criteria=list(self.stopping_criteria),
                **gen_kwargs,
                **self._extra_generate_kwargs,
            )

        # ``output_ids`` shape: (1, prompt_emb_len + gen_length [+ suffix_len]).
        # The first ``output_ids.shape[1] - gen_length`` positions correspond
        # to prompt embeddings (left as MASK token id in the output tensor).
        # Slice off the prompt-mask region to get just the generated tokens.
        prompt_pad_len = max(0, output_ids.shape[1] - gen_length)
        gen_ids = output_ids[:, prompt_pad_len:]

        # Decode without special tokens to drop trailing <|eot_id|> and the
        # MASK placeholders that may persist if early-stopping fired.
        full_decoded = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        gen_decoded = self.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True
        )[0]

        # Prefer the generation slice; if that's empty (rare), fall back to
        # the full decoded text.
        raw = gen_decoded if gen_decoded.strip() else full_decoded

        output: dict[str, Any] = {"origin_output": raw}
        try:
            json_data = raw.split("```json")[1].split("```")[0]
            output["skill_sequence"] = json.loads(json_data)
        except Exception:
            output["format_error"] = "format_error"
        return output
