"""
Qwen2-VL loader for the VLABench VLM sanity grid.

The official VLABench wrapper hardcodes ``qwen/Qwen2-VL-7B-Instruct`` (downloaded
via ``modelscope``) and forces ``flash_attention_2``. Our local box has 12 GB
VRAM (RTX 4070 Ti), which is too small for the 7B variant in bf16, and we
don't want a hard dependency on ``flash-attn``.

This module subclasses the official ``Qwen2_VL`` wrapper so all of:

  * ``evaluate(...)`` (chat-template build, generation, JSON-from-fence parsing)
  * ``build_prompt_with_tilist(...)``
  * ``get_name()``

are inherited *unchanged* (the official prompt and parsing are preserved).
We only override ``__init__`` to:

  * load from HuggingFace (no modelscope) so all caches stay under
    ``~/.cache/huggingface``;
  * accept an explicit model name (default 2B-Instruct, which fits in 12 GB);
  * pick a safe attention backend (``sdpa``);
  * be configurable for ``device_map`` / ``dtype``.
"""

from __future__ import annotations

import torch

from vla_bench import vlabench_paths

vlabench_paths.setup()

from VLABench.evaluation.model.vlm.qwen_vl import Qwen2_VL  # noqa: E402


class Qwen2_VL_Local(Qwen2_VL):
    """Qwen2-VL wrapper loaded from HuggingFace — preserves official evaluate()."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        # Skip the official Qwen2_VL.__init__ entirely (it forces modelscope +
        # flash_attention_2). We replicate just the bits we need, then call
        # BaseVLM.__init__ at the end so `self.name` is set.
        from transformers import (
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
        )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        load_kwargs: dict = {
            "torch_dtype": dtype,
            "device_map": device_map,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_name)

        # BaseVLM.__init__ only does ``self.name = self.get_name()``.
        from VLABench.evaluation.model.vlm.base import BaseVLM

        BaseVLM.__init__(self)

    def get_name(self) -> str:
        return "Qwen2_VL"

    # NOTE: We override `evaluate` here only to thread `max_new_tokens` and
    # to avoid hardcoding 128. The textual prompt build, ti_list construction,
    # JSON-from-fence parsing and output dict shape are identical to the
    # official wrapper, so VLABench's parsing/scoring path stays unchanged.
    def evaluate(self, input_dict, language, with_CoT=False):  # noqa: N803
        import json
        from qwen_vl_utils import process_vision_info
        from VLABench.evaluation.model.vlm.base import get_ti_list

        ti_list = get_ti_list(input_dict, language, with_CoT=with_CoT)
        content = self.build_prompt_with_tilist(ti_list)

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        output: dict = {"origin_output": output_text[0]}
        try:
            json_data = output_text[0].split("```json")[1].split("```")[0]
            output["skill_sequence"] = json.loads(json_data)
        except Exception:
            output["format_error"] = "format_error"
        return output

    def evaluate_with_ti_list(self, ti_list, with_CoT: bool = False) -> dict:
        """
        Drop-in alternative to ``evaluate`` that takes a *pre-built* ti_list,
        so callers can apply image-mode filters before the model sees it.

        Mirrors ``Qwen2_VL.evaluate`` byte-for-byte except we skip rebuilding
        the ti_list from the input_dict.
        """
        import json
        from qwen_vl_utils import process_vision_info

        content = self.build_prompt_with_tilist(ti_list)
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        output: dict = {"origin_output": output_text[0]}
        try:
            json_data = output_text[0].split("```json")[1].split("```")[0]
            output["skill_sequence"] = json.loads(json_data)
        except Exception:
            output["format_error"] = "format_error"
        return output
