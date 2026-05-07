"""
Parsing diagnostics for VLM outputs.

We keep the *official* parsing untouched — every supported VLABench wrapper
(e.g. ``Qwen2_VL``) returns a dict that is one of:

  * ``{"origin_output": ..., "skill_sequence": <list>}``  — parsed OK
  * ``{"origin_output": ..., "format_error": "format_error"}`` — parser failed

This module only adds *diagnostics* on top: `summarize_output(...)` extracts a
flat record with `parse_success`, `raw_output`, `parsed_output`, and an
`error_type`/`error_message` if the parser failed.
"""

from __future__ import annotations

from typing import Any


def summarize_output(output: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize an official VLM-wrapper output dict into a diagnostics record.
    """
    if not isinstance(output, dict):
        return {
            "raw_output": str(output),
            "parsed_output": None,
            "parse_success": False,
            "error_type": "non_dict_return",
            "error_message": f"Wrapper returned a {type(output).__name__}",
        }

    raw = output.get("origin_output")
    if "skill_sequence" in output:
        return {
            "raw_output": raw,
            "parsed_output": {"skill_sequence": output["skill_sequence"]},
            "parse_success": True,
            "error_type": None,
            "error_message": None,
        }

    # Mirror VLMEvaluator.check_filled_output: anything with format_error
    # counts as a (recorded) parser failure.
    if "format_error" in output:
        return {
            "raw_output": raw,
            "parsed_output": None,
            "parse_success": False,
            "error_type": "format_error",
            "error_message": str(output.get("format_error")),
        }

    return {
        "raw_output": raw,
        "parsed_output": None,
        "parse_success": False,
        "error_type": "missing_keys",
        "error_message": f"Wrapper output missing skill_sequence/format_error: keys={list(output)}",
    }
