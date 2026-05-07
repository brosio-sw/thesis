"""
Image-mode helpers for the VLABench VLM sanity grid.

The official pipeline (``VLABench.evaluation.model.vlm.base.get_ti_list``) sends
*both* the original four-view image (``input.png``) and the numbered/segmented
image (``input_mask.png``) to the model. We support two modes:

  * ``"original_plus_numbered"`` — the official path, unchanged.
  * ``"numbered_only"`` — drop the original image; keep only the numbered one.

The transform is performed *after* ``get_ti_list`` returns, so the textual
chat-template the model sees is always the official one. Only the image
attachments differ.
"""

from __future__ import annotations

from typing import Iterable

# Keep these in sync with ``get_ti_list`` text labels for the English prompt.
_NUMBERED_LABEL_EN = "Input picture with numbered tags"
_ORIGINAL_LABEL_EN = "Input picture"
_NUMBERED_SHOT_LABEL_EN = "input picture with numbered tags"
_ORIGINAL_SHOT_LABEL_EN = "input picture:"


def filter_ti_list_by_image_mode(ti_list, image_mode: str) -> list:
    """
    Return a new ti_list (text/image pair list) consistent with `image_mode`.

    ``"original_plus_numbered"``: pass-through (no change).
    ``"numbered_only"``: drop the *original* (non-numbered) images and the
        ``"Input picture"``/``"Example N input picture:"`` text labels that
        introduce them, for both the test image and any few-shot images.

    We do not mutate the input list.
    """
    if image_mode == "original_plus_numbered":
        return list(ti_list)
    if image_mode != "numbered_only":
        raise ValueError(
            f"Unknown image_mode={image_mode!r}. "
            "Expected 'original_plus_numbered' or 'numbered_only'."
        )

    out = []
    skip_next_image = False
    for kind, value in ti_list:
        if kind == "text":
            text_l = value.strip().lower()
            if text_l in {
                _ORIGINAL_LABEL_EN.lower(),
            } or text_l.startswith("example ") and text_l.endswith(_ORIGINAL_SHOT_LABEL_EN):
                # Drop this label and the *next* image entry that follows it.
                skip_next_image = True
                continue
            out.append([kind, value])
        elif kind == "image":
            if skip_next_image:
                skip_next_image = False
                continue
            out.append([kind, value])
        else:
            out.append([kind, value])
    return out


def collect_image_paths(ti_list) -> list[str]:
    """Return all image paths actually attached to a ti_list, in order."""
    return [v for kind, v in ti_list if kind == "image"]


def humanize_image_paths(paths: Iterable[str], dataset_root: str | None = None) -> list[str]:
    """Optionally rebase paths under `dataset_root` to keep logs short."""
    if dataset_root is None:
        return list(paths)
    out = []
    for p in paths:
        sp = str(p)
        out.append(sp[len(dataset_root):].lstrip("/") if sp.startswith(dataset_root) else sp)
    return out
