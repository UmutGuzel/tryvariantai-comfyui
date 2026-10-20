"""Pure logic behind the Head Segmenter node: checkpoint header check, size rule,
batched segmentation.

Imports only json, torch and safetensors (no ComfyUI) so it runs under plain pytest.
The model contract (`predict_tensor`) lives in the vendored `model.py`; nothing here
re-implements the resize.
"""
from __future__ import annotations

import json
from typing import Any

import torch
from safetensors import safe_open

CHECKPOINT_FORMAT = "selfieseg/1"


def check_header(path: str) -> dict[str, Any]:
    """Return the parsed `meta` dict from a selfieseg checkpoint's safetensors header.

    Raises ValueError naming the file and the format found when `format` is missing
    or is not `selfieseg/1`.
    """
    with safe_open(str(path), framework="pt", device="cpu") as f:
        header = f.metadata() or {}
    fmt = header.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: not a {CHECKPOINT_FORMAT} checkpoint (format={fmt!r})"
        )
    return json.loads(header.get("meta", "{}"))


def read_trained_size(meta: dict[str, Any]) -> int | None:
    """Training resolution from `meta["train_config"]["size"]` (older runs) or
    `meta["job"]["size"]` (newer runs). None when neither is present."""
    for key in ("train_config", "job"):
        section = meta.get(key)
        if not isinstance(section, dict):
            continue
        size = section.get("size")
        if isinstance(size, int) and not isinstance(size, bool):
            return size
    return None


def pick_size(
    trained_size: int | None, native_hw: tuple[int, int], requested: int
) -> tuple[int, int]:
    """Inference resolution (h, w) the image is stretched to before the net runs.

    `requested == 0` means auto: the trained size when the checkpoint records one,
    otherwise the graph's native input size. Any other value is used as a square.
    """
    if requested < 0:
        raise ValueError(f"size must be >= 0, got {requested}")
    if requested == 0:
        if trained_size is not None:
            return (trained_size, trained_size)
        return (int(native_hw[0]), int(native_hw[1]))
    return (requested, requested)


def segment(
    model: Any,
    image_bhwc: torch.Tensor,
    size_hw: tuple[int, int],
    threshold: float,
    output_binary: bool,
    invert: bool,
) -> torch.Tensor:
    """B×H×W×3 float32 in [0, 1] (any device) → B×H×W float32 mask on the model's device.

    The whole batch goes through one `predict_tensor` call, which stretches to
    `size_hw`, runs the net, and resizes the probability back to H×W. Channel 0 of
    the B×1×H×W output is the head probability. A 4th input channel (alpha) is ignored.
    """
    if image_bhwc.dim() != 4 or image_bhwc.shape[-1] < 3:
        raise ValueError(
            f"expected a B×H×W×3 image tensor, got shape {tuple(image_bhwc.shape)}"
        )
    device = next(model.parameters()).device
    x = image_bhwc[..., :3].permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    prob = model.predict_tensor(x, size=size_hw)  # B×1×H×W
    mask = prob[:, 0].to(torch.float32)
    if invert:
        mask = 1.0 - mask
    if output_binary:
        mask = (mask > threshold).to(torch.float32)
    return mask
