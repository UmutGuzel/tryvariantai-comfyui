from __future__ import annotations

import torch
from typing import Any


@torch.inference_mode()
def detect_white_areas_gpu(image: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    GPU white detection with color uniformity check.

    Detects only actual white/gray areas by ensuring:
    1. All RGB channels are above threshold (brightness check)
    2. All RGB channels are similar to each other (uniformity check)

    Args:
        image: (B, H, W, C) float tensor [0, 1]
        threshold: Float threshold for white detection (0.0-1.0)

    Returns:
        (B, H, W) float tensor with white areas marked as 1.0, others as 0.0
    """
    # Brightness: if min channel >= threshold, all channels are above it
    rgb: torch.Tensor = image[..., :3]
    max_ch: torch.Tensor = rgb.amax(dim=-1)
    min_ch: torch.Tensor = rgb.amin(dim=-1)
    is_bright: torch.Tensor = min_ch >= threshold

    # Uniformity: RGB channels must be similar (within 5% of each other)
    is_uniform: torch.Tensor = (max_ch - min_ch) < 0.05

    # White detection: must be both bright AND uniform
    white_mask: torch.Tensor = (is_bright & is_uniform).float()
    return white_mask


class WhiteToTransparentNode:
    """
    A ComfyUI node that makes solid white parts of an image transparent.
    Outputs an RGBA image with transparency where white areas are detected.
    """

    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("rgba_image", "transparency_mask")
    FUNCTION: str = "make_white_transparent"
    CATEGORY: str = "TryVariant.ai/postprocessing"
    DESCRIPTION: str = "Makes white parts of an image transparent (binary 0 or 1). Optionally use a mask to control where white detection is applied."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
            "optional": {
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    @torch.inference_mode()
    def make_white_transparent(self, image: torch.Tensor, threshold: float = 0.95, mask: torch.Tensor | None = None, invert_mask: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        device: torch.device = image.device

        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size: int = image.shape[0]
        height: int = image.shape[1]
        width: int = image.shape[2]

        # Detect white areas on GPU
        white_mask: torch.Tensor = detect_white_areas_gpu(image, threshold)

        # Apply optional mask
        transparency_mask: torch.Tensor
        if mask is not None:
            msk: torch.Tensor = mask
            if msk.dim() == 2:
                msk = msk.unsqueeze(0)

            # Broadcast single mask to batch
            if msk.shape[0] < batch_size:
                msk = msk[0:1].expand(batch_size, -1, -1)

            if invert_mask:
                msk = 1.0 - msk

            # Combine white detection with mask, then binarize
            transparency_mask = torch.where(white_mask * msk > 0.5, 1.0, 0.0)
        else:
            transparency_mask = white_mask

        # Create RGBA image
        rgba: torch.Tensor = torch.zeros(batch_size, height, width, 4, device=device, dtype=image.dtype)
        rgba[..., :3] = image[..., :3]
        rgba[..., 3] = 1.0 - transparency_mask

        return (rgba, transparency_mask)


class SimpleWhiteDetectorNode:
    """
    A simple node that detects white areas in an image and creates a mask.
    """

    RETURN_TYPES: tuple[str, ...] = ("MASK",)
    RETURN_NAMES: tuple[str, ...] = ("white_areas_mask",)
    FUNCTION: str = "detect_white"
    CATEGORY: str = "TryVariant.ai/postprocessing"
    DESCRIPTION: str = "Detects white areas in an image and creates a mask."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }

    @torch.inference_mode()
    def detect_white(self, image: torch.Tensor, threshold: float = 0.95) -> tuple[torch.Tensor]:
        if image.dim() == 3:
            image = image.unsqueeze(0)

        mask: torch.Tensor = detect_white_areas_gpu(image, threshold)
        return (mask,)
