from __future__ import annotations

import torch
from typing import Any

from .gpu_ops import gpu_erode


class MaskShrinkNode:
    """
    A ComfyUI node that shrinks (erodes) a mask.
    Can accept either a MASK or IMAGE input and outputs a shrunk MASK.
    """

    RETURN_TYPES: tuple[str, ...] = ("MASK",)
    RETURN_NAMES: tuple[str, ...] = ("shrunk_mask",)
    FUNCTION: str = "shrink_mask"
    CATEGORY: str = "TryVariant.ai/mask"
    DESCRIPTION: str = "Shrinks (erodes) a mask. Can accept MASK or IMAGE input."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "shrink_pixels": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "kernel_shape": (["ellipse", "rectangle", "cross"], {
                    "default": "ellipse"
                }),
            },
            "optional": {
                "mask": ("MASK",),
                "image": ("IMAGE",),
            }
        }

    @torch.inference_mode()
    def shrink_mask(self, shrink_pixels: int, iterations: int, kernel_shape: str, mask: torch.Tensor | None = None, image: torch.Tensor | None = None) -> tuple[torch.Tensor]:
        if mask is None and image is None:
            raise ValueError("Either 'mask' or 'image' input must be provided")

        if image is not None:
            mask = self._image_to_mask(image)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        result: torch.Tensor = gpu_erode(mask, shrink_pixels, shrink_pixels, kernel_shape, iterations)
        return (result,)

    def _image_to_mask(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an IMAGE tensor to a MASK tensor using luminance."""
        if image.dim() == 3:
            image = image.unsqueeze(0)

        num_channels: int = image.shape[-1]
        mask: torch.Tensor
        if num_channels >= 3:
            mask = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        elif num_channels == 1:
            mask = image[..., 0]
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}")

        return mask
