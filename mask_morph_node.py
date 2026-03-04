from __future__ import annotations

import torch
from typing import Any

from .gpu_ops import gpu_morph


class MaskMorphNode:
    """
    A ComfyUI node for pixel-wise mask expansion and shrinking using morphological operations.
    Supports both expand (dilate) and shrink (erode) in a single node.
    """

    RETURN_TYPES: tuple[str, ...] = ("MASK",)
    RETURN_NAMES: tuple[str, ...] = ("mask",)
    FUNCTION: str = "morph_mask"
    CATEGORY: str = "TryVariant.ai/mask"
    DESCRIPTION: str = "Expand (positive pixels) or shrink (negative pixels) a mask with axis control. Supports uniform, horizontal, vertical, or custom axis-based morphing."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "pixels": ("INT", {
                    "default": 5,
                    "min": -100,
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
                "axis_mode": (["uniform", "horizontal", "vertical", "custom"], {
                    "default": "uniform"
                }),
            },
            "optional": {
                "mask": ("MASK",),
                "image": ("IMAGE",),
                "horizontal_pixels": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "vertical_pixels": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    @torch.inference_mode()
    def morph_mask(self, pixels: int, iterations: int, kernel_shape: str, axis_mode: str, mask: torch.Tensor | None = None, image: torch.Tensor | None = None, horizontal_pixels: int = 0, vertical_pixels: int = 0) -> tuple[torch.Tensor]:
        if mask is None and image is None:
            raise ValueError("Either 'mask' or 'image' input must be provided")

        if image is not None:
            mask = self._image_to_mask(image)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Determine h/v pixels based on axis mode
        h_pixels: int
        v_pixels: int
        if axis_mode == "uniform":
            h_pixels = pixels
            v_pixels = pixels
        elif axis_mode == "horizontal":
            h_pixels = pixels
            v_pixels = 0
        elif axis_mode == "vertical":
            h_pixels = 0
            v_pixels = pixels
        else:  # custom
            h_pixels = horizontal_pixels
            v_pixels = vertical_pixels

        if h_pixels == 0 and v_pixels == 0:
            return (mask,)

        # Determine operation per axis
        h_op: str | None = "dilate" if h_pixels > 0 else "erode" if h_pixels < 0 else None
        v_op: str | None = "dilate" if v_pixels > 0 else "erode" if v_pixels < 0 else None

        abs_h: int = abs(h_pixels)
        abs_v: int = abs(v_pixels)

        # kernel_h = height radius (vertical), kernel_w = width radius (horizontal)
        result: torch.Tensor
        if h_op == v_op and h_op is not None:
            # Same operation on both axes — single call with selected kernel shape
            result = gpu_morph(mask, abs_v, abs_h, kernel_shape, h_op, iterations)
        else:
            # Different operations per axis — separate calls with rectangle kernels
            result = mask
            if h_op is not None:
                result = gpu_morph(result, 0, abs_h, "rectangle", h_op, iterations)
            if v_op is not None:
                result = gpu_morph(result, abs_v, 0, "rectangle", v_op, iterations)

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
