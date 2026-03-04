from __future__ import annotations

import torch
from typing import Any

from .gpu_ops import gpu_dilate, gpu_gaussian_blur


class MaskExpandBorder:
    """
    A ComfyUI node that expands the borders of a mask using morphological dilation.
    """

    RETURN_TYPES: tuple[str, ...] = ("MASK",)
    RETURN_NAMES: tuple[str, ...] = ("expanded_mask",)
    FUNCTION: str = "expand_mask_border"
    CATEGORY: str = "TryVariant.ai/mask"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "mask": ("MASK",),
                "expand_pixels": ("INT", {
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
            }
        }

    @torch.inference_mode()
    def expand_mask_border(self, mask: torch.Tensor, expand_pixels: int, iterations: int, kernel_shape: str) -> tuple[torch.Tensor]:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        result: torch.Tensor = gpu_dilate(mask, expand_pixels, expand_pixels, kernel_shape, iterations)
        return (result,)


class MaskExpandBorderAdvanced:
    """
    Advanced version with additional options for mask border expansion.
    """

    RETURN_TYPES: tuple[str, ...] = ("MASK",)
    RETURN_NAMES: tuple[str, ...] = ("expanded_mask",)
    FUNCTION: str = "expand_mask_border_advanced"
    CATEGORY: str = "TryVariant.ai/mask"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "mask": ("MASK",),
                "expand_pixels": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "method": (["dilation", "gaussian_blur", "distance_transform"], {
                    "default": "dilation"
                }),
                "kernel_shape": (["ellipse", "rectangle", "cross"], {
                    "default": "ellipse"
                }),
                "feather_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
            }
        }

    @torch.inference_mode()
    def expand_mask_border_advanced(self, mask: torch.Tensor, expand_pixels: int, method: str, kernel_shape: str, feather_amount: float) -> tuple[torch.Tensor]:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        device: torch.device = mask.device

        result: torch.Tensor
        if method == "dilation":
            result = gpu_dilate(mask, expand_pixels, expand_pixels, kernel_shape, iterations=1)

        elif method == "gaussian_blur":
            blur_size: int = expand_pixels * 2 + 1
            blurred: torch.Tensor = gpu_gaussian_blur(mask, blur_size)
            result = (blurred > 0.498).float()

        elif method == "distance_transform":
            # CPU fallback — no GPU equivalent, rarely used
            import cv2
            import numpy as np
            results: list[torch.Tensor] = []
            for i in range(mask.shape[0]):
                mask_np: np.ndarray = (mask[i].cpu().numpy() * 255).astype(np.uint8)
                dist: np.ndarray = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
                expanded: np.ndarray = ((dist > 0) | (dist <= expand_pixels)).astype(np.uint8) * 255
                results.append(torch.from_numpy(expanded.astype(np.float32) / 255.0))
            result = torch.stack(results).to(device)
        else:
            result = mask

        # Apply feathering if requested
        if feather_amount > 0:
            feather_size: int = int(feather_amount * 2) * 2 + 1
            result = gpu_gaussian_blur(result, feather_size, sigma=feather_amount)

        return (result,)
