from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Any


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

    def expand_mask_border(self, mask: torch.Tensor, expand_pixels: int, iterations: int, kernel_shape: str) -> tuple[torch.Tensor]:
        # Ensure mask is in the right format
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add batch dimension

        batch_size: int = mask.shape[0]
        height: int = mask.shape[1]
        width: int = mask.shape[2]
        expanded_masks: list[torch.Tensor] = []

        # Create morphological kernel
        kernel_size: int = expand_pixels * 2 + 1

        kernel: np.ndarray
        if kernel_shape == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape == "rectangle":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        else:  # cross
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

        # Process each mask in the batch
        for i in range(batch_size):
            # Convert mask to numpy array (0-255 range)
            mask_np: np.ndarray = (mask[i].cpu().numpy() * 255).astype(np.uint8)

            # Apply morphological dilation
            expanded_mask_np: np.ndarray = cv2.dilate(mask_np, kernel, iterations=iterations)

            # Convert back to tensor (0-1 range)
            expanded_mask: torch.Tensor = torch.from_numpy(expanded_mask_np.astype(np.float32) / 255.0)

            # Move to same device as input
            expanded_mask = expanded_mask.to(mask.device)
            expanded_masks.append(expanded_mask)

        # Stack all masks back together
        result: torch.Tensor = torch.stack(expanded_masks, dim=0)

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

    def expand_mask_border_advanced(self, mask: torch.Tensor, expand_pixels: int, method: str, kernel_shape: str, feather_amount: float) -> tuple[torch.Tensor]:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        batch_size: int = mask.shape[0]
        height: int = mask.shape[1]
        width: int = mask.shape[2]
        expanded_masks: list[torch.Tensor] = []

        for i in range(batch_size):
            mask_np: np.ndarray = (mask[i].cpu().numpy() * 255).astype(np.uint8)

            expanded_mask_np: np.ndarray
            if method == "dilation":
                kernel_size: int = expand_pixels * 2 + 1
                kernel: np.ndarray
                if kernel_shape == "ellipse":
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                elif kernel_shape == "rectangle":
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                else:  # cross
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

                expanded_mask_np = cv2.dilate(mask_np, kernel, iterations=1)

            elif method == "gaussian_blur":
                # Create a blurred version and threshold it
                blurred: np.ndarray = cv2.GaussianBlur(mask_np, (expand_pixels * 2 + 1, expand_pixels * 2 + 1), 0)
                _, expanded_mask_np = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

            elif method == "distance_transform":
                # Use distance transform for smoother expansion
                dist_transform: np.ndarray = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
                expanded_mask_np = ((dist_transform > 0) | (dist_transform <= expand_pixels)).astype(np.uint8) * 255

            # Apply feathering if requested
            if feather_amount > 0:
                feather_kernel_size: int = int(feather_amount * 2) * 2 + 1
                expanded_mask_np = cv2.GaussianBlur(expanded_mask_np, (feather_kernel_size, feather_kernel_size), feather_amount)

            # Convert back to tensor
            expanded_mask: torch.Tensor = torch.from_numpy(expanded_mask_np.astype(np.float32) / 255.0)
            expanded_mask = expanded_mask.to(mask.device)
            expanded_masks.append(expanded_mask)

        result: torch.Tensor = torch.stack(expanded_masks, dim=0)
        return (result,)
