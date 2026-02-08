from __future__ import annotations

import torch
import numpy as np
import cv2
from typing import Any


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

    def shrink_mask(self, shrink_pixels: int, iterations: int, kernel_shape: str, mask: torch.Tensor | None = None, image: torch.Tensor | None = None) -> tuple[torch.Tensor]:
        # Validate inputs
        if mask is None and image is None:
            raise ValueError("Either 'mask' or 'image' input must be provided")

        # Convert image to mask if provided
        if image is not None:
            mask = self._image_to_mask(image)

        # Keep track of device
        device: torch.device = mask.device

        # Ensure mask is in the right format
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add batch dimension

        batch_size: int = mask.shape[0]
        height: int = mask.shape[1]
        width: int = mask.shape[2]
        shrunk_masks: list[torch.Tensor] = []

        # Create morphological kernel
        kernel_size: int = shrink_pixels * 2 + 1

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

            # Apply morphological erosion (shrinking)
            shrunk_mask_np: np.ndarray = cv2.erode(mask_np, kernel, iterations=iterations)

            # Convert back to tensor (0-1 range)
            shrunk_mask: torch.Tensor = torch.from_numpy(shrunk_mask_np.astype(np.float32) / 255.0)

            # Move to same device as input
            shrunk_mask = shrunk_mask.to(device)
            shrunk_masks.append(shrunk_mask)

        # Stack all masks back together
        result: torch.Tensor = torch.stack(shrunk_masks, dim=0)

        return (result,)

    def _image_to_mask(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convert an IMAGE tensor to a MASK tensor.
        Uses grayscale conversion for RGB images.
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        batch_size: int = image.shape[0]
        masks: list[np.ndarray] = []

        for i in range(batch_size):
            img: np.ndarray = image[i].cpu().numpy()

            # Convert to grayscale if RGB/RGBA
            gray: np.ndarray
            if img.shape[2] == 3:
                gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            elif img.shape[2] == 4:
                gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            elif img.shape[2] == 1:
                gray = img[:, :, 0]
            else:
                raise ValueError(f"Unsupported number of channels: {img.shape[2]}")

            masks.append(gray)

        # Convert to tensor
        mask_tensor: torch.Tensor = torch.from_numpy(np.stack(masks).astype(np.float32))
        return mask_tensor.to(image.device)
