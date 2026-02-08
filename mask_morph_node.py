from __future__ import annotations

import torch
import numpy as np
import cv2
from typing import Any


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

    def morph_mask(self, pixels: int, iterations: int, kernel_shape: str, axis_mode: str, mask: torch.Tensor | None = None, image: torch.Tensor | None = None, horizontal_pixels: int = 0, vertical_pixels: int = 0) -> tuple[torch.Tensor]:
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
        morphed_masks: list[torch.Tensor] = []

        # Determine kernel dimensions based on axis mode
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

        # Check if any operation is needed
        if h_pixels == 0 and v_pixels == 0:
            return (mask,)

        # Determine kernel size (width, height)
        kernel_width: int = abs(h_pixels) * 2 + 1 if h_pixels != 0 else 1
        kernel_height: int = abs(v_pixels) * 2 + 1 if v_pixels != 0 else 1

        # Determine if we need separate operations for expand/shrink on different axes
        h_operation: int | None = cv2.MORPH_DILATE if h_pixels > 0 else cv2.MORPH_ERODE if h_pixels < 0 else None
        v_operation: int | None = cv2.MORPH_DILATE if v_pixels > 0 else cv2.MORPH_ERODE if v_pixels < 0 else None

        # Process each mask in the batch
        for i in range(batch_size):
            # Convert mask to numpy array (0-255 range)
            mask_np: np.ndarray = (mask[i].cpu().numpy() * 255).astype(np.uint8)

            morphed_mask_np: np.ndarray
            # If both axes have same operation, do it in one step
            if h_operation == v_operation and h_operation is not None:
                # Create morphological kernel
                kernel: np.ndarray
                if kernel_shape == "ellipse":
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))
                elif kernel_shape == "rectangle":
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
                else:  # cross
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_width, kernel_height))

                morphed_mask_np = cv2.dilate(mask_np, kernel, iterations=iterations) if h_operation == cv2.MORPH_DILATE else cv2.erode(mask_np, kernel, iterations=iterations)
            else:
                # Apply operations separately for each axis
                morphed_mask_np = mask_np.copy()

                # Horizontal operation
                if h_operation is not None:
                    h_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
                    morphed_mask_np = cv2.dilate(morphed_mask_np, h_kernel, iterations=iterations) if h_operation == cv2.MORPH_DILATE else cv2.erode(morphed_mask_np, h_kernel, iterations=iterations)

                # Vertical operation
                if v_operation is not None:
                    v_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
                    morphed_mask_np = cv2.dilate(morphed_mask_np, v_kernel, iterations=iterations) if v_operation == cv2.MORPH_DILATE else cv2.erode(morphed_mask_np, v_kernel, iterations=iterations)

            # Convert back to tensor (0-1 range)
            morphed_mask: torch.Tensor = torch.from_numpy(morphed_mask_np.astype(np.float32) / 255.0)

            # Move to same device as input
            morphed_mask = morphed_mask.to(device)
            morphed_masks.append(morphed_mask)

        # Stack all masks back together
        result: torch.Tensor = torch.stack(morphed_masks, dim=0)

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
