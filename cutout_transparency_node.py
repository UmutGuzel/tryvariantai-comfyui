from __future__ import annotations

import torch
import numpy as np
from typing import Any


def detect_white_areas(img_np: np.ndarray, threshold: float) -> np.ndarray:
    """
    Shared white detection logic with color uniformity check.

    Detects only actual white/gray areas by ensuring:
    1. All RGB channels are above threshold (brightness check)
    2. All RGB channels are similar to each other (uniformity check)

    This prevents accidentally removing bright colored areas (like bright red, blue, green)
    and ensures image integrity by only targeting genuine white/gray regions.

    Args:
        img_np: Numpy array of images (B, H, W, C)
        threshold: Float threshold for white detection (0.0-1.0)

    Returns:
        Numpy array of masks (B, H, W) with white areas marked as 1, others as 0
    """
    if len(img_np.shape) == 4:
        batch_size: int = img_np.shape[0]
    else:
        img_np = np.expand_dims(img_np, 0)
        batch_size = 1

    masks: list[np.ndarray] = []

    for i in range(batch_size):
        img: np.ndarray = img_np[i]

        # Brightness check: all RGB channels must be above threshold
        is_bright: np.ndarray = np.all(img >= threshold, axis=2)

        # Uniformity check: RGB channels must be similar (within 5% of each other)
        max_channel: np.ndarray = np.max(img, axis=2)
        min_channel: np.ndarray = np.min(img, axis=2)
        channel_diff: np.ndarray = max_channel - min_channel
        is_uniform: np.ndarray = channel_diff < 0.05

        # White detection: must be both bright AND uniform
        white_mask: np.ndarray = np.where(is_bright & is_uniform, 1.0, 0.0).astype(np.float32)
        masks.append(white_mask)

    return np.stack(masks)


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

    def make_white_transparent(self, image: torch.Tensor, threshold: float = 0.95, mask: torch.Tensor | None = None, invert_mask: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep tensors on their original device
        device: torch.device = image.device

        # Ensure correct dimensions
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        batch_size: int = image.shape[0]

        # Convert to numpy for processing
        img_np: np.ndarray = image.cpu().numpy()

        # Handle optional mask
        mask_np: np.ndarray | None = None
        if mask is not None:
            mask_np = mask.cpu().numpy()
            # Ensure mask has batch dimension
            if len(mask_np.shape) == 2:
                mask_np = np.expand_dims(mask_np, 0)

        # Detect white areas using shared logic
        white_masks: np.ndarray = detect_white_areas(img_np, threshold)

        # Process each image in batch
        processed_images: list[np.ndarray] = []
        final_masks: list[np.ndarray] = []

        for i in range(batch_size):
            # Get current image
            img: np.ndarray = img_np[i]
            white_mask: np.ndarray = white_masks[i]

            # Apply mask if provided
            transparency_mask: np.ndarray
            if mask_np is not None:
                msk: np.ndarray = mask_np[i] if i < mask_np.shape[0] else mask_np[0]

                # Invert mask if requested
                if invert_mask:
                    msk = 1.0 - msk

                # Combine white detection with mask (result remains binary 0 or 1)
                transparency_mask = white_mask * msk
                # Ensure binary values after mask combination
                transparency_mask = np.where(transparency_mask > 0.5, 1.0, 0.0)
            else:
                # Use white detection directly (already binary 0 or 1)
                transparency_mask = white_mask

            # Create RGBA image
            rgba: np.ndarray = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float32)

            # Copy RGB channels
            rgba[:, :, :3] = img

            # Set alpha channel (1 = opaque, 0 = transparent)
            rgba[:, :, 3] = 1.0 - transparency_mask

            processed_images.append(rgba)
            final_masks.append(transparency_mask)

        # Convert back to tensors
        output_images: np.ndarray = np.stack(processed_images)
        output_masks: np.ndarray = np.stack(final_masks)

        # Ensure masks have correct 2D shape (B, H, W) - no channel dimension
        if len(output_masks.shape) == 4:
            output_masks = output_masks[:, :, :, 0]

        # Move back to original device
        return (torch.from_numpy(output_images).to(device), torch.from_numpy(output_masks).to(device))


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

    def detect_white(self, image: torch.Tensor, threshold: float = 0.95) -> tuple[torch.Tensor]:
        # Keep track of device
        device: torch.device = image.device

        # Convert tensor to numpy
        img_np: np.ndarray = image.cpu().numpy()

        # Use shared white detection logic
        output_masks: np.ndarray = detect_white_areas(img_np, threshold)

        # Ensure 2D mask format (B, H, W) - no channel dimension
        if len(output_masks.shape) == 4:
            output_masks = output_masks[:, :, :, 0]

        return (torch.from_numpy(output_masks).to(device),)
