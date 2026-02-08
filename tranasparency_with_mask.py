from __future__ import annotations

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from typing import Any


class MaskToTransparentNode:
    """
    A ComfyUI node that applies transparency to an image based on a mask.
    Properly handles mask normalization and provides threshold control.
    """

    RETURN_TYPES: tuple[str, ...] = ("IMAGE",)
    RETURN_NAMES: tuple[str, ...] = ("rgba_image",)
    FUNCTION: str = "apply_transparency"
    CATEGORY: str = "TryVariant.ai/postprocessing"
    DESCRIPTION: str = "Applies transparency to an image using a mask. Mask black = transparent, white = opaque (or inverted)."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "opacity_mode": (["mask_as_opacity", "threshold_cutout"],),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    def apply_transparency(self, image: torch.Tensor, mask: torch.Tensor, mask_threshold: float = 0.5, opacity_mode: str = "mask_as_opacity", feather: int = 0, invert_mask: bool = False) -> tuple[torch.Tensor]:
        mask_h: int = mask.shape[1]
        mask_w: int = mask.shape[2]

        # Check if image needs resizing to match mask
        if image.shape[1] != mask_h or image.shape[2] != mask_w:
            image_resized: torch.Tensor = image.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            image_resized = F.interpolate(image_resized, size=(mask_h, mask_w), mode='bilinear', align_corners=False)
            image = image_resized.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        # Convert tensors to numpy arrays
        img_np: np.ndarray = image.cpu().numpy()
        mask_np: np.ndarray = mask.cpu().numpy()

        # Ensure correct dimensions
        if len(img_np.shape) == 3:
            img_np = np.expand_dims(img_np, 0)
        batch_size: int = img_np.shape[0]

        # Handle mask dimensions
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(np.expand_dims(mask_np, 0), -1)
        elif len(mask_np.shape) == 3:
            if mask_np.shape[0] != batch_size:
                mask_np = np.expand_dims(mask_np, 0)
            else:
                mask_np = np.expand_dims(mask_np, -1)

        # Process each image in batch
        processed_images: list[np.ndarray] = []

        for i in range(batch_size):
            # Get current image and mask
            img: np.ndarray = img_np[i]
            msk: np.ndarray = mask_np[i] if i < mask_np.shape[0] else mask_np[0]

            # Ensure mask is 2D and normalized
            if len(msk.shape) == 3:
                msk = msk[:, :, 0]

            # Normalize mask to 0-1 range if needed
            msk_min: float = float(msk.min())
            msk_max: float = float(msk.max())
            if msk_max > msk_min:
                msk = (msk - msk_min) / (msk_max - msk_min)
            else:
                msk = np.ones_like(msk) if msk_max > 0.5 else np.zeros_like(msk)

            # Invert mask if requested
            if invert_mask:
                msk = 1.0 - msk

            # Apply opacity mode
            alpha_mask: np.ndarray
            if opacity_mode == "threshold_cutout":
                alpha_mask = (msk >= mask_threshold).astype(np.float32)
            else:
                alpha_mask = msk

            # Apply feathering if requested
            if feather > 0:
                alpha_mask = gaussian_filter(alpha_mask, sigma=feather)
                alpha_mask = np.clip(alpha_mask, 0, 1)

            # Create RGBA image
            height: int = img.shape[0]
            width: int = img.shape[1]

            # Check if image already has alpha channel
            rgba: np.ndarray
            if img.shape[2] == 4:
                rgba = img.copy()
                rgba[:, :, 3] = rgba[:, :, 3] * alpha_mask
            else:
                rgba = np.zeros((height, width, 4), dtype=np.float32)
                rgba[:, :, :3] = img[:, :, :3]
                rgba[:, :, 3] = alpha_mask

            processed_images.append(rgba)

        # Convert back to tensor
        output_images: np.ndarray = np.stack(processed_images)

        return (torch.from_numpy(output_images),)


class DebugMaskNode:
    """
    A helper node to visualize what the mask looks like.
    Useful for debugging mask issues.
    """

    RETURN_TYPES: tuple[str, ...] = ("IMAGE",)
    RETURN_NAMES: tuple[str, ...] = ("mask_preview",)
    FUNCTION: str = "visualize_mask"
    CATEGORY: str = "TryVariant.ai/postprocessing"
    DESCRIPTION: str = "Converts a mask to a visible grayscale image for debugging."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "mask": ("MASK",),
                "normalize": ("BOOLEAN", {"default": True}),
            }
        }

    def visualize_mask(self, mask: torch.Tensor, normalize: bool = True) -> tuple[torch.Tensor]:
        # Convert tensor to numpy
        mask_np: np.ndarray = mask.cpu().numpy()

        # Handle dimensions
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(mask_np, 0)
        if len(mask_np.shape) == 3:
            batch_size: int = mask_np.shape[0]
        else:
            batch_size = mask_np.shape[0]
            mask_np = mask_np[:, :, :, 0] if mask_np.shape[3] == 1 else mask_np[:, :, :, 0:1]

        images: list[np.ndarray] = []

        for i in range(batch_size):
            msk: np.ndarray = mask_np[i] if len(mask_np.shape) == 3 else mask_np[i, :, :, 0]

            # Normalize if requested
            if normalize:
                msk_min: float = float(msk.min())
                msk_max: float = float(msk.max())
                if msk_max > msk_min:
                    msk = (msk - msk_min) / (msk_max - msk_min)

            # Create RGB image from mask
            rgb: np.ndarray = np.stack([msk, msk, msk], axis=2)
            images.append(rgb)

        output: np.ndarray = np.stack(images)
        return (torch.from_numpy(output),)
