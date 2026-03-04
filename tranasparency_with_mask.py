from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any

from .gpu_ops import gpu_gaussian_blur


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

    @torch.inference_mode()
    def apply_transparency(self, image: torch.Tensor, mask: torch.Tensor, mask_threshold: float = 0.5, opacity_mode: str = "mask_as_opacity", feather: int = 0, invert_mask: bool = False) -> tuple[torch.Tensor]:
        device: torch.device = image.device

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        batch_size: int = image.shape[0]
        mask_h: int = mask.shape[1]
        mask_w: int = mask.shape[2]

        # Resize image to match mask if needed
        if image.shape[1] != mask_h or image.shape[2] != mask_w:
            image_resized: torch.Tensor = image.permute(0, 3, 1, 2)  # (B, C, H, W)
            image_resized = F.interpolate(image_resized, size=(mask_h, mask_w), mode='bilinear', align_corners=False)
            image = image_resized.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Broadcast single mask to batch if needed
        if mask.shape[0] < batch_size:
            mask = mask[0:1].expand(batch_size, -1, -1)

        # Normalize mask to 0-1 range
        msk_min: torch.Tensor = mask.amin(dim=(1, 2), keepdim=True)
        msk_max: torch.Tensor = mask.amax(dim=(1, 2), keepdim=True)
        denom: torch.Tensor = msk_max - msk_min
        # Where range is zero, use a flat value based on the constant
        mask_normalized: torch.Tensor = torch.where(
            denom > 0,
            (mask - msk_min) / denom.clamp(min=1e-7),
            torch.where(msk_max > 0.5, torch.ones_like(mask), torch.zeros_like(mask))
        )

        # Invert if requested
        if invert_mask:
            mask_normalized = 1.0 - mask_normalized

        # Apply opacity mode
        alpha_mask: torch.Tensor
        if opacity_mode == "threshold_cutout":
            alpha_mask = (mask_normalized >= mask_threshold).float()
        else:
            alpha_mask = mask_normalized

        # Apply feathering using GPU gaussian blur
        if feather > 0:
            kernel_size: int = feather * 2 + 1
            alpha_mask = gpu_gaussian_blur(alpha_mask, kernel_size, sigma=float(feather))

        # Create RGBA image (B, H, W, 4)
        num_channels: int = image.shape[-1]
        rgba: torch.Tensor = torch.zeros(batch_size, mask_h, mask_w, 4, device=device, dtype=image.dtype)
        rgba[..., :3] = image[..., :3]

        if num_channels == 4:
            rgba[..., 3] = image[..., 3] * alpha_mask
        else:
            rgba[..., 3] = alpha_mask

        return (rgba,)


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

    @torch.inference_mode()
    def visualize_mask(self, mask: torch.Tensor, normalize: bool = True) -> tuple[torch.Tensor]:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # If 4D, squeeze to 3D
        if mask.dim() == 4:
            mask = mask[:, :, :, 0]

        if normalize:
            msk_min: torch.Tensor = mask.amin(dim=(1, 2), keepdim=True)
            msk_max: torch.Tensor = mask.amax(dim=(1, 2), keepdim=True)
            denom: torch.Tensor = msk_max - msk_min
            mask = torch.where(denom > 0, (mask - msk_min) / denom.clamp(min=1e-7), mask)

        # (B, H, W) → (B, H, W, 3)
        image: torch.Tensor = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        return (image,)
