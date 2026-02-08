from __future__ import annotations

import torch
import numpy as np
from PIL import ImageColor
from typing import Any


class FillTransparencyNode:
    """
    A ComfyUI node that fills image areas based on a mask or alpha channel.
    - With valid mask: fills masked areas (mask=1) with specified color
    - Without mask (or invalid mask): fills transparent areas (alpha=0) with specified color
    - Invalid masks: 0x0 or 64x64 dimensions
    """

    RETURN_TYPES: tuple[str, ...] = ("IMAGE",)
    RETURN_NAMES: tuple[str, ...] = ("image",)
    FUNCTION: str = "fill_transparency"
    CATEGORY: str = "TryVariant.ai/postprocessing"
    DESCRIPTION: str = "Fills masked areas of an image with a selected color based on provided mask"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Mask defining areas to fill with white (0=keep original, 1=fill white)"
                }),
                "fill_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                    "tooltip": "Color to fill masked areas (hex, RGB, or color name)"
                }),
            }
        }

    def fill_transparency(self, image: torch.Tensor, mask: torch.Tensor | None = None, fill_color: str = "#FFFFFF") -> tuple[torch.Tensor]:
        batch_size: int = image.shape[0]
        processed_images: list[torch.Tensor] = []

        # Check if mask is valid (not 0x0 or 64x64)
        use_mask: bool = mask is not None and not self._is_invalid_mask(mask)

        # Parse fill color
        fill_rgb: tuple[int, ...]
        try:
            if isinstance(fill_color, str):
                if fill_color.startswith('#'):
                    fill_rgb = ImageColor.getrgb(fill_color)
                else:
                    fill_rgb = ImageColor.getrgb(fill_color)
            else:
                fill_rgb = (255, 255, 255)
        except (ValueError, TypeError):
            fill_rgb = (255, 255, 255)

        for i in range(batch_size):
            img_tensor: torch.Tensor = image[i]  # (H, W, C)

            if use_mask:
                # Use provided mask
                mask_tensor: torch.Tensor
                if mask.shape[0] > 1:
                    mask_tensor = mask[i]  # (H, W)
                else:
                    mask_tensor = mask[0]  # Single mask for all images

                processed_img: torch.Tensor = self._fill_with_mask(img_tensor, mask_tensor, fill_rgb)
            else:
                # No valid mask - use alpha channel if available
                if img_tensor.shape[2] == 4:
                    # Extract alpha as mask
                    alpha_mask: torch.Tensor = img_tensor[:, :, 3]
                    processed_img = self._fill_with_alpha(img_tensor, alpha_mask, fill_rgb)
                else:
                    # No mask and no alpha - return as-is
                    processed_img = img_tensor

            processed_images.append(processed_img)

        result: torch.Tensor = torch.stack(processed_images, dim=0)
        return (result,)

    def _is_invalid_mask(self, mask: torch.Tensor | None) -> bool:
        """Check if mask is invalid (0x0 or 64x64)"""
        if mask is None:
            return True

        h: int = mask.shape[-2]
        w: int = mask.shape[-1]

        # Check for 0x0
        if h == 0 or w == 0:
            return True

        # Check for 64x64
        if h == 64 and w == 64:
            return True

        return False

    def _fill_with_mask(self, img_tensor: torch.Tensor, mask_tensor: torch.Tensor, fill_color: tuple[int, ...]) -> torch.Tensor:
        # Ensure RGB output
        img_rgb: torch.Tensor
        if img_tensor.shape[2] == 4:
            img_rgb = img_tensor[:, :, :3]
        else:
            img_rgb = img_tensor

        # Convert to numpy for processing
        img_array: np.ndarray = img_rgb.cpu().numpy()  # (H, W, 3), values 0-1
        mask_array: np.ndarray = mask_tensor.cpu().numpy()  # (H, W), values 0-1

        # Normalize fill color to 0-1
        fill_normalized: np.ndarray = np.array(fill_color[:3], dtype=np.float32) / 255.0

        # Expand mask to 3 channels
        mask_3d: np.ndarray = np.stack([mask_array] * 3, axis=2)

        # Blend: where mask=1 use fill_color, where mask=0 use original
        result: np.ndarray = img_array * (1 - mask_3d) + fill_normalized * mask_3d

        # Convert back to tensor
        result_tensor: torch.Tensor = torch.from_numpy(result.astype(np.float32))

        return result_tensor

    def _fill_with_alpha(self, img_tensor: torch.Tensor, alpha_mask: torch.Tensor, fill_color: tuple[int, ...]) -> torch.Tensor:
        # Get RGB channels
        img_rgb: np.ndarray = img_tensor[:, :, :3].cpu().numpy()
        alpha_array: np.ndarray = alpha_mask.cpu().numpy()

        # Normalize fill color
        fill_normalized: np.ndarray = np.array(fill_color[:3], dtype=np.float32) / 255.0

        # Where alpha is low (transparent), fill with color
        fill_mask: np.ndarray = 1.0 - alpha_array
        mask_3d: np.ndarray = np.stack([fill_mask] * 3, axis=2)

        # Blend
        result: np.ndarray = img_rgb * (1 - mask_3d) + fill_normalized * mask_3d

        # Convert back to tensor
        result_tensor: torch.Tensor = torch.from_numpy(result.astype(np.float32))

        return result_tensor
