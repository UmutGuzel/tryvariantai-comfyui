from __future__ import annotations

import torch
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

    @torch.inference_mode()
    def fill_transparency(self, image: torch.Tensor, mask: torch.Tensor | None = None, fill_color: str = "#FFFFFF") -> tuple[torch.Tensor]:
        batch_size: int = image.shape[0]
        device: torch.device = image.device

        # Check if mask is valid (not 0x0 or 64x64)
        use_mask: bool = mask is not None and not self._is_invalid_mask(mask)

        # Parse fill color to [0, 1] tensor on device
        fill_rgb: tuple[int, ...]
        try:
            fill_rgb = ImageColor.getrgb(fill_color) if isinstance(fill_color, str) else (255, 255, 255)
        except (ValueError, TypeError):
            fill_rgb = (255, 255, 255)

        fill_tensor: torch.Tensor = torch.tensor([fill_rgb[0] / 255.0, fill_rgb[1] / 255.0, fill_rgb[2] / 255.0], device=device, dtype=image.dtype)

        rgb: torch.Tensor = image[..., :3]  # (B, H, W, 3)

        if use_mask:
            # Broadcast single mask to batch
            msk: torch.Tensor = mask if mask.shape[0] > 1 else mask[0:1].expand(batch_size, -1, -1)
            msk_3d: torch.Tensor = msk.unsqueeze(-1)  # (B, H, W, 1)
            result: torch.Tensor = rgb * (1.0 - msk_3d) + fill_tensor * msk_3d
        elif image.shape[-1] == 4:
            alpha: torch.Tensor = image[..., 3:4]  # (B, H, W, 1)
            fill_mask: torch.Tensor = 1.0 - alpha
            result = rgb * alpha + fill_tensor * fill_mask
        else:
            result = rgb

        return (result,)

    def _is_invalid_mask(self, mask: torch.Tensor | None) -> bool:
        """Check if mask is invalid (0x0 or 64x64)"""
        if mask is None:
            return True

        h: int = mask.shape[-2]
        w: int = mask.shape[-1]

        if h == 0 or w == 0:
            return True
        if h == 64 and w == 64:
            return True

        return False
