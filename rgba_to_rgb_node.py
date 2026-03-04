from __future__ import annotations

import torch
from typing import Any


class RGBAtoRGBNode:
    """
    A ComfyUI node that converts RGBA images to RGB by compositing over a background color.
    """

    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("image", "alpha_mask")
    FUNCTION: str = "convert_to_rgb"
    CATEGORY: str = "TryVariant.ai/postprocessing"
    DESCRIPTION: str = "Converts RGBA images to RGB by compositing over a background color. Alpha threshold filters transparent pixels."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "background_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                }),
                "alpha_threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "threshold_mode": (["composite", "replace"], {
                    "default": "composite"
                }),
            }
        }

    @torch.inference_mode()
    def convert_to_rgb(self, image: torch.Tensor, background_color: str = "#FFFFFF", alpha_threshold: float = 0.0, threshold_mode: str = "composite") -> tuple[torch.Tensor, torch.Tensor]:
        device: torch.device = image.device

        # Parse background color to [0, 1] tensor on device
        bg_rgb: list[float] = self._parse_color(background_color)
        bg_tensor: torch.Tensor = torch.tensor(bg_rgb, device=device, dtype=image.dtype).view(1, 1, 1, 3)

        num_channels: int = image.shape[-1]

        result: torch.Tensor
        alpha_mask: torch.Tensor

        if num_channels == 4:
            rgb: torch.Tensor = image[..., :3]
            alpha: torch.Tensor = image[..., 3:4]

            # Apply alpha threshold
            alpha_t: torch.Tensor
            if alpha_threshold > 0.0:
                if threshold_mode == "replace":
                    alpha_t = torch.where(alpha >= alpha_threshold, torch.ones_like(alpha), torch.zeros_like(alpha))
                else:  # composite
                    alpha_t = torch.where(alpha >= alpha_threshold, alpha, torch.zeros_like(alpha))
            else:
                alpha_t = alpha

            # Composite: result = foreground * alpha + background * (1 - alpha)
            result = rgb * alpha_t + bg_tensor * (1.0 - alpha_t)

            # Alpha mask (B, H, W)
            alpha_mask = alpha_t.squeeze(-1)
        elif num_channels == 3:
            result = image
            alpha_mask = torch.ones(image.shape[0], image.shape[1], image.shape[2], device=device, dtype=image.dtype)
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}")

        return (result, alpha_mask)

    def _parse_color(self, color_str: str) -> list[float]:
        """Parse hex color string to RGB values in range [0, 1]."""
        try:
            color_str = color_str.strip()
            if color_str.startswith('#'):
                color_str = color_str[1:]

            if len(color_str) == 6:
                r: float = int(color_str[0:2], 16) / 255.0
                g: float = int(color_str[2:4], 16) / 255.0
                b: float = int(color_str[4:6], 16) / 255.0
                return [r, g, b]
            else:
                return [1.0, 1.0, 1.0]
        except (ValueError, IndexError):
            return [1.0, 1.0, 1.0]
