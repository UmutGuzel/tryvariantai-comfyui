from __future__ import annotations

import torch
from typing import Any

from .gpu_ops import gpu_erode, gpu_gaussian_blur


class MaskCleanupNode:
    """
    A ComfyUI node for cleaning up mask edges using morphological operations.
    Practical approach that removes edge artifacts and creates smooth anti-aliased boundaries.
    """

    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("output", "clean_mask")
    FUNCTION: str = "cleanup_mask"
    CATEGORY: str = "TryVariant.ai/postprocessing"
    DESCRIPTION: str = "Cleans up mask edges using morphological erosion and blur. Removes edge artifacts and creates smooth anti-aliased boundaries."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "background_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                }),
                "color_tolerance": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "erode_pixels": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "display": "number"
                }),
                "blur_radius": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "display": "number"
                }),
                "output_mode": (["rgba", "mask_only"], {
                    "default": "rgba"
                }),
            }
        }

    @torch.inference_mode()
    def cleanup_mask(self, image: torch.Tensor, background_color: str, color_tolerance: float, erode_pixels: int, blur_radius: int, output_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        device: torch.device = image.device

        # Parse background color to tensor on device
        bg_color: torch.Tensor = self._parse_color(background_color, device, image.dtype)

        # Step 1: Binary mask from background detection (vectorized across batch)
        rgb: torch.Tensor = image[..., :3]
        color_diff: torch.Tensor = torch.abs(rgb - bg_color.view(1, 1, 1, 3))
        max_diff: torch.Tensor = color_diff.amax(dim=-1)  # (B, H, W)
        binary_mask: torch.Tensor = (max_diff > color_tolerance).float()

        # Step 2: Erode mask
        eroded: torch.Tensor
        if erode_pixels > 0:
            eroded = gpu_erode(binary_mask, erode_pixels, erode_pixels, "ellipse", iterations=1)
        else:
            eroded = binary_mask

        # Step 3: Smooth alpha with Gaussian blur
        smooth: torch.Tensor
        if blur_radius > 0:
            kernel_size: int = blur_radius * 2 + 1
            smooth = gpu_gaussian_blur(eroded, kernel_size)
        else:
            smooth = eroded

        # Step 4: Create output
        output: torch.Tensor
        if output_mode == "rgba":
            alpha_3d: torch.Tensor = smooth.unsqueeze(-1)  # (B, H, W, 1)
            bg_3d: torch.Tensor = bg_color.view(1, 1, 1, 3)
            composited: torch.Tensor = rgb * alpha_3d + bg_3d * (1.0 - alpha_3d)
            output = torch.zeros(*image.shape[:-1], 4, device=device, dtype=image.dtype)
            output[..., :3] = composited
            output[..., 3] = smooth
        else:  # mask_only
            smooth_3d: torch.Tensor = smooth.unsqueeze(-1).expand(-1, -1, -1, 3)
            if image.shape[-1] == 4:
                ones: torch.Tensor = torch.ones(*smooth.shape, 1, device=device, dtype=image.dtype)
                output = torch.cat([smooth_3d, ones], dim=-1)
            else:
                output = smooth_3d

        return (output, smooth)

    def _parse_color(self, color_str: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Parse hex color string to RGB tensor in range [0, 1]."""
        try:
            color_str = color_str.strip()
            if color_str.startswith('#'):
                color_str = color_str[1:]

            if len(color_str) == 6:
                r: float = int(color_str[0:2], 16) / 255.0
                g: float = int(color_str[2:4], 16) / 255.0
                b: float = int(color_str[4:6], 16) / 255.0
                return torch.tensor([r, g, b], device=device, dtype=dtype)
            else:
                return torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        except (ValueError, IndexError):
            return torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
