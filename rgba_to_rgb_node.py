from __future__ import annotations

import torch
import numpy as np
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

    def convert_to_rgb(self, image: torch.Tensor, background_color: str = "#FFFFFF", alpha_threshold: float = 0.0, threshold_mode: str = "composite") -> tuple[torch.Tensor, torch.Tensor]:
        device: torch.device = image.device
        batch_size: int = image.shape[0]

        # Parse background color
        bg_color: list[float] = self._parse_color(background_color)

        processed_images: list[torch.Tensor] = []
        alpha_masks: list[torch.Tensor] = []

        for i in range(batch_size):
            img_tensor: torch.Tensor = image[i]

            # Convert to numpy
            img_np: np.ndarray = img_tensor.cpu().numpy()

            result: np.ndarray
            alpha_mask: np.ndarray
            if img_np.shape[2] == 4:
                # RGBA image - composite over background
                rgb: np.ndarray = img_np[:, :, :3]
                alpha: np.ndarray = img_np[:, :, 3:4]

                # Apply alpha threshold
                alpha_thresholded: np.ndarray
                if alpha_threshold > 0.0:
                    if threshold_mode == "replace":
                        alpha_thresholded = np.where(alpha >= alpha_threshold, 1.0, 0.0)
                    else:  # composite mode
                        alpha_thresholded = np.where(alpha >= alpha_threshold, alpha, 0.0)
                else:
                    alpha_thresholded = alpha

                # Composite: result = foreground * alpha + background * (1 - alpha)
                bg_array: np.ndarray = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
                result = rgb * alpha_thresholded + bg_array * (1 - alpha_thresholded)

                # Store alpha mask (2D)
                alpha_mask = alpha_thresholded.squeeze(-1)
            elif img_np.shape[2] == 3:
                # Already RGB, just pass through
                result = img_np
                # Create full opacity mask
                alpha_mask = np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.float32)
            else:
                raise ValueError(f"Unsupported number of channels: {img_np.shape[2]}")

            # Convert back to tensor
            result_tensor: torch.Tensor = torch.from_numpy(result.astype(np.float32))
            alpha_mask_tensor: torch.Tensor = torch.from_numpy(alpha_mask.astype(np.float32))

            processed_images.append(result_tensor)
            alpha_masks.append(alpha_mask_tensor)

        # Stack batch
        output: torch.Tensor = torch.stack(processed_images, dim=0).to(device)
        mask_output: torch.Tensor = torch.stack(alpha_masks, dim=0).to(device)

        return (output, mask_output)

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
        except:
            return [1.0, 1.0, 1.0]
