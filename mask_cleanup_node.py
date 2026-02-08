from __future__ import annotations

import torch
import numpy as np
import cv2
from typing import Any


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

    def cleanup_mask(self, image: torch.Tensor, background_color: str, color_tolerance: float, erode_pixels: int, blur_radius: int, output_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        device: torch.device = image.device
        batch_size: int = image.shape[0]

        # Parse background color
        bg_color: np.ndarray = self._parse_color(background_color)

        processed_images: list[torch.Tensor] = []
        clean_masks: list[torch.Tensor] = []

        for i in range(batch_size):
            img_np: np.ndarray = image[i].cpu().numpy()

            # Step 1: Generate binary mask from background detection
            binary_mask: np.ndarray = self._create_binary_mask(img_np, bg_color, color_tolerance)

            # Step 2: Refine mask with morphological erosion
            eroded_mask: np.ndarray
            if erode_pixels > 0:
                eroded_mask = self._erode_mask(binary_mask, erode_pixels)
            else:
                eroded_mask = binary_mask

            # Step 3: Create smooth alpha channel with Gaussian blur
            smooth_alpha: np.ndarray
            if blur_radius > 0:
                smooth_alpha = self._smooth_alpha(eroded_mask, blur_radius)
            else:
                smooth_alpha = eroded_mask

            # Step 4: Create output based on mode
            output_img: np.ndarray
            if output_mode == "rgba":
                rgba: np.ndarray = self._create_rgba(img_np, smooth_alpha, bg_color)
                output_img = rgba
            else:  # mask_only
                mask_vis: np.ndarray = np.stack([smooth_alpha] * 3, axis=2)
                if img_np.shape[2] == 4:
                    mask_vis = np.concatenate([mask_vis, np.ones_like(smooth_alpha[:, :, np.newaxis])], axis=2)
                output_img = mask_vis

            # Convert to tensors
            output_tensor: torch.Tensor = torch.from_numpy(output_img.astype(np.float32)).to(device)
            mask_tensor: torch.Tensor = torch.from_numpy(smooth_alpha.astype(np.float32)).to(device)

            processed_images.append(output_tensor)
            clean_masks.append(mask_tensor)

        # Stack batches
        output_images: torch.Tensor = torch.stack(processed_images, dim=0)
        output_masks: torch.Tensor = torch.stack(clean_masks, dim=0)

        return (output_images, output_masks)

    def _parse_color(self, color_str: str) -> np.ndarray:
        """Parse hex color string to RGB values in range [0, 1]."""
        try:
            color_str = color_str.strip()
            if color_str.startswith('#'):
                color_str = color_str[1:]

            if len(color_str) == 6:
                r: float = int(color_str[0:2], 16) / 255.0
                g: float = int(color_str[2:4], 16) / 255.0
                b: float = int(color_str[4:6], 16) / 255.0
                return np.array([r, g, b], dtype=np.float32)
            else:
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        except:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _create_binary_mask(self, image: np.ndarray, bg_color: np.ndarray, tolerance: float) -> np.ndarray:
        """
        Create binary mask by detecting background color.

        Returns:
            Binary mask (H, W) with 1=foreground, 0=background
        """
        # Get RGB channels
        rgb: np.ndarray
        if image.shape[2] >= 3:
            rgb = image[:, :, :3]
        else:
            rgb = np.stack([image[:, :, 0]] * 3, axis=2)

        # Calculate color distance from background
        color_diff: np.ndarray = np.abs(rgb - bg_color.reshape(1, 1, 3))
        max_diff: np.ndarray = np.max(color_diff, axis=2)

        # Create binary mask: 1 where different from background, 0 where similar
        binary_mask: np.ndarray = (max_diff > tolerance).astype(np.float32)

        return binary_mask

    def _erode_mask(self, mask: np.ndarray, pixels: int) -> np.ndarray:
        """
        Erode mask to remove thin artifacts and defects.
        """
        # Convert to uint8 for OpenCV
        mask_uint8: np.ndarray = (mask * 255).astype(np.uint8)

        # Create circular kernel for smooth erosion
        kernel_size: int = pixels * 2 + 1
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Erode the mask
        eroded: np.ndarray = cv2.erode(mask_uint8, kernel, iterations=1)

        # Convert back to float
        return eroded.astype(np.float32) / 255.0

    def _smooth_alpha(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """
        Apply Gaussian blur to create smooth anti-aliased edges.
        """
        if radius == 0:
            return mask

        # Gaussian blur needs odd kernel size
        kernel_size: int = radius * 2 + 1

        # Apply Gaussian blur
        blurred: np.ndarray = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

        # Ensure values are in [0, 1] range
        return np.clip(blurred, 0.0, 1.0)

    def _create_rgba(self, image: np.ndarray, alpha: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
        """
        Create RGBA image by compositing with alpha channel.
        """
        # Get RGB channels
        rgb: np.ndarray
        if image.shape[2] >= 3:
            rgb = image[:, :, :3]
        else:
            rgb = np.stack([image[:, :, 0]] * 3, axis=2)

        # Expand alpha to 3D
        alpha_3d: np.ndarray = alpha[:, :, np.newaxis]

        # Composite formula
        bg_array: np.ndarray = bg_color.reshape(1, 1, 3)
        composited_rgb: np.ndarray = rgb * alpha_3d + bg_array * (1 - alpha_3d)

        # Create RGBA
        rgba: np.ndarray = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
        rgba[:, :, :3] = composited_rgb
        rgba[:, :, 3] = alpha

        return rgba
