from __future__ import annotations

import torch
import cv2
import numpy as np
from typing import Any


class MaskFromCounterOpenCV:

    RETURN_TYPES: tuple[str, ...] = ("IMAGE",)
    RETURN_NAMES: tuple[str, ...] = ("image",)
    FUNCTION: str = "make_mask"
    CATEGORY: str = "TryVariant.ai/mask"
    DEPRECATED: bool = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold_value": ("INT", {"default": 220, "min": 0, "max": 255}),
                "min_area": ("INT", {"default": 3000, "min": 0}),
                "finger_gap_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "gap_thickness": ("INT", {"default": 1, "min": 0, "max": 5}),
                "finger_border_cutting": ("BOOLEAN", {"default": True}),
                "border_cut_depth": ("INT", {"default": 3, "min": 1, "max": 10}),
                "finger_separation_kernel": ("INT", {"default": 5, "min": 3, "max": 15}),
                "protect_fingertips": ("BOOLEAN", {"default": True}),
                "enhance_finger_gaps": ("BOOLEAN", {"default": True}),
            }
        }

    def make_mask(
        self,
        image: torch.Tensor,
        threshold_value: int,
        min_area: int,
        finger_gap_strength: float,
        gap_thickness: int,
        finger_border_cutting: bool = True,
        border_cut_depth: int = 3,
        finger_separation_kernel: int = 5,
        protect_fingertips: bool = True,
        enhance_finger_gaps: bool = True,
    ) -> tuple[torch.Tensor]:
        # Convert ComfyUI tensor to numpy array
        img_array: np.ndarray = image[0].cpu().numpy()

        # Convert from float [0,1] to uint8 [0,255]
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = (img_array * 255).astype(np.uint8)

        # Convert to grayscale if needed
        img: np.ndarray
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            img = img_array[:, :, 0]
        elif len(img_array.shape) == 2:
            img = img_array
        else:
            raise ValueError(f"Unsupported image format: {img_array.shape}")

        # Threshold to binary
        _: float
        binary: np.ndarray
        _, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

        # Invert: make shapes white, background black
        inverted: np.ndarray = cv2.bitwise_not(binary)

        # Find contours and create basic filled mask
        contours: list[np.ndarray]
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask: np.ndarray = np.zeros_like(img)

        # Fill contours (create basic white body)
        for cnt in contours:
            area: float = cv2.contourArea(cnt)
            if area > min_area:
                cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # === ENHANCED FINGER BORDER CUTTING ===
        if finger_gap_strength > 0:
            # Get the original finger border lines (black lines in original image)
            finger_borders: np.ndarray = cv2.bitwise_not(binary)

            # Enhanced finger border processing for better finger separation
            if finger_border_cutting and enhance_finger_gaps:
                vertical_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (1, finger_separation_kernel))
                finger_borders = cv2.morphologyEx(finger_borders, cv2.MORPH_CLOSE, vertical_kernel)

                horizontal_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (finger_separation_kernel // 2, 1))
                finger_webs: np.ndarray = cv2.morphologyEx(finger_borders, cv2.MORPH_CLOSE, horizontal_kernel)

                finger_borders = cv2.bitwise_or(finger_borders, finger_webs)

            # Apply border cutting depth
            if finger_border_cutting and border_cut_depth > 0:
                cut_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_cut_depth, border_cut_depth))
                finger_borders = cv2.dilate(finger_borders, cut_kernel, iterations=1)

            # Standard gap thickness processing
            if gap_thickness > 0:
                gap_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_thickness, gap_thickness))
                finger_borders = cv2.dilate(finger_borders, gap_kernel, iterations=1)

            # Determine cutting area based on fingertip protection setting
            kernel_erode: np.ndarray
            interior_mask: np.ndarray
            cutting_area: np.ndarray
            if protect_fingertips:
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
                interior_mask = cv2.erode(mask, kernel_erode, iterations=2)
                cutting_area = interior_mask
            else:
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                interior_mask = cv2.erode(mask, kernel_erode, iterations=1)
                cutting_area = interior_mask

            # Only keep finger border lines that are in the designated cutting area
            interior_lines: np.ndarray = cv2.bitwise_and(finger_borders, cutting_area)

            # Apply finger border cutting with controlled strength
            mask_float: np.ndarray = mask.astype(np.float32)
            gaps_float: np.ndarray = interior_lines.astype(np.float32)

            # Subtract finger borders with strength control
            mask_float = mask_float - (gaps_float * finger_gap_strength)
            mask = np.clip(mask_float, 0, 255).astype(np.uint8)

            # Clean up small artifacts created by finger cutting
            if finger_border_cutting:
                cleanup_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cleanup_kernel)

        # Convert mask back to ComfyUI format
        mask_rgb: np.ndarray = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_float_final: np.ndarray = mask_rgb.astype(np.float32) / 255.0
        mask_batch: np.ndarray = np.expand_dims(mask_float_final, axis=0)

        mask_tensor: torch.Tensor = torch.from_numpy(mask_batch)

        return (mask_tensor,)
