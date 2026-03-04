from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from .gpu_ops import gpu_gaussian_blur


class MaskFromCounter:
    RETURN_TYPES: tuple[str, ...] = ("IMAGE",)
    RETURN_NAMES: tuple[str, ...] = ("contour_mask",)
    FUNCTION: str = "make_mask"
    CATEGORY: str = "TryVariant.ai/mask"
    SEARCH_ALIASES: list[str] = ["try", "tryvariant", "variant", "tryvariant.ai"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "finger_gap_strength": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
            }
        }

    @torch.inference_mode()
    def make_mask(
        self,
        image: torch.Tensor,
        finger_gap_strength: float = 0.3,
    ) -> tuple[torch.Tensor]:
        out_device: torch.device = image.device

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Move to optimal hardware for GPU-accelerated preprocessing
        import comfy.model_management as mm

        target_device = mm.get_torch_device()
        if out_device != target_device:
            image = image.to(target_device)

        batch_size: int = image.shape[0]
        results: list[torch.Tensor] = []

        for i in range(batch_size):
            gray: torch.Tensor = self._prepare_gray(image[i])
            thresh: torch.Tensor = self._otsu_threshold(gray)

            # Contour finding requires CPU/OpenCV - single transfer
            thresh_u8: np.ndarray = (thresh.cpu().numpy() * 255).astype(np.uint8)
            mask_u8: np.ndarray = self._contour_mask(thresh_u8)

            # Apply finger gap cutting if strength > 0
            if finger_gap_strength > 0:
                mask_u8 = self._apply_finger_gaps(
                    mask_u8, thresh_u8, finger_gap_strength
                )

            mask_t: torch.Tensor = (
                torch.from_numpy(mask_u8).to(device=out_device, dtype=torch.float32)
                / 255.0
            )
            results.append(mask_t.unsqueeze(-1).expand(-1, -1, 3))

        return (torch.stack(results, dim=0),)

    def _prepare_gray(self, img: torch.Tensor) -> torch.Tensor:
        """Convert (H, W, 3) float32 RGB tensor to (H, W) grayscale on same device."""
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    @torch.inference_mode()
    def _otsu_threshold(self, gray: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        """GPU Otsu thresholding: blur + histogram-based optimal threshold."""
        # Blur using gpu_gaussian_blur (expects (B, H, W))
        blurred: torch.Tensor = gpu_gaussian_blur(gray.unsqueeze(0), kernel_size=5)[0]

        # Compute Otsu threshold on GPU via histogram
        hist: torch.Tensor = torch.histc(blurred, bins=256, min=0.0, max=1.0)
        total: float = float(blurred.numel())
        hist_norm: torch.Tensor = hist / total

        # Cumulative sums for between-class variance
        weights: torch.Tensor = torch.cumsum(hist_norm, dim=0)
        bin_centers: torch.Tensor = torch.linspace(0.0, 1.0, 256, device=gray.device)
        means: torch.Tensor = torch.cumsum(hist_norm * bin_centers, dim=0)
        global_mean: float = float(means[-1])

        # Between-class variance: w0 * w1 * (mu0 - mu1)^2
        w0: torch.Tensor = weights
        w1: torch.Tensor = 1.0 - weights
        mu0_num: torch.Tensor = means
        numerator: torch.Tensor = (global_mean * w0 - mu0_num) ** 2
        denominator: torch.Tensor = w0 * w1
        # Avoid division by zero
        valid: torch.Tensor = denominator > 1e-10
        variance: torch.Tensor = torch.where(
            valid, numerator / denominator, torch.zeros_like(denominator)
        )

        best_t: float = float(bin_centers[variance.argmax()])
        return torch.where(blurred > best_t, max_val, 0.0)

    def _contour_mask(self, image: np.ndarray) -> np.ndarray:
        """CPU contour finding (no GPU equivalent for findContours)."""
        contours: list[np.ndarray]
        contours, _ = cv2.findContours(
            cv2.bitwise_not(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask: np.ndarray = np.zeros_like(image)

        for contour in contours:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        return mask

    def _apply_finger_gaps(
        self,
        mask: np.ndarray,
        thresh_u8: np.ndarray,
        finger_gap_strength: float,
    ) -> np.ndarray:
        """Apply finger gap border cutting from thresholded image onto mask."""
        finger_borders: np.ndarray = cv2.bitwise_not(thresh_u8)

        # Enhance finger gaps with morphological closing
        vertical_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        finger_borders = cv2.morphologyEx(
            finger_borders, cv2.MORPH_CLOSE, vertical_kernel
        )

        horizontal_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2, 1)
        )
        finger_webs: np.ndarray = cv2.morphologyEx(
            finger_borders, cv2.MORPH_CLOSE, horizontal_kernel
        )
        finger_borders = cv2.bitwise_or(finger_borders, finger_webs)

        # Apply border cut depth
        cut_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        finger_borders = cv2.dilate(finger_borders, cut_kernel, iterations=1)

        # Gap thickness
        gap_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        finger_borders = cv2.dilate(finger_borders, gap_kernel, iterations=1)

        # Protect fingertips: only cut in interior
        kernel_erode: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        interior_mask: np.ndarray = cv2.erode(mask, kernel_erode, iterations=2)
        interior_lines: np.ndarray = cv2.bitwise_and(finger_borders, interior_mask)

        # Subtract finger borders with strength control
        mask_float: np.ndarray = mask.astype(np.float32)
        gaps_float: np.ndarray = interior_lines.astype(np.float32)
        mask_float = mask_float - (gaps_float * finger_gap_strength)
        result: np.ndarray = np.clip(mask_float, 0, 255).astype(np.uint8)

        # Clean up small artifacts
        cleanup_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2, 2)
        )
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, cleanup_kernel)

        return result


class MaskFromCounterOpenCV:

    RETURN_TYPES: tuple[str, ...] = ("IMAGE",)
    RETURN_NAMES: tuple[str, ...] = ("image",)
    FUNCTION: str = "make_mask"
    CATEGORY: str = "TryVariant.ai/mask"
    # DEPRECATED: bool = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold_value": ("INT", {"default": 220, "min": 0, "max": 255}),
                "min_area": ("INT", {"default": 3000, "min": 0}),
                "finger_gap_strength": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "gap_thickness": ("INT", {"default": 1, "min": 0, "max": 5}),
                "finger_border_cutting": ("BOOLEAN", {"default": True}),
                "border_cut_depth": ("INT", {"default": 3, "min": 1, "max": 10}),
                "finger_separation_kernel": (
                    "INT",
                    {"default": 5, "min": 3, "max": 15},
                ),
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
        contours: Any
        contours, _ = cv2.findContours(
            inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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
                vertical_kernel: np.ndarray = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (1, finger_separation_kernel)
                )
                finger_borders = cv2.morphologyEx(
                    finger_borders, cv2.MORPH_CLOSE, vertical_kernel
                )

                horizontal_kernel: np.ndarray = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (finger_separation_kernel // 2, 1)
                )
                finger_webs: np.ndarray = cv2.morphologyEx(
                    finger_borders, cv2.MORPH_CLOSE, horizontal_kernel
                )

                finger_borders = cv2.bitwise_or(finger_borders, finger_webs)

            # Apply border cutting depth
            if finger_border_cutting and border_cut_depth > 0:
                cut_kernel: np.ndarray = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (border_cut_depth, border_cut_depth)
                )
                finger_borders = cv2.dilate(finger_borders, cut_kernel, iterations=1)

            # Standard gap thickness processing
            if gap_thickness > 0:
                gap_kernel: np.ndarray = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (gap_thickness, gap_thickness)
                )
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
                cleanup_kernel: np.ndarray = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2, 2)
                )
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cleanup_kernel)

        # Convert mask back to ComfyUI format
        mask_rgb: np.ndarray = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_float_final: np.ndarray = mask_rgb.astype(np.float32) / 255.0
        mask_batch: np.ndarray = np.expand_dims(mask_float_final, axis=0)

        mask_tensor: torch.Tensor = torch.from_numpy(mask_batch).to(image.device)

        return (mask_tensor,)
