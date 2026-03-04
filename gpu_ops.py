from __future__ import annotations

import torch
import torch.nn.functional as F
import cv2
import numpy as np

_CACHE_MAX: int = 32

# Cache for structuring elements (CPU numpy arrays, tiny)
_kernel_cache: dict[tuple[str, int, int], np.ndarray] = {}

# Cache for GPU gaussian kernels
_gaussian_cache: dict[tuple[int, float, torch.device], torch.Tensor] = {}

# Cache for GPU structuring elements (avoids numpy→GPU transfer every call)
_se_gpu_cache: dict[tuple[str, int, int, torch.device], torch.Tensor] = {}


_SHAPE_MAP: dict[str, int] = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rectangle": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}


def _get_structuring_element(shape: str, kh: int, kw: int) -> np.ndarray:
    """Get or create a cached structuring element."""
    key: tuple[str, int, int] = (shape, kh, kw)
    if key not in _kernel_cache:
        if len(_kernel_cache) >= _CACHE_MAX:
            _kernel_cache.pop(next(iter(_kernel_cache)))
        cv_shape: int = _SHAPE_MAP.get(shape, cv2.MORPH_ELLIPSE)
        _kernel_cache[key] = cv2.getStructuringElement(cv_shape, (kw, kh))
    return _kernel_cache[key]


def _get_se_gpu(shape: str, kh: int, kw: int, device: torch.device) -> torch.Tensor:
    """Get or create a cached GPU structuring element tensor."""
    key: tuple[str, int, int, torch.device] = (shape, kh, kw, device)
    if key not in _se_gpu_cache:
        if len(_se_gpu_cache) >= _CACHE_MAX:
            _se_gpu_cache.pop(next(iter(_se_gpu_cache)))
        se_np: np.ndarray = _get_structuring_element(shape, kh, kw)
        _se_gpu_cache[key] = torch.from_numpy(se_np.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    return _se_gpu_cache[key]


def _make_gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Create a 1D gaussian kernel."""
    x: torch.Tensor = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel: torch.Tensor = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


@torch.inference_mode()
def gpu_gaussian_blur(mask: torch.Tensor, kernel_size: int, sigma: float = 0.0) -> torch.Tensor:
    """
    GPU Gaussian blur using separable convolution.

    Args:
        mask: (B, H, W) float tensor
        kernel_size: Must be odd
        sigma: Gaussian sigma. If 0, auto-computed from kernel_size.

    Returns:
        Blurred (B, H, W) float tensor on same device
    """
    if kernel_size <= 1:
        return mask

    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    if sigma <= 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    device: torch.device = mask.device

    # Get or create cached kernel
    cache_key: tuple[int, float, torch.device] = (kernel_size, sigma, device)
    if cache_key not in _gaussian_cache:
        if len(_gaussian_cache) >= _CACHE_MAX:
            _gaussian_cache.pop(next(iter(_gaussian_cache)))
        k1d: torch.Tensor = _make_gaussian_kernel_1d(kernel_size, sigma)
        _gaussian_cache[cache_key] = k1d.to(device)
    k1d = _gaussian_cache[cache_key]

    # Reshape mask to (B, 1, H, W)
    needs_squeeze: bool = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
        needs_squeeze = True

    pad: int = kernel_size // 2

    # Horizontal pass: kernel shape (1, 1, 1, K)
    k_h: torch.Tensor = k1d.view(1, 1, 1, kernel_size)
    blurred: torch.Tensor = F.conv2d(F.pad(mask, [pad, pad, 0, 0], mode="reflect"), k_h)

    # Vertical pass: kernel shape (1, 1, K, 1)
    k_v: torch.Tensor = k1d.view(1, 1, kernel_size, 1)
    blurred = F.conv2d(F.pad(blurred, [0, 0, pad, pad], mode="reflect"), k_v)

    if needs_squeeze:
        blurred = blurred.squeeze(1)

    return torch.clamp(blurred, 0.0, 1.0)


@torch.inference_mode()
def gpu_dilate(mask: torch.Tensor, kernel_h: int, kernel_w: int, shape: str = "ellipse", iterations: int = 1) -> torch.Tensor:
    """
    GPU morphological dilation using F.conv2d.

    Args:
        mask: (B, H, W) float tensor [0, 1]
        kernel_h: Kernel height in pixels (radius, actual size = 2*kernel_h + 1)
        kernel_w: Kernel width in pixels (radius, actual size = 2*kernel_w + 1)
        shape: "ellipse", "rectangle", or "cross"
        iterations: Number of dilation iterations

    Returns:
        Dilated (B, H, W) float tensor on same device
    """
    if kernel_h <= 0 and kernel_w <= 0:
        return mask

    kh: int = max(kernel_h * 2 + 1, 1)
    kw: int = max(kernel_w * 2 + 1, 1)

    # Get cached GPU structuring element
    se: torch.Tensor = _get_se_gpu(shape, kh, kw, mask.device)

    pad_h: int = kh // 2
    pad_w: int = kw // 2

    # Reshape to (B, 1, H, W)
    needs_squeeze: bool = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
        needs_squeeze = True

    result: torch.Tensor = mask
    for _ in range(iterations):
        padded: torch.Tensor = F.pad(result, [pad_w, pad_w, pad_h, pad_h], mode="constant", value=0.0)
        conv: torch.Tensor = F.conv2d(padded, se)
        result = (conv > 0.5).float()

    if needs_squeeze:
        result = result.squeeze(1)

    return result


@torch.inference_mode()
def gpu_erode(mask: torch.Tensor, kernel_h: int, kernel_w: int, shape: str = "ellipse", iterations: int = 1) -> torch.Tensor:
    """
    GPU morphological erosion using F.conv2d.

    Args:
        mask: (B, H, W) float tensor [0, 1]
        kernel_h: Kernel height in pixels (radius, actual size = 2*kernel_h + 1)
        kernel_w: Kernel width in pixels (radius, actual size = 2*kernel_w + 1)
        shape: "ellipse", "rectangle", or "cross"
        iterations: Number of erosion iterations

    Returns:
        Eroded (B, H, W) float tensor on same device
    """
    if kernel_h <= 0 and kernel_w <= 0:
        return mask

    kh: int = max(kernel_h * 2 + 1, 1)
    kw: int = max(kernel_w * 2 + 1, 1)

    # Get cached GPU structuring element
    se: torch.Tensor = _get_se_gpu(shape, kh, kw, mask.device)
    se_sum: float = float(se.sum())

    pad_h: int = kh // 2
    pad_w: int = kw // 2

    # Reshape to (B, 1, H, W)
    needs_squeeze: bool = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
        needs_squeeze = True

    result: torch.Tensor = mask
    for _ in range(iterations):
        padded: torch.Tensor = F.pad(result, [pad_w, pad_w, pad_h, pad_h], mode="constant", value=1.0)
        conv: torch.Tensor = F.conv2d(padded, se)
        result = (conv >= se_sum - 0.5).float()

    if needs_squeeze:
        result = result.squeeze(1)

    return result


def gpu_morph(mask: torch.Tensor, kernel_h: int, kernel_w: int, shape: str = "ellipse", operation: str = "dilate", iterations: int = 1) -> torch.Tensor:
    """
    Unified GPU morphological operation.

    Args:
        mask: (B, H, W) float tensor [0, 1]
        kernel_h: Kernel height radius
        kernel_w: Kernel width radius
        shape: "ellipse", "rectangle", or "cross"
        operation: "dilate" or "erode"
        iterations: Number of iterations

    Returns:
        Morphed (B, H, W) float tensor on same device
    """
    if operation == "dilate":
        return gpu_dilate(mask, kernel_h, kernel_w, shape, iterations)
    else:
        return gpu_erode(mask, kernel_h, kernel_w, shape, iterations)
