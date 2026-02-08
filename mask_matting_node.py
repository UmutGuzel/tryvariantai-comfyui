from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Any

# Module-level model cache for VitMatte
_vitmatte_models: dict[str, tuple[Any, Any]] = {}


class MaskMattingNode:
    """
    A ComfyUI node for GPU alpha matting / mask refinement.
    Takes a coarse binary mask and original image, produces a refined soft alpha matte
    with natural edges (hair, fur, semi-transparent areas).

    Methods: vitmatte (best quality), guided_filter (fastest) — both run on GPU.
    """

    VITMATTE_IDS: dict[str, str] = {
        "small": "hustvl/vitmatte-small-composition-1k",
        "base": "hustvl/vitmatte-base-composition-1k",
    }

    RETURN_TYPES: tuple[str, ...] = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES: tuple[str, ...] = ("alpha", "foreground", "trimap")
    FUNCTION: str = "alpha_matting"
    CATEGORY: str = "TryVariant.ai/mask"
    DESCRIPTION: str = (
        "GPU alpha matting: refine a coarse mask into a soft alpha matte. "
        "vitmatte = best quality (deep learning), guided_filter = fastest (edge-aware smoothing). "
        "Both run entirely on GPU."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "method": (["vitmatte", "guided_filter"], {
                    "default": "vitmatte"
                }),
                "trimap_erosion": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "trimap_dilation": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "trimap_kernel_shape": (["ellipse", "rectangle", "cross"], {
                    "default": "ellipse"
                }),
            },
            "optional": {
                "vitmatte_model": (["small", "base"], {
                    "default": "small"
                }),
                "guide_radius": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "guide_epsilon": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.0001,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "number"
                }),
                "estimate_foreground": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    def alpha_matting(self, image: torch.Tensor, mask: torch.Tensor, method: str, trimap_erosion: int, trimap_dilation: int,
                      trimap_kernel_shape: str, vitmatte_model: str = "small", guide_radius: int = 8,
                      guide_epsilon: float = 0.02, estimate_foreground: bool = False, invert_mask: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        device: torch.device = image.device

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        batch_size: int = image.shape[0]
        alpha_results: list[torch.Tensor] = []
        fg_results: list[torch.Tensor] = []
        trimap_results: list[torch.Tensor] = []

        for i in range(batch_size):
            img_np: np.ndarray = image[i].cpu().numpy()  # (H, W, C) float [0,1]
            mask_np: np.ndarray = mask[i].cpu().numpy()   # (H, W) float [0,1]

            img_h: int = img_np.shape[0]
            img_w: int = img_np.shape[1]
            mask_h: int = mask_np.shape[0]
            mask_w: int = mask_np.shape[1]
            if mask_h != img_h or mask_w != img_w:
                mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            if invert_mask:
                mask_np = 1.0 - mask_np

            # Binarize and generate trimap
            mask_binary: np.ndarray = (mask_np > 0.5).astype(np.uint8) * 255
            trimap_np: np.ndarray = self._generate_trimap(
                mask_binary, trimap_erosion, trimap_dilation, trimap_kernel_shape
            )

            # Run matting
            alpha_np: np.ndarray
            if method == "vitmatte":
                alpha_np = self._vitmatte_alpha(img_np, trimap_np, vitmatte_model)
            else:
                alpha_np = self._guided_filter_gpu(image[i], mask[i] if not invert_mask else 1.0 - mask[i],
                                                   guide_radius, guide_epsilon, img_h, img_w)

            alpha_np = np.clip(alpha_np, 0.0, 1.0).astype(np.float32)

            # Foreground estimation
            fg_np: np.ndarray
            if estimate_foreground:
                alpha_3ch: np.ndarray = alpha_np[:, :, np.newaxis]
                fg_np = (img_np[:, :, :3] * alpha_3ch).astype(np.float32)
            else:
                fg_np = img_np[:, :, :3].copy().astype(np.float32)

            trimap_vis: np.ndarray = np.stack([trimap_np, trimap_np, trimap_np], axis=-1).astype(np.float32)

            alpha_results.append(torch.from_numpy(alpha_np))
            fg_results.append(torch.from_numpy(fg_np))
            trimap_results.append(torch.from_numpy(trimap_vis))

        alpha_out: torch.Tensor = torch.stack(alpha_results, dim=0).to(device)
        fg_out: torch.Tensor = torch.stack(fg_results, dim=0).to(device)
        trimap_out: torch.Tensor = torch.stack(trimap_results, dim=0).to(device)

        return (alpha_out, fg_out, trimap_out)

    def _generate_trimap(self, mask_uint8: np.ndarray, erosion_px: int, dilation_px: int, kernel_shape: str) -> np.ndarray:
        """Generate trimap: white=foreground, black=background, gray=unknown."""
        shape_map: dict[str, int] = {
            "ellipse": cv2.MORPH_ELLIPSE,
            "rectangle": cv2.MORPH_RECT,
            "cross": cv2.MORPH_CROSS,
        }
        cv_shape: int = shape_map[kernel_shape]

        erode_kernel: np.ndarray = cv2.getStructuringElement(cv_shape, (erosion_px * 2 + 1, erosion_px * 2 + 1))
        dilate_kernel: np.ndarray = cv2.getStructuringElement(cv_shape, (dilation_px * 2 + 1, dilation_px * 2 + 1))

        fg_definite: np.ndarray = cv2.erode(mask_uint8, erode_kernel)
        bg_boundary: np.ndarray = cv2.dilate(mask_uint8, dilate_kernel)

        trimap: np.ndarray = np.full(mask_uint8.shape, 0.5, dtype=np.float64)
        trimap[fg_definite > 127] = 1.0
        trimap[bg_boundary < 127] = 0.0

        return trimap

    def _vitmatte_alpha(self, image_np: np.ndarray, trimap_np: np.ndarray, model_size: str) -> np.ndarray:
        """VitMatte alpha estimation on GPU — best quality."""
        from transformers import VitMatteImageProcessor, VitMatteForImageMatting
        global _vitmatte_models

        model_id: str = self.VITMATTE_IDS[model_size]

        if model_size not in _vitmatte_models:
            print(f"[MaskMatting] Loading VitMatte {model_size} from {model_id}...")
            processor: Any = VitMatteImageProcessor.from_pretrained(model_id)
            model: Any = VitMatteForImageMatting.from_pretrained(model_id)
            model.to("cuda").eval()
            _vitmatte_models[model_size] = (processor, model)
            print(f"[MaskMatting] VitMatte {model_size} loaded on CUDA")

        processor, model = _vitmatte_models[model_size]

        # Prepare PIL inputs
        img_uint8: np.ndarray = (image_np[:, :, :3] * 255).astype(np.uint8)
        pil_image: Image.Image = Image.fromarray(img_uint8, mode="RGB")

        # Trimap: 0=bg, 128=unknown, 255=fg
        trimap_uint8: np.ndarray = np.zeros(trimap_np.shape, dtype=np.uint8)
        trimap_uint8[trimap_np > 0.9] = 255
        trimap_uint8[(trimap_np >= 0.1) & (trimap_np <= 0.9)] = 128
        pil_trimap: Image.Image = Image.fromarray(trimap_uint8, mode="L")

        inputs: dict[str, torch.Tensor] = processor(images=pil_image, trimaps=pil_trimap, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            alpha_tensor: torch.Tensor = model(**inputs).alphas

        # Crop padded output to original size
        orig_h: int = image_np.shape[0]
        orig_w: int = image_np.shape[1]
        alpha: np.ndarray = alpha_tensor[0, 0, :orig_h, :orig_w].cpu().numpy()

        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    def _guided_filter_gpu(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor, radius: int, epsilon: float, target_h: int, target_w: int) -> np.ndarray:
        """GPU guided filter using PyTorch — fastest method."""
        img: torch.Tensor = image_tensor[:, :, :3].to("cuda").float()
        msk: torch.Tensor = mask_tensor.to("cuda").float()

        # Resize mask if needed
        if msk.shape[0] != target_h or msk.shape[1] != target_w:
            msk = F.interpolate(
                msk.unsqueeze(0).unsqueeze(0), size=(target_h, target_w),
                mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)

        # Grayscale guide
        guide: torch.Tensor = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # Reshape to (1, 1, H, W) for avg_pool2d
        guide_4d: torch.Tensor = guide.unsqueeze(0).unsqueeze(0)
        src_4d: torch.Tensor = msk.unsqueeze(0).unsqueeze(0)

        ksize: int = 2 * radius + 1
        pad: int = radius

        mean_I: torch.Tensor = F.avg_pool2d(F.pad(guide_4d, [pad]*4, mode="reflect"), ksize, stride=1)
        mean_p: torch.Tensor = F.avg_pool2d(F.pad(src_4d, [pad]*4, mode="reflect"), ksize, stride=1)
        corr_Ip: torch.Tensor = F.avg_pool2d(F.pad(guide_4d * src_4d, [pad]*4, mode="reflect"), ksize, stride=1)
        corr_II: torch.Tensor = F.avg_pool2d(F.pad(guide_4d * guide_4d, [pad]*4, mode="reflect"), ksize, stride=1)

        var_I: torch.Tensor = corr_II - mean_I * mean_I
        cov_Ip: torch.Tensor = corr_Ip - mean_I * mean_p

        a: torch.Tensor = cov_Ip / (var_I + epsilon)
        b: torch.Tensor = mean_p - a * mean_I

        mean_a: torch.Tensor = F.avg_pool2d(F.pad(a, [pad]*4, mode="reflect"), ksize, stride=1)
        mean_b: torch.Tensor = F.avg_pool2d(F.pad(b, [pad]*4, mode="reflect"), ksize, stride=1)

        q: torch.Tensor = (mean_a * guide_4d + mean_b).squeeze(0).squeeze(0)

        return torch.clamp(q, 0.0, 1.0).cpu().numpy().astype(np.float32)
