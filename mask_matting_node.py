from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .gpu_ops import gpu_dilate, gpu_erode


def _get_device() -> torch.device:
    import comfy.model_management as mm

    return mm.get_torch_device()


_vitmatte_model: tuple[Any, Any] | None = None


class MaskMattingNode:
    """Refine a coarse mask into a soft alpha matte using VitMatte."""

    VITMATTE_ID = "hustvl/vitmatte-base-composition-1k"

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("alpha", "foreground", "trimap")
    FUNCTION = "alpha_matting"
    CATEGORY = "TryVariant.ai/mask"
    DESCRIPTION = (
        "GPU alpha matting: refine a coarse mask into a soft alpha matte using VitMatte. "
        "Removes background color bleed from edges automatically."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "trimap_erosion": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "trimap_dilation": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "trimap_kernel_shape": (
                    ["ellipse", "rectangle", "cross"],
                    {"default": "ellipse"},
                ),
                "auto_scale_trimap": ("BOOLEAN", {"default": True}),
                "bg_color": (["white", "black", "none"], {"default": "white"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    @torch.inference_mode()
    def alpha_matting(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        trimap_erosion: int = 10,
        trimap_dilation: int = 10,
        trimap_kernel_shape: str = "ellipse",
        auto_scale_trimap: bool = True,
        bg_color: str = "white",
        invert_mask: bool = False,
    ):

        device = image.device

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        batch_size, img_h, img_w = image.shape[0], image.shape[1], image.shape[2]

        if auto_scale_trimap:
            scale = min(img_h, img_w) / 1024
            trimap_erosion = max(1, round(trimap_erosion * scale))
            trimap_dilation = max(1, round(trimap_dilation * scale))

        alpha_out = torch.zeros(batch_size, img_h, img_w, device=device)
        fg_out = torch.zeros(
            batch_size, img_h, img_w, 3, device=device, dtype=image.dtype
        )
        trimap_out = torch.zeros(batch_size, img_h, img_w, 3, device=device)

        for i in range(batch_size):
            img_i = image[i]
            mask_i = mask[i]

            if mask_i.shape[0] != img_h or mask_i.shape[1] != img_w:
                mask_i = (
                    F.interpolate(
                        mask_i.unsqueeze(0).unsqueeze(0),
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

            if invert_mask:
                mask_i = 1.0 - mask_i

            trimap = self._generate_trimap_gpu(
                mask_i, trimap_erosion, trimap_dilation, trimap_kernel_shape
            )

            alpha_np = self._vitmatte_alpha(img_i.cpu().numpy(), trimap.cpu().numpy())
            alpha_tensor = torch.from_numpy(alpha_np).to(device).clamp(0.0, 1.0)

            alpha_out[i] = alpha_tensor
            fg_out[i] = self._decontaminate(img_i[..., :3], alpha_tensor, bg_color)
            trimap_out[i] = trimap.unsqueeze(-1).expand(-1, -1, 3)

        import comfy.model_management as mm

        mm.soft_empty_cache()
        return (alpha_out, fg_out, trimap_out)

    def _generate_trimap_gpu(
        self,
        mask: torch.Tensor,
        erosion_px: int,
        dilation_px: int,
        kernel_shape: str = "ellipse",
    ) -> torch.Tensor:
        """1.0=foreground, 0.0=background, 0.5=unknown."""
        binary = (mask > 0.5).float().unsqueeze(0)

        fg_definite = gpu_erode(
            binary, erosion_px, erosion_px, kernel_shape, iterations=1
        )
        bg_boundary = gpu_dilate(
            binary, dilation_px, dilation_px, kernel_shape, iterations=1
        )

        trimap = torch.full_like(binary, 0.5)
        trimap[fg_definite > 0.5] = 1.0
        trimap[bg_boundary < 0.5] = 0.0

        return trimap.squeeze(0)

    def _decontaminate(
        self, rgb: torch.Tensor, alpha: torch.Tensor, bg_color: str
    ) -> torch.Tensor:
        """Remove background bleed from semi-transparent edge pixels (0.01 < alpha < 0.99).
        Inverts alpha compositing: fg = (observed - bg * (1 - alpha)) / alpha
        """
        if bg_color == "none":
            return rgb

        alpha_3d = alpha.unsqueeze(-1)
        is_boundary = (alpha_3d > 0.01) & (alpha_3d < 0.99)
        safe_alpha = alpha_3d.clamp(min=0.01)

        if bg_color == "white":
            decontaminated = (rgb - 1.0 + alpha_3d) / safe_alpha
        else:
            decontaminated = rgb / safe_alpha

        return torch.where(is_boundary, decontaminated, rgb).clamp(0.0, 1.0)

    def _vitmatte_alpha(
        self, image_np: np.ndarray, trimap_np: np.ndarray
    ) -> np.ndarray:
        import glob

        import comfy.utils
        import folder_paths
        from huggingface_hub import snapshot_download
        from transformers import (
            VitMatteConfig,
            VitMatteForImageMatting,
            VitMatteImageProcessor,
        )

        global _vitmatte_model

        if _vitmatte_model is None:
            # 1. Vendor config files to utils/vitmatte_lib
            lib_path = os.path.join(os.path.dirname(__file__), "utils", "vitmatte_lib")

            # 2. Setup ComfyUI models paths and search for weights
            if "vitmatte" not in folder_paths.folder_names_and_paths:
                folder_paths.add_model_folder_path(
                    "vitmatte", os.path.join(folder_paths.models_dir, "vitmatte")
                )

            local_weight_dir = None
            safetensors_file = None

            for path in folder_paths.get_folder_paths("vitmatte"):
                if os.path.exists(path):
                    st_files = (
                        glob.glob(os.path.join(path, "*.safetensors"))
                        + glob.glob(os.path.join(path, "*.pth"))
                        + glob.glob(os.path.join(path, "*.pt"))
                    )
                    if len(st_files) > 0:
                        local_weight_dir = path
                        safetensors_file = st_files[0]
                        break

            if safetensors_file is None:
                target_base = folder_paths.get_folder_paths("vitmatte")[0]
                local_weight_dir = target_base
                print(f"[MaskMatting] Downloading weights to {local_weight_dir}...")
                snapshot_download(
                    repo_id=self.VITMATTE_ID,
                    local_dir=local_weight_dir,
                    local_dir_use_symlinks=False,
                    allow_patterns=[
                        "*.safetensors",
                        "*.bin",
                        "*.pth",
                        "*.pt",
                    ],
                )
                st_files = (
                    glob.glob(os.path.join(local_weight_dir, "*.safetensors"))
                    + glob.glob(os.path.join(local_weight_dir, "*.pth"))
                    + glob.glob(os.path.join(local_weight_dir, "*.pt"))
                    + glob.glob(os.path.join(local_weight_dir, "*.bin"))
                )
                if len(st_files) > 0:
                    downloaded_file = st_files[0]
                    resource_name = self.VITMATTE_ID.split("/")[-1]
                    ext = os.path.splitext(downloaded_file)[1]
                    new_file_path = os.path.join(
                        local_weight_dir, f"{resource_name}{ext}"
                    )

                    if downloaded_file != new_file_path:
                        if os.path.exists(new_file_path):
                            os.remove(new_file_path)
                        os.rename(downloaded_file, new_file_path)
                        safetensors_file = new_file_path
                    else:
                        safetensors_file = downloaded_file

                # HuggingFace creates a .cache folder when downloading to local_dir
                # We delete it so the ComfyUI models folder stays clean.
                cache_folder = os.path.join(local_weight_dir, ".cache")
                if os.path.exists(cache_folder):
                    import shutil

                    shutil.rmtree(cache_folder, ignore_errors=True)

            if safetensors_file is None:
                raise FileNotFoundError(
                    f"Could not find or download ViTMatte weights in {local_weight_dir}"
                )

            print(f"[MaskMatting] Instantiating architecture from: {lib_path}")
            processor = VitMatteImageProcessor.from_pretrained(lib_path)
            config = VitMatteConfig.from_pretrained(lib_path)
            model = VitMatteForImageMatting(config)

            print(f"[MaskMatting] Loading weights from: {safetensors_file}")
            sd = comfy.utils.load_torch_file(safetensors_file)
            model.load_state_dict(sd)

            import comfy.model_management as mm

            offload_device = mm.unet_offload_device()
            model.to(offload_device).eval()
            _vitmatte_model = (processor, model)
            print(f"[MaskMatting] VitMatte loaded to offload device ({offload_device})")

        import comfy.model_management as mm

        device_to_use = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        processor, model = _vitmatte_model

        # 1. Move model to GPU for inference
        model.to(device_to_use)

        img_uint8 = (image_np[:, :, :3] * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_uint8, mode="RGB")

        trimap_uint8 = np.zeros(trimap_np.shape, dtype=np.uint8)
        trimap_uint8[trimap_np > 0.9] = 255
        trimap_uint8[(trimap_np >= 0.1) & (trimap_np <= 0.9)] = 128
        pil_trimap = Image.fromarray(trimap_uint8, mode="L")

        inputs = processor(images=pil_image, trimaps=pil_trimap, return_tensors="pt")

        inputs = {k: v.to(device_to_use) for k, v in inputs.items()}

        try:
            with torch.inference_mode():
                alpha_tensor = model(**inputs).alphas
        finally:
            # 2. Immediately offload model to free VRAM for other nodes
            model.to(offload_device)
            mm.soft_empty_cache()

        del inputs

        orig_h, orig_w = image_np.shape[0], image_np.shape[1]
        alpha = alpha_tensor[0, 0, :orig_h, :orig_w].cpu().numpy()
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)
