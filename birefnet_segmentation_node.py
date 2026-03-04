from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

# Module-level model cache — loaded once, reused across invocations
_loaded_model: Any | None = None

# Cached normalization tensors (avoids GPU alloc every inference call)
_norm_tensors: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


class BiRefNetSegmentationNode:
    """
    A ComfyUI node for semantic foreground/background segmentation using BiRefNet.
    Correctly detects interior holes (bag handles, cup handles) as background.
    Uses the general model (best quality).
    """

    MODEL_ID: str = "ZhengPeng7/BiRefNet"

    # ImageNet normalization constants
    _MEAN: list[float] = [0.485, 0.456, 0.406]
    _STD: list[float] = [0.229, 0.224, 0.225]

    RETURN_TYPES: tuple[str, ...] = ("MASK", "IMAGE")
    RETURN_NAMES: tuple[str, ...] = ("mask", "rgba_image")
    FUNCTION: str = "segment"
    CATEGORY: str = "TryVariant.ai/segmentation"
    DESCRIPTION: str = (
        "Semantic foreground/background segmentation using BiRefNet. "
        "Detects interior holes (bag handles, cup handles) as background."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 2048,
                        "step": 64,
                        "display": "number",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            },
            "optional": {
                "output_binary": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    @torch.inference_mode()
    def segment(
        self,
        image: torch.Tensor,
        resolution: int,
        threshold: float,
        output_binary: bool = False,
        invert_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        device: torch.device = image.device

        if image.dim() == 3:
            image = image.unsqueeze(0)

        import comfy.model_management as mm

        model: Any = self._load_model()

        device_to_use = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # 1. Move to GPU just for inference
        model.to(device_to_use)
        model_device = device_to_use

        batch_size: int = image.shape[0]
        orig_h: int = image.shape[1]
        orig_w: int = image.shape[2]

        # Preprocess on model device (CUDA) for inference
        # (B, H, W, C) → (B, C, H, W)
        img_rgb: torch.Tensor = (
            image[..., :3].permute(0, 3, 1, 2).to(model_device).float()
        )

        # Resize to model resolution
        img_resized: torch.Tensor = F.interpolate(
            img_rgb, size=(resolution, resolution), mode="bilinear", align_corners=False
        )
        del img_rgb

        # ImageNet normalization (cached tensors)
        if model_device not in _norm_tensors:
            _norm_tensors[model_device] = (
                torch.tensor(self._MEAN, device=model_device, dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
                torch.tensor(self._STD, device=model_device, dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
            )
        mean, std = _norm_tensors[model_device]
        img_normalized: torch.Tensor = ((img_resized - mean) / std).half()
        del img_resized

        # Batch inference
        try:
            preds: torch.Tensor = model(img_normalized)[
                -1
            ].sigmoid()  # (B, 1, res, res)
        finally:
            # 2. Immediately offload model to free VRAM for other nodes
            model.to(offload_device)
            mm.soft_empty_cache()

        del img_normalized

        # Resize predictions back to original resolution
        pred_masks: torch.Tensor = F.interpolate(
            preds.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False
        ).squeeze(
            1
        )  # (B, H, W)

        if invert_mask:
            pred_masks = 1.0 - pred_masks

        if output_binary:
            pred_masks = (pred_masks > threshold).float()

        # Move masks to original device
        pred_masks = pred_masks.to(device)

        # Build RGBA on device
        rgba: torch.Tensor = torch.zeros(
            batch_size, orig_h, orig_w, 4, device=device, dtype=image.dtype
        )
        rgba[..., :3] = image[..., :3]
        rgba[..., 3] = pred_masks

        return (pred_masks, rgba)

    def _load_model(self) -> Any:
        """Load BiRefNet architecture into node utilities, and weights from ComfyUI models."""
        global _loaded_model

        if _loaded_model is not None:
            return _loaded_model

        import glob

        import comfy.utils
        import folder_paths
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoModelForImageSegmentation

        # 1. Path to natively vendored architecture scripts
        lib_path = os.path.join(os.path.dirname(__file__), "utils", "birefnet_lib")

        # 2. Ensure birefnet is registered in ComfyUI paths for weights
        if "birefnet" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(
                "birefnet", os.path.join(folder_paths.models_dir, "birefnet")
            )

        local_weight_dir = None
        safetensors_file = None

        # Check all registered birefnet paths for safetensors
        for path in folder_paths.get_folder_paths("birefnet"):
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

        # If not found, download weights to the first registered path
        if safetensors_file is None:
            target_base = folder_paths.get_folder_paths("birefnet")[0]
            local_weight_dir = target_base
            print(f"[BiRefNet] Downloading weights to {local_weight_dir}...")
            snapshot_download(
                repo_id=self.MODEL_ID,
                local_dir=local_weight_dir,
                local_dir_use_symlinks=False,
                allow_patterns=[
                    "*.safetensors",
                    "*.bin",
                    "*.pth",
                    "*.pt",
                ],
            )
            # Find the downloaded file
            st_files = (
                glob.glob(os.path.join(local_weight_dir, "*.safetensors"))
                + glob.glob(os.path.join(local_weight_dir, "*.pth"))
                + glob.glob(os.path.join(local_weight_dir, "*.pt"))
                + glob.glob(os.path.join(local_weight_dir, "*.bin"))
            )
            if len(st_files) > 0:
                downloaded_file = st_files[0]
                resource_name = self.MODEL_ID.split("/")[-1]
                ext = os.path.splitext(downloaded_file)[1]
                new_file_path = os.path.join(local_weight_dir, f"{resource_name}{ext}")

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
                f"Could not find or download BiRefNet weights in {local_weight_dir}"
            )

        print(f"[BiRefNet] Instantiating architecture from: {lib_path}")
        config = AutoConfig.from_pretrained(lib_path, trust_remote_code=True)
        model = AutoModelForImageSegmentation.from_config(
            config, trust_remote_code=True
        )

        print(f"[BiRefNet] Loading weights from: {safetensors_file}")
        sd = comfy.utils.load_torch_file(safetensors_file)
        model.load_state_dict(sd)

        offload_device = comfy.model_management.unet_offload_device()
        model.to(offload_device).eval().half()
        _loaded_model = model
        print(
            f"[BiRefNet] Model loaded successfully to offload device ({offload_device})"
        )
        return model
