from __future__ import annotations

import glob
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
    # Pinned commit of the Hugging Face repo. The architecture code (MIT, by the
    # BiRefNet authors) and the weights are fetched from this revision at runtime
    # into the ComfyUI models/birefnet folder, so this package ships none of it.
    MODEL_REVISION: str = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
    ARCH_FILES: tuple[str, ...] = ("birefnet.py", "BiRefNet_config.py", "config.json")
    WEIGHT_GLOBS: tuple[str, ...] = ("*.safetensors", "*.pth", "*.pt")

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

    @classmethod
    def _find_weights(cls, folder: str, include_bin: bool = False) -> str | None:
        """First weight file in `folder` matching WEIGHT_GLOBS (plus *.bin on request)."""
        if not os.path.isdir(folder):
            return None
        patterns = list(cls.WEIGHT_GLOBS) + (["*.bin"] if include_bin else [])
        for pattern in patterns:
            matches = sorted(glob.glob(os.path.join(folder, pattern)))
            if matches:
                return matches[0]
        return None

    def _load_model(self) -> Any:
        """Load BiRefNet from the ComfyUI models/birefnet folder.

        Architecture files (ARCH_FILES) and weights live side by side in that folder.
        Whatever is missing is fetched from MODEL_ID at MODEL_REVISION on first use.
        """
        global _loaded_model

        if _loaded_model is not None:
            return _loaded_model

        import shutil

        import comfy.model_management
        import comfy.utils
        import folder_paths
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoModelForImageSegmentation

        if "birefnet" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(
                "birefnet", os.path.join(folder_paths.models_dir, "birefnet")
            )

        # Prefer a registered folder that already holds weights; otherwise the first one.
        model_dir: str | None = None
        weight_file: str | None = None
        for path in folder_paths.get_folder_paths("birefnet"):
            found = self._find_weights(path)
            if found is not None:
                model_dir, weight_file = path, found
                break
        if model_dir is None:
            model_dir = folder_paths.get_folder_paths("birefnet")[0]

        missing_arch = [
            name for name in self.ARCH_FILES if not os.path.isfile(os.path.join(model_dir, name))
        ]
        patterns: list[str] = list(missing_arch)
        if weight_file is None:
            patterns += list(self.WEIGHT_GLOBS) + ["*.bin"]

        if patterns:
            print(
                f"[BiRefNet] Downloading {patterns} from {self.MODEL_ID}@{self.MODEL_REVISION[:7]} "
                f"to {model_dir}..."
            )
            snapshot_download(
                repo_id=self.MODEL_ID,
                revision=self.MODEL_REVISION,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                allow_patterns=patterns,
            )
            # HuggingFace creates a .cache folder when downloading to local_dir.
            # We delete it so the ComfyUI models folder stays clean.
            shutil.rmtree(os.path.join(model_dir, ".cache"), ignore_errors=True)

        if weight_file is None:
            weight_file = self._find_weights(model_dir, include_bin=True)
            if weight_file is None:
                raise FileNotFoundError(
                    f"Could not find or download BiRefNet weights in {model_dir}"
                )
            # Rename the downloaded file (model.safetensors) to BiRefNet.<ext>
            resource_name = self.MODEL_ID.split("/")[-1]
            ext = os.path.splitext(weight_file)[1]
            new_file_path = os.path.join(model_dir, f"{resource_name}{ext}")
            if weight_file != new_file_path:
                if os.path.exists(new_file_path):
                    os.remove(new_file_path)
                os.rename(weight_file, new_file_path)
                weight_file = new_file_path

        still_missing = [
            name for name in self.ARCH_FILES if not os.path.isfile(os.path.join(model_dir, name))
        ]
        if still_missing:
            raise FileNotFoundError(
                f"BiRefNet architecture files missing in {model_dir}: {still_missing}"
            )

        print(f"[BiRefNet] Instantiating architecture from: {model_dir}")
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForImageSegmentation.from_config(
            config, trust_remote_code=True
        )

        print(f"[BiRefNet] Loading weights from: {weight_file}")
        sd = comfy.utils.load_torch_file(weight_file)
        model.load_state_dict(sd)

        offload_device = comfy.model_management.unet_offload_device()
        model.to(offload_device).eval().half()
        _loaded_model = model
        print(
            f"[BiRefNet] Model loaded successfully to offload device ({offload_device})"
        )
        return model
