from __future__ import annotations

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Any

# Module-level model cache — loaded once, reused across invocations
_loaded_models: dict[str, Any] = {}


class BiRefNetSegmentationNode:
    """
    A ComfyUI node for semantic foreground/background segmentation using BiRefNet.
    Correctly detects interior holes (bag handles, cup handles) as background.
    Supports general (best quality) and lite (faster) model variants.
    """

    MODEL_IDS: dict[str, str] = {
        "general": "ZhengPeng7/BiRefNet",
        "lite": "ZhengPeng7/BiRefNet_lite",
    }

    RETURN_TYPES: tuple[str, ...] = ("MASK", "IMAGE")
    RETURN_NAMES: tuple[str, ...] = ("mask", "rgba_image")
    FUNCTION: str = "segment"
    CATEGORY: str = "TryVariant.ai/segmentation"
    DESCRIPTION: str = (
        "Semantic foreground/background segmentation using BiRefNet. "
        "Detects interior holes (bag handles, cup handles) as background. "
        "Models: general (best quality, 1024px) or lite (faster)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["general", "lite"], {
                    "default": "general"
                }),
                "resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "display": "number"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
            "optional": {
                "output_binary": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    def segment(self, image: torch.Tensor, model_variant: str, resolution: int, threshold: float,
                output_binary: bool = False, invert_mask: bool = False) -> tuple[torch.Tensor, torch.Tensor]:

        device: torch.device = image.device

        if image.dim() == 3:
            image = image.unsqueeze(0)

        model: Any = self._load_model(model_variant)

        transform_pipeline: transforms.Compose = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        batch_size: int = image.shape[0]
        mask_results: list[torch.Tensor] = []
        rgba_results: list[torch.Tensor] = []

        for i in range(batch_size):
            img_np: np.ndarray = image[i].cpu().numpy()  # (H, W, C) float [0,1]
            orig_h: int = img_np.shape[0]
            orig_w: int = img_np.shape[1]

            # Convert to PIL for torchvision transforms
            img_uint8: np.ndarray = (img_np[:, :, :3] * 255).astype(np.uint8)
            pil_image: Image.Image = Image.fromarray(img_uint8)

            # Preprocess
            input_tensor: torch.Tensor = transform_pipeline(pil_image).unsqueeze(0).to("cuda").half()

            # Inference
            with torch.no_grad():
                preds: torch.Tensor = model(input_tensor)[-1].sigmoid().cpu()

            # Extract mask and resize to original dimensions
            pred_mask: torch.Tensor = preds[0].squeeze()
            mask_pil: Image.Image = transforms.ToPILImage()(pred_mask)
            mask_resized: Image.Image = mask_pil.resize((orig_w, orig_h), Image.BILINEAR)
            mask_np: np.ndarray = np.array(mask_resized).astype(np.float32) / 255.0

            # Apply inversion
            if invert_mask:
                mask_np = 1.0 - mask_np

            # Apply binarization
            if output_binary:
                mask_np = (mask_np > threshold).astype(np.float32)

            # Build RGBA image
            alpha_channel: np.ndarray = mask_np[:, :, np.newaxis]
            rgba: np.ndarray = np.concatenate([img_np[:, :, :3], alpha_channel], axis=-1).astype(np.float32)

            mask_results.append(torch.from_numpy(mask_np))
            rgba_results.append(torch.from_numpy(rgba))

        mask_out: torch.Tensor = torch.stack(mask_results, dim=0).to(device)
        rgba_out: torch.Tensor = torch.stack(rgba_results, dim=0).to(device)

        return (mask_out, rgba_out)

    def _load_model(self, variant: str) -> Any:
        """Load BiRefNet model with local-first, HuggingFace fallback strategy."""
        global _loaded_models

        if variant in _loaded_models:
            return _loaded_models[variant]

        from transformers import AutoModelForImageSegmentation

        # Try local path first (ComfyUI models directory)
        local_path: str | None = self._get_local_model_path(variant)
        model: Any
        if local_path and os.path.exists(local_path):
            print(f"[BiRefNet] Loading {variant} model from local path: {local_path}")
            model = AutoModelForImageSegmentation.from_pretrained(
                local_path, trust_remote_code=True
            )
        else:
            model_id: str = self.MODEL_IDS[variant]
            print(f"[BiRefNet] Downloading {variant} model from HuggingFace: {model_id}")
            model = AutoModelForImageSegmentation.from_pretrained(
                model_id, trust_remote_code=True
            )

        model.to("cuda").eval().half()
        _loaded_models[variant] = model
        print(f"[BiRefNet] {variant} model loaded successfully on CUDA (fp16)")
        return model

    def _get_local_model_path(self, variant: str) -> str | None:
        """Check for pre-installed model in ComfyUI models directory."""
        try:
            import folder_paths
            return os.path.join(folder_paths.models_dir, "birefnet", variant)
        except ImportError:
            return None
