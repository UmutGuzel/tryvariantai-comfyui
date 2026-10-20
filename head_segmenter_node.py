from __future__ import annotations

import os
from typing import Any

import torch

from .utils.selfieseg_lib.model import SelfieSegmenter, load_checkpoint
from .utils.selfieseg_lib.segment import (
    check_header,
    pick_size,
    read_trained_size,
    segment,
)

MODEL_FOLDER: str = "segmentation"

# Module-level cache: absolute checkpoint path -> (model, trained_size). Loaded once.
_loaded_models: dict[str, tuple[SelfieSegmenter, int | None]] = {}


def _register_model_folder() -> None:
    """Register models/segmentation with ComfyUI, the same way the BiRefNet node
    registers models/birefnet."""
    import folder_paths

    if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(
            MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER)
        )


def _list_checkpoints() -> list[str]:
    import folder_paths

    _register_model_folder()
    return [
        name
        for name in folder_paths.get_filename_list(MODEL_FOLDER)
        if name.lower().endswith(".safetensors")
    ]


class HeadSegmenterNode:
    """
    Head mask from a fine-tuned MediaPipe selfie segmenter (selfieseg/1 checkpoint).
    Any input size; the mask comes back at the input's H×W.
    """

    RETURN_TYPES: tuple[str, ...] = ("MASK",)
    RETURN_NAMES: tuple[str, ...] = ("mask",)
    FUNCTION: str = "segment"
    CATEGORY: str = "TryVariant.ai/segmentation"
    DESCRIPTION: str = (
        "Head segmentation with a fine-tuned MediaPipe selfie segmenter loaded from "
        "models/segmentation/*.safetensors (selfieseg/1 format)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (_list_checkpoints(),),
                "size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2048,
                        "step": 32,
                        "display": "number",
                        "tooltip": "Inference resolution (square). 0 = the checkpoint's "
                        "training size, or the graph's native 256 if unknown.",
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
                        "tooltip": "Used only when output_binary is on.",
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
        model_name: str,
        size: int,
        threshold: float,
        output_binary: bool = False,
        invert_mask: bool = False,
    ) -> tuple[torch.Tensor]:
        if image.dim() == 3:
            image = image.unsqueeze(0)

        model, trained_size = self._load_model(model_name)
        size_hw = pick_size(trained_size, model.input_hw, int(size))
        mask = segment(
            model, image, size_hw, float(threshold), bool(output_binary), bool(invert_mask)
        )
        return (mask.to(image.device),)

    @staticmethod
    def _load_model(model_name: str) -> tuple[SelfieSegmenter, int | None]:
        """Resolve `model_name` in the segmentation folder and load it once."""
        import comfy.model_management as mm
        import folder_paths

        _register_model_folder()
        path = folder_paths.get_full_path(MODEL_FOLDER, model_name)
        if path is None:
            raise FileNotFoundError(
                f"{model_name}: not found in any '{MODEL_FOLDER}' model folder "
                f"({folder_paths.get_folder_paths(MODEL_FOLDER)})"
            )
        path = os.path.abspath(path)

        cached = _loaded_models.get(path)
        if cached is not None:
            return cached

        meta = check_header(path)
        trained_size = read_trained_size(meta)
        device = mm.get_torch_device()
        model, _ = load_checkpoint(path, device)
        # 0.46 MB / 110 tensors: kept resident on the compute device on purpose; per-call
        # offloading and soft_empty_cache would cost more than the VRAM they free.
        model.float().eval()

        _loaded_models[path] = (model, trained_size)
        print(
            f"[HeadSegmenter] Loaded {path} on {device} "
            f"(trained size: {trained_size}, native input: {model.input_hw})"
        )
        return model, trained_size
