"""The selfie segmenter as a torch model: fetch, convert, checkpoint, infer.

Google publishes two variants of the same architecture (Apache 2.0):
    general    256x256 input   selfie_segmenter.tflite
    landscape  144x256 input   selfie_segmenter_landscape.tflite
Both take an RGB image scaled to [0, 1] (metadata: mean 0 / std 255) and emit a
per-pixel person probability at the input resolution. The graph is fully
convolutional once converted (tflite_graph), so it runs at other sizes too — but
it was trained at its native size, and that is the default everywhere here.

Checkpoint format: one `.safetensors` file — the weights as tensors, the graph spec
and run metadata as JSON strings in the safetensors metadata header (`format`
= `selfieseg/1`). Same container as every other model artifact on this platform,
no pickle, and a fine-tuned model reloads with torch + safetensors alone — no
.tflite, no flatbuffer bindings.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .tflite_graph import TFLiteNet, parse_tflite

CHECKPOINT_FORMAT = "selfieseg/1"
CHECKPOINT_EXT = ".safetensors"
# not from ..config: that pulls pydantic, and this module must import in
# .venv-curation (the editor's head-label proposer loads fine-tuned checkpoints)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_CACHE = PROJECT_ROOT / ".cache" / "models" / "selfie_segmenter"

_BASE = "https://storage.googleapis.com/mediapipe-models/image_segmenter"
VARIANTS = {
    "general": f"{_BASE}/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "landscape": f"{_BASE}/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
}


def fetch_tflite(variant: str, cache_dir: Path = MODELS_CACHE) -> Path:
    """Download the variant's .tflite into the local model cache (idempotent)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (known: {sorted(VARIANTS)})")
    dest = cache_dir / Path(VARIANTS[variant]).name
    if not dest.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(VARIANTS[variant], tmp)
        tmp.replace(dest)
    return dest


class SelfieSegmenter(nn.Module):
    """Wraps the converted graph with the model's input contract.

    forward(x) takes NCHW float in [0, 1] (or uint8, scaled here) and returns the
    person probability map [B, 1, H, W]; forward(x, logits=True) returns the
    pre-sigmoid map for BCE-with-logits training.
    """

    def __init__(self, spec: dict[str, Any]):
        super().__init__()
        self.net = TFLiteNet(spec)

    @property
    def input_hw(self) -> tuple[int, int]:
        return self.net.input_hw

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0
        return self.net(x, logits=logits)

    @torch.no_grad()
    def predict_tensor(self, image: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        """B×3×H×W (uint8 or float [0,1], any resolution, on the model's device) ->
        B×1×H×W probability at the SAME resolution. Both resizes run on the device,
        so on a GPU the whole call is one upload + one download."""
        h, w = size or self.input_hw
        if image.dtype == torch.uint8:
            image = image.to(torch.float32) / 255.0
        x = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        prob = self(x)
        return F.interpolate(prob, size=image.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)

    @torch.no_grad()
    def predict(self, image: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
        """Probability mask (float32, HxW in [0, 1]) at the image's own resolution.

        The image is stretched to `size` (default: the native input size — the same
        non-aspect-preserving resize MediaPipe applies), run once, and the map is
        resized back bilinearly. Resizing happens on the model's device.
        """
        device = next(self.parameters()).device
        rgb = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)[None].to(device)
        return self.predict_tensor(rgb, size)[0, 0].cpu().numpy()


def convert(tflite_path: str | Path) -> SelfieSegmenter:
    """tflite file -> SelfieSegmenter carrying Google's weights."""
    spec, weights = parse_tflite(tflite_path)
    model = SelfieSegmenter(spec)
    model.net.load_state_dict(weights, strict=True)  # every parsed tensor must land somewhere
    model.eval()
    return model


def save_checkpoint(model: SelfieSegmenter, path: str | Path, meta: dict[str, Any] | None = None) -> Path:
    """Write <path>.safetensors (the extension is enforced): tensors = the state_dict
    (parameters and the const buffers), metadata = format / spec / meta as JSON."""
    from safetensors.torch import save_file

    path = Path(path)
    if path.suffix != CHECKPOINT_EXT:
        path = path.with_suffix(CHECKPOINT_EXT)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.net.state_dict().items()}
    save_file(tensors, str(path), metadata={
        "format": CHECKPOINT_FORMAT,
        "spec": json.dumps(model.net.spec, separators=(",", ":")),
        "meta": json.dumps(meta or {}, default=str),
    })
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[SelfieSegmenter, dict[str, Any]]:
    from safetensors import safe_open

    path = Path(path)
    with safe_open(str(path), framework="pt", device=str(device)) as f:
        header = f.metadata() or {}
        if header.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: not a {CHECKPOINT_FORMAT} checkpoint (format={header.get('format')!r})")
        state = {k: f.get_tensor(k) for k in f.keys()}
    model = SelfieSegmenter(json.loads(header["spec"]))
    model.net.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, json.loads(header.get("meta", "{}"))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
