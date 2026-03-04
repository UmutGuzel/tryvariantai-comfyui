from __future__ import annotations

import torch
import numpy as np
from PIL import Image
import base64
import io
import re
from typing import Any


class Base64DecodeNode:
    """
    Decodes a base64 string to an image tensor and mask.
    Supports both RGB and RGBA images.
    Extracts mask from alpha channel if present.
    """

    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("image", "mask")
    FUNCTION: str = "decode_from_base64"
    CATEGORY: str = "tryvariantai"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "base64_string": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    def decode_from_base64(self, base64_string: str, invert_mask: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        # Remove data URI prefix if present
        if base64_string.startswith('data:'):
            base64_string = re.sub(r'^data:image/[^;]+;base64,', '', base64_string)

        # Remove whitespace and newlines
        base64_string = base64_string.strip().replace('\n', '').replace('\r', '')

        # Decode base64 to bytes
        img_bytes: bytes = base64.b64decode(base64_string)

        # Open image with PIL
        pil_image: Image.Image = Image.open(io.BytesIO(img_bytes))

        # Convert to RGBA or RGB
        has_alpha: bool = False
        if pil_image.mode == 'RGBA':
            img_np: np.ndarray = np.array(pil_image)
            has_alpha = True
        elif pil_image.mode == 'RGB':
            img_np = np.array(pil_image)
        else:
            pil_image = pil_image.convert('RGB')
            img_np = np.array(pil_image)

        pil_image.close()

        # Convert from uint8 [0, 255] to float [0, 1]
        img_np = img_np.astype(np.float32) / 255.0

        # Convert to torch tensor and add batch dimension
        img_tensor: torch.Tensor = torch.from_numpy(img_np).unsqueeze(0)

        # Extract mask from alpha channel
        mask: torch.Tensor
        if has_alpha:
            mask = 1.0 - img_tensor[:, :, :, 3]
        else:
            batch_size: int = img_tensor.shape[0]
            height: int = img_tensor.shape[1]
            width: int = img_tensor.shape[2]
            mask = torch.zeros((batch_size, height, width), dtype=torch.float32)

        # Apply invert_mask option
        if invert_mask:
            mask = 1.0 - mask

        return (img_tensor, mask)
