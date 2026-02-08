from __future__ import annotations

import torch
import numpy as np
from typing import Any


class ImageToMaskNode:
    """
    Converts an IMAGE tensor to a MASK tensor.
    Supports extracting a specific channel or computing luminance.
    """

    RETURN_TYPES: tuple[str, ...] = ("MASK",)
    RETURN_NAMES: tuple[str, ...] = ("mask",)
    FUNCTION: str = "image_to_mask"
    CATEGORY: str = "TryVariant.ai/conversion"
    DESCRIPTION: str = "Converts an image to a mask by extracting a channel or computing luminance."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (["luminance", "red", "green", "blue", "alpha", "average"], {
                    "default": "luminance"
                }),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    def image_to_mask(self, image: torch.Tensor, channel: str, invert: bool = False) -> tuple[torch.Tensor]:
        device: torch.device = image.device

        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size: int = image.shape[0]
        masks: list[np.ndarray] = []

        for i in range(batch_size):
            img: np.ndarray = image[i].cpu().numpy()  # (H, W, C) float [0,1]
            num_channels: int = img.shape[2]

            mask: np.ndarray
            if channel == "red":
                mask = img[:, :, 0]
            elif channel == "green":
                mask = img[:, :, 1]
            elif channel == "blue":
                mask = img[:, :, 2]
            elif channel == "alpha":
                if num_channels >= 4:
                    mask = img[:, :, 3]
                else:
                    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
            elif channel == "average":
                mask = np.mean(img[:, :, :3], axis=2)
            else:  # luminance
                mask = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

            mask = mask.astype(np.float32)

            if invert:
                mask = 1.0 - mask

            masks.append(mask)

        result: torch.Tensor = torch.from_numpy(np.stack(masks)).to(device)
        return (result,)


class MaskToImageNode:
    """
    Converts a MASK tensor to an IMAGE tensor (grayscale RGB).
    """

    RETURN_TYPES: tuple[str, ...] = ("IMAGE",)
    RETURN_NAMES: tuple[str, ...] = ("image",)
    FUNCTION: str = "mask_to_image"
    CATEGORY: str = "TryVariant.ai/conversion"
    DESCRIPTION: str = "Converts a mask to a grayscale RGB image."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    def mask_to_image(self, mask: torch.Tensor, invert: bool = False) -> tuple[torch.Tensor]:
        device: torch.device = mask.device

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        batch_size: int = mask.shape[0]
        images: list[np.ndarray] = []

        for i in range(batch_size):
            m: np.ndarray = mask[i].cpu().numpy().astype(np.float32)  # (H, W)

            if invert:
                m = 1.0 - m

            rgb: np.ndarray = np.stack([m, m, m], axis=2)  # (H, W, 3)
            images.append(rgb)

        result: torch.Tensor = torch.from_numpy(np.stack(images)).to(device)
        return (result,)
