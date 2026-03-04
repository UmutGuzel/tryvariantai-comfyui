from __future__ import annotations

import torch
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

    @torch.inference_mode()
    def image_to_mask(self, image: torch.Tensor, channel: str, invert: bool = False) -> tuple[torch.Tensor]:
        if image.dim() == 3:
            image = image.unsqueeze(0)

        num_channels: int = image.shape[-1]

        mask: torch.Tensor
        if channel == "red":
            mask = image[..., 0]
        elif channel == "green":
            mask = image[..., 1]
        elif channel == "blue":
            mask = image[..., 2]
        elif channel == "alpha":
            if num_channels >= 4:
                mask = image[..., 3]
            else:
                mask = torch.ones(image.shape[0], image.shape[1], image.shape[2], device=image.device, dtype=image.dtype)
        elif channel == "average":
            mask = image[..., :3].mean(dim=-1)
        else:  # luminance
            mask = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

        if invert:
            mask = 1.0 - mask

        return (mask,)


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

    @torch.inference_mode()
    def mask_to_image(self, mask: torch.Tensor, invert: bool = False) -> tuple[torch.Tensor]:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if invert:
            mask = 1.0 - mask

        # (B, H, W) → (B, H, W, 3)
        image: torch.Tensor = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        return (image,)
