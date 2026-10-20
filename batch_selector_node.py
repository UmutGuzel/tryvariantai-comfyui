from typing import Any

import torch

class BatchSelectorNode:
    """
    Simply extracts a specific individual Frame (Image or Mask) from a multi-batch Tensors grouping.
    If the indexed frame does not exist, it safely returns the last available frame.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "tooltip": "0-based index to extract. 0 = First image"}),
            },
            "optional": {
                "batch_images": ("IMAGE", {"tooltip": "Connect a multi-frame IMAGE batch here"}),
                "batch_masks": ("MASK", {"tooltip": "Connect a multi-frame MASK batch here"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "select_from_batch"
    CATEGORY = "TryVariant.ai/utils"

    def select_from_batch(self, index: int, batch_images: torch.Tensor = None, batch_masks: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        out_image = None
        out_mask = None
        
        if batch_images is not None:
            # images shape: [B, H, W, C]
            B = batch_images.shape[0]
            safe_idx = min(index, max(0, B - 1))
            # Extract and maintain the batch dimension [1, H, W, C]
            out_image = batch_images[safe_idx].unsqueeze(0)
            
        if batch_masks is not None:
            # masks shape: [B, H, W]
            B = batch_masks.shape[0]
            safe_idx = min(index, max(0, B - 1))
            # Extract and maintain the batch dimension [1, H, W]
            out_mask = batch_masks[safe_idx].unsqueeze(0)
            
        # ComfyUI requires returned tuple elements to exist. Provide a blank dummy tensor if a port was unused.
        if out_image is None:
            out_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
        if out_mask is None:
            out_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
            
        return (out_image, out_mask)
