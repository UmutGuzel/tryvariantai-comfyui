import torch
from typing import Any

class CropWithMaskNode:
    """
    Takes an IMAGE and a MASK, calculates the tightest bounding box around the white areas 
    of the mask, and crops both the IMAGE and MASK to that exact footprint.
    Maintains batch dimensions safely.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {"default": 16, "min": 0, "max": 1024, "step": 1, "tooltip": "Extra pixel padding around the tight crop"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("cropped_image", "cropped_mask")
    FUNCTION = "crop_to_mask"
    CATEGORY = "TryVariant.ai/utils"

    def crop_to_mask(self, image: torch.Tensor, mask: torch.Tensor, padding: int) -> tuple[torch.Tensor, torch.Tensor]:
        # image shape: [B, H, W, C]
        # mask shape: [B, H, W]
        
        # Ensure mask and image heights/widths align just in case
        min_H = min(image.shape[1], mask.shape[1])
        min_W = min(image.shape[2], mask.shape[2])
        
        sliced_image = image[:, :min_H, :min_W, :]
        sliced_mask = mask[:, :min_H, :min_W]
        
        # Collapse batch dimension to find the global bounding footprint across all frames
        mask_coll = sliced_mask.sum(dim=0) # [H, W]
        rows = torch.any(mask_coll > 0.01, dim=1)
        cols = torch.any(mask_coll > 0.01, dim=0)

        indices_rows = torch.where(rows)[0]
        indices_cols = torch.where(cols)[0]

        # If the mask is entirely pitch black, fallback to returning a 1x1 dummy to prevent crashing
        if len(indices_rows) == 0 or len(indices_cols) == 0:
            return (torch.zeros_like(sliced_image[:, :1, :1, :]), torch.zeros_like(sliced_mask[:, :1, :1]))

        # Calculate exact crop anchors
        y_min, y_max = indices_rows[0].item(), indices_rows[-1].item()
        x_min, x_max = indices_cols[0].item(), indices_cols[-1].item()

        # Apply padding while ensuring we never step outside the image boundaries
        y_min = max(0, y_min - padding)
        y_max = min(min_H - 1, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(min_W - 1, x_max + padding)

        # Execute final crop array slicing
        cropped_image = sliced_image[:, y_min:y_max+1, x_min:x_max+1, :]
        cropped_mask = sliced_mask[:, y_min:y_max+1, x_min:x_max+1]

        return (cropped_image, cropped_mask)
