import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F

class WhiteToTransparentNode:
    """
    A ComfyUI node that makes solid white parts of an image transparent using a mask.
    Outputs an RGBA image with transparency where white areas are detected.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "threshold": ("FLOAT", {
                    "default": 0.95, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display": "slider"
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba_image", "transparency_mask")
    FUNCTION = "make_white_transparent"
    CATEGORY = "TryVariant.ai/postprocessing"
    DESCRIPTION = "Makes white parts of an image transparent based on a mask. Outputs RGBA image with alpha channel."

    def make_white_transparent(self, image, mask, threshold=0.95, feather=0, invert_mask=False):
        # Keep tensors on their original device
        device = image.device
        
        # Get mask dimensions (target dimensions)
        mask_h = mask.shape[1]
        mask_w = mask.shape[2]
        print(mask.shape)
        print(image.shape)
        # Check if image needs resizing to match mask
        if image.shape[1] != mask_h or image.shape[2] != mask_w:
            # Resize image to match mask dimensions
            # F.interpolate expects (B, C, H, W) format
            
            image_resized = image.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            image_resized = F.interpolate(image_resized, size=(mask_h, mask_w), mode='nearest', align_corners=False)
            image = image_resized.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        # Ensure correct dimensions
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(-1)
        elif len(mask.shape) == 3 and mask.shape[-1] != 1:
            mask = mask.unsqueeze(-1)
        
        batch_size = image.shape[0]
        
        # Convert to numpy for processing
        img_np = image.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        # Process each image in batch
        processed_images = []
        final_masks = []
        
        for i in range(batch_size):
            # Get current image and mask
            img = img_np[i]
            msk = mask_np[i] if i < mask_np.shape[0] else mask_np[0]
            
            # Ensure mask is 2D
            if len(msk.shape) == 3:
                msk = msk[:, :, 0]
            
            # Invert mask if requested
            if invert_mask:
                msk = 1.0 - msk
            
            # Detect white pixels (all channels above threshold)
            white_mask = np.all(img >= threshold, axis=2).astype(np.float32)
            
            # Combine with input mask - only make white pixels transparent where mask allows
            transparency_mask = white_mask * msk
            
            # Apply feathering if requested for smoother edges
            if feather > 0:
                transparency_mask = gaussian_filter(transparency_mask, sigma=feather)
                transparency_mask = np.clip(transparency_mask, 0, 1)
            
            # Create RGBA image
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float32)
            
            # Copy RGB channels
            rgba[:, :, :3] = img
            
            # Set alpha channel (1 = opaque, 0 = transparent)
            # Where we detected white (transparency_mask = 1), alpha should be 0 (transparent)
            rgba[:, :, 3] = 1.0 - transparency_mask
            
            processed_images.append(rgba)
            final_masks.append(transparency_mask)
        
        # Convert back to tensors
        output_images = np.stack(processed_images)
        output_masks = np.stack(final_masks)
        
        # Ensure masks have correct shape (B, H, W, 1)
        if len(output_masks.shape) == 3:
            output_masks = np.expand_dims(output_masks, -1)
        
        # Move back to original device
        return (torch.from_numpy(output_images).to(device), torch.from_numpy(output_masks).to(device))


class SimpleWhiteDetectorNode:
    """
    A simple node that detects white areas in an image and creates a mask.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("white_areas_mask",)
    FUNCTION = "detect_white"
    CATEGORY = "TryVariant.ai/postprocessing"
    DESCRIPTION = "Detects white areas in an image and creates a mask."

    def detect_white(self, image, threshold=0.95):
        # Keep track of device
        device = image.device
        
        # Convert tensor to numpy
        img_np = image.cpu().numpy()
        
        if len(img_np.shape) == 4:
            batch_size = img_np.shape[0]
        else:
            img_np = np.expand_dims(img_np, 0)
            batch_size = 1
        
        masks = []
        
        for i in range(batch_size):
            img = img_np[i]
            
            # Detect white pixels (all RGB channels above threshold)
            white_mask = np.all(img >= threshold, axis=2).astype(np.float32)
            
            masks.append(white_mask)
        
        # Convert back to tensor
        output_masks = np.stack(masks)
        if len(output_masks.shape) == 3:
            output_masks = np.expand_dims(output_masks, -1)
        
        return (torch.from_numpy(output_masks).to(device),)

