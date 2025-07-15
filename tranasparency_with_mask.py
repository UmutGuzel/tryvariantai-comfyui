import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F

class MaskToTransparentNode:
    """
    A ComfyUI node that applies transparency to an image based on a mask.
    Properly handles mask normalization and provides threshold control.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "opacity_mode": (["mask_as_opacity", "threshold_cutout"],),
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
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba_image",)
    FUNCTION = "apply_transparency"
    CATEGORY = "TryVariant.ai/postprocessing"
    DESCRIPTION = "Applies transparency to an image using a mask. Mask black = transparent, white = opaque (or inverted)."

    def apply_transparency(self, image, mask, mask_threshold=0.5, opacity_mode="mask_as_opacity", feather=0, invert_mask=False):
        mask_h = mask.shape[1]
        mask_w = mask.shape[2]
        print(mask.shape)
        print(image.shape)
        # Check if image needs resizing to match mask
        if image.shape[1] != mask_h or image.shape[2] != mask_w:
            # Resize image to match mask dimensions
            # F.interpolate expects (B, C, H, W) format
            
            image_resized = image.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            image_resized = F.interpolate(image_resized, size=(mask_h, mask_w), mode='bilinear', align_corners=False)
            image = image_resized.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        
        # Convert tensors to numpy arrays
        img_np = image.cpu().numpy()
        mask_np = mask.cpu().numpy()
        


        # Ensure correct dimensions
        if len(img_np.shape) == 3:
            img_np = np.expand_dims(img_np, 0)
        batch_size = img_np.shape[0]
        
        # Handle mask dimensions
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(np.expand_dims(mask_np, 0), -1)
        elif len(mask_np.shape) == 3:
            if mask_np.shape[0] != batch_size:
                mask_np = np.expand_dims(mask_np, 0)
            else:
                mask_np = np.expand_dims(mask_np, -1)
        
        # Process each image in batch
        processed_images = []
        
        for i in range(batch_size):
            # Get current image and mask
            img = img_np[i]
            msk = mask_np[i] if i < mask_np.shape[0] else mask_np[0]
            
            # Ensure mask is 2D and normalized
            if len(msk.shape) == 3:
                msk = msk[:, :, 0]
            
            # Normalize mask to 0-1 range if needed
            msk_min = msk.min()
            msk_max = msk.max()
            if msk_max > msk_min:
                msk = (msk - msk_min) / (msk_max - msk_min)
            else:
                # If mask is uniform, treat it as all white or all black
                msk = np.ones_like(msk) if msk_max > 0.5 else np.zeros_like(msk)
            
            # Invert mask if requested
            if invert_mask:
                msk = 1.0 - msk
            
            # Apply opacity mode
            if opacity_mode == "threshold_cutout":
                # Binary mode: above threshold = opaque, below = transparent
                alpha_mask = (msk >= mask_threshold).astype(np.float32)
            else:
                # Direct mode: use mask values as opacity
                alpha_mask = msk
            
            # Apply feathering if requested
            if feather > 0:
                alpha_mask = gaussian_filter(alpha_mask, sigma=feather)
                alpha_mask = np.clip(alpha_mask, 0, 1)
            
            # Create RGBA image
            height, width = img.shape[:2]
            
            # Check if image already has alpha channel
            if img.shape[2] == 4:
                rgba = img.copy()
                # Multiply existing alpha with new alpha
                rgba[:, :, 3] = rgba[:, :, 3] * alpha_mask
            else:
                # Create new RGBA image
                rgba = np.zeros((height, width, 4), dtype=np.float32)
                rgba[:, :, :3] = img[:, :, :3]
                rgba[:, :, 3] = alpha_mask
            
            processed_images.append(rgba)
        
        # Convert back to tensor
        output_images = np.stack(processed_images)
        
        return (torch.from_numpy(output_images),)


class DebugMaskNode:
    """
    A helper node to visualize what the mask looks like.
    Useful for debugging mask issues.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "normalize": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask_preview",)
    FUNCTION = "visualize_mask"
    CATEGORY = "TryVariant.ai/postprocessing"
    DESCRIPTION = "Converts a mask to a visible grayscale image for debugging."

    def visualize_mask(self, mask, normalize=True):
        # Convert tensor to numpy
        mask_np = mask.cpu().numpy()
        
        # Handle dimensions
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(mask_np, 0)
        if len(mask_np.shape) == 3:
            batch_size = mask_np.shape[0]
        else:
            batch_size = mask_np.shape[0]
            mask_np = mask_np[:, :, :, 0] if mask_np.shape[3] == 1 else mask_np[:, :, :, 0:1]
        
        images = []
        
        for i in range(batch_size):
            msk = mask_np[i] if len(mask_np.shape) == 3 else mask_np[i, :, :, 0]
            
            # Normalize if requested
            if normalize:
                msk_min = msk.min()
                msk_max = msk.max()
                if msk_max > msk_min:
                    msk = (msk - msk_min) / (msk_max - msk_min)
            
            # Create RGB image from mask
            rgb = np.stack([msk, msk, msk], axis=2)
            images.append(rgb)
        
        output = np.stack(images)
        return (torch.from_numpy(output),)
