import torch
import torch.nn.functional as F
import numpy as np
import cv2


class MaskExpandBorder:
    """
    A ComfyUI node that expands the borders of a mask using morphological dilation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand_pixels": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "kernel_shape": (["ellipse", "rectangle", "cross"], {
                    "default": "ellipse"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("expanded_mask",)
    FUNCTION = "expand_mask_border"
    CATEGORY = "TryVariant.ai/mask"
    
    def expand_mask_border(self, mask, expand_pixels, iterations, kernel_shape):
        """
        Expand the borders of a mask using morphological dilation.
        
        Args:
            mask: Input mask tensor (B, H, W) or (H, W)
            expand_pixels: Number of pixels to expand the border
            iterations: Number of dilation iterations
            kernel_shape: Shape of the morphological kernel
        
        Returns:
            Expanded mask tensor
        """
        # Ensure mask is in the right format
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add batch dimension
        
        batch_size, height, width = mask.shape
        expanded_masks = []
        
        # Create morphological kernel
        kernel_size = expand_pixels * 2 + 1
        
        if kernel_shape == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape == "rectangle":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        else:  # cross
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        
        # Process each mask in the batch
        for i in range(batch_size):
            # Convert mask to numpy array (0-255 range)
            mask_np = (mask[i].cpu().numpy() * 255).astype(np.uint8)
            
            # Apply morphological dilation
            expanded_mask_np = cv2.dilate(mask_np, kernel, iterations=iterations)
            
            # Convert back to tensor (0-1 range)
            expanded_mask = torch.from_numpy(expanded_mask_np.astype(np.float32) / 255.0)
            
            # Move to same device as input
            expanded_mask = expanded_mask.to(mask.device)
            expanded_masks.append(expanded_mask)
        
        # Stack all masks back together
        result = torch.stack(expanded_masks, dim=0)
        
        return (result,)


class MaskExpandBorderAdvanced:
    """
    Advanced version with additional options for mask border expansion.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand_pixels": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "method": (["dilation", "gaussian_blur", "distance_transform"], {
                    "default": "dilation"
                }),
                "kernel_shape": (["ellipse", "rectangle", "cross"], {
                    "default": "ellipse"
                }),
                "feather_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("expanded_mask",)
    FUNCTION = "expand_mask_border_advanced"
    CATEGORY = "TryVariant.ai/mask"
    
    def expand_mask_border_advanced(self, mask, expand_pixels, method, kernel_shape, feather_amount):
        """
        Advanced mask border expansion with multiple methods.
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        batch_size, height, width = mask.shape
        expanded_masks = []
        
        for i in range(batch_size):
            mask_np = (mask[i].cpu().numpy() * 255).astype(np.uint8)
            
            if method == "dilation":
                kernel_size = expand_pixels * 2 + 1
                if kernel_shape == "ellipse":
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                elif kernel_shape == "rectangle":
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                else:  # cross
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
                
                expanded_mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                
            elif method == "gaussian_blur":
                # Create a blurred version and threshold it
                blurred = cv2.GaussianBlur(mask_np, (expand_pixels * 2 + 1, expand_pixels * 2 + 1), 0)
                _, expanded_mask_np = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
                
            elif method == "distance_transform":
                # Use distance transform for smoother expansion
                dist_transform = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
                # Create expanded mask based on distance
                expanded_mask_np = ((dist_transform > 0) | (dist_transform <= expand_pixels)).astype(np.uint8) * 255
            
            # Apply feathering if requested
            if feather_amount > 0:
                expanded_mask_np = cv2.GaussianBlur(expanded_mask_np, (int(feather_amount * 2) * 2 + 1, int(feather_amount * 2) * 2 + 1), feather_amount)
            
            # Convert back to tensor
            expanded_mask = torch.from_numpy(expanded_mask_np.astype(np.float32) / 255.0)
            expanded_mask = expanded_mask.to(mask.device)
            expanded_masks.append(expanded_mask)
        
        result = torch.stack(expanded_masks, dim=0)
        return (result,)
