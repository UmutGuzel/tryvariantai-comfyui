import torch
import numpy as np
from PIL import Image, ImageColor
import comfy.model_management

class FillTransparencyNode:
    """
    A ComfyUI node that fills transparent parts of an image with a selected color.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fill_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                    "tooltip": "Color to fill transparent areas (hex, RGB, or color name)"
                }),
                "blend_mode": (["replace", "multiply", "screen", "overlay"], {
                    "default": "replace",
                    "tooltip": "How to blend the fill color"
                }),
            },
            "optional": {
                "alpha_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Alpha threshold below which pixels are considered transparent"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "fill_transparency"
    CATEGORY = "TryVariant.ai/postprocessing"
    DESCRIPTION = "Fills transparent parts of an image with a selected color"

    def fill_transparency(self, image, fill_color, blend_mode="replace", alpha_threshold=0.1):
        """
        Fill transparent parts of the image with the specified color.
        
        Args:
            image: Input image tensor (B, H, W, C)
            fill_color: Color to fill with (hex, RGB tuple, or color name)
            blend_mode: How to blend the fill color
            alpha_threshold: Alpha value below which pixels are considered transparent
        """
        # Convert ComfyUI tensor to PIL Image
        # ComfyUI images are in format (batch, height, width, channels) with values 0-1
        batch_size = image.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            # Get single image from batch
            img_tensor = image[i]
            
            # Convert to PIL Image
            if img_tensor.shape[2] == 3:  # RGB
                # Add alpha channel if not present
                img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_array, mode='RGB')
                pil_img = pil_img.convert('RGBA')
            elif img_tensor.shape[2] == 4:  # RGBA
                img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_array, mode='RGBA')
            else:
                raise ValueError(f"Unsupported number of channels: {img_tensor.shape[2]}")
            
            # Parse fill color
            try:
                if isinstance(fill_color, str):
                    if fill_color.startswith('#'):
                        # Hex color
                        fill_rgb = ImageColor.getrgb(fill_color)
                    else:
                        # Try as color name
                        fill_rgb = ImageColor.getrgb(fill_color)
                else:
                    fill_rgb = tuple(fill_color)
                
                # Ensure RGB tuple has 3 values
                if len(fill_rgb) == 4:
                    fill_rgb = fill_rgb[:3]  # Remove alpha if present
                elif len(fill_rgb) != 3:
                    raise ValueError("Invalid color format")
                    
            except (ValueError, TypeError):
                # Default to white if color parsing fails
                fill_rgb = (255, 255, 255)
            
            # Process the image
            processed_img = self._fill_transparent_areas(
                pil_img, fill_rgb, blend_mode, alpha_threshold
            )
            
            # Convert back to tensor
            if processed_img.mode != 'RGB':
                processed_img = processed_img.convert('RGB')
            
            img_array = np.array(processed_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)
            
            processed_images.append(img_tensor)
        
        # Stack batch back together
        result = torch.stack(processed_images, dim=0)
        
        return (result,)
    
    def _fill_transparent_areas(self, pil_img, fill_color, blend_mode, alpha_threshold):
        """
        Fill transparent areas of a PIL Image with the specified color.
        """
        # Ensure we have an RGBA image
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        
        # Create a background image with the fill color
        background = Image.new('RGB', pil_img.size, fill_color)
        
        # Get alpha channel
        alpha = pil_img.split()[-1]
        alpha_array = np.array(alpha).astype(np.float32) / 255.0
        
        # Create mask for transparent areas
        transparent_mask = alpha_array < alpha_threshold
        
        if blend_mode == "replace":
            # Simple alpha compositing
            result = Image.alpha_composite(
                background.convert('RGBA'), 
                pil_img
            )
        elif blend_mode == "multiply":
            # Multiply blend mode
            fg_array = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
            bg_array = np.array(background).astype(np.float32) / 255.0
            
            # Apply multiply blend where alpha is low
            blended = fg_array * bg_array
            
            # Use alpha to blend between original and multiplied
            alpha_3d = np.stack([alpha_array] * 3, axis=2)
            result_array = alpha_3d * fg_array + (1 - alpha_3d) * blended
            
            result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
            result = Image.fromarray(result_array, mode='RGB')
            
        elif blend_mode == "screen":
            # Screen blend mode
            fg_array = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
            bg_array = np.array(background).astype(np.float32) / 255.0
            
            # Apply screen blend: 1 - (1-fg) * (1-bg)
            blended = 1 - (1 - fg_array) * (1 - bg_array)
            
            alpha_3d = np.stack([alpha_array] * 3, axis=2)
            result_array = alpha_3d * fg_array + (1 - alpha_3d) * blended
            
            result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
            result = Image.fromarray(result_array, mode='RGB')
            
        elif blend_mode == "overlay":
            # Overlay blend mode
            fg_array = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
            bg_array = np.array(background).astype(np.float32) / 255.0
            
            # Overlay formula
            mask = bg_array < 0.5
            blended = np.where(mask, 
                              2 * fg_array * bg_array,
                              1 - 2 * (1 - fg_array) * (1 - bg_array))
            
            alpha_3d = np.stack([alpha_array] * 3, axis=2)
            result_array = alpha_3d * fg_array + (1 - alpha_3d) * blended
            
            result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
            result = Image.fromarray(result_array, mode='RGB')
        else:
            # Default to replace
            result = Image.alpha_composite(
                background.convert('RGBA'), 
                pil_img
            )
        
        return result.convert('RGB')
