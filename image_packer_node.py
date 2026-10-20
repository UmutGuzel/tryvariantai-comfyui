from __future__ import annotations

from typing import Any
import math

import torch
import torch.nn.functional as F

class DynamicImagePacker:
    """
    Dynamically packs N input images into a tight bounding box around a target (e.g., 1024x1024)
    using mathematical combinatorial calculation to find the optimal centered layout.
    Entire layout calculation and pixel allocation is fully vectorized using PyTorch Tensors.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        opt_inputs = {f"image_{i}": ("IMAGE",) for i in range(1, 9)}
        return {
            "required": {
                "target_width": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 64}),
                "target_height": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 64}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "normalize_sizing": (["true", "false"], {"default": "true"}),
                "fill_color": ("STRING", {"default": "#000000"}),
            },
            "optional": opt_inputs
        }

    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("combined_image", "individual_masks")
    FUNCTION: str = "pack"
    CATEGORY: str = "TryVariant.ai/layout"

    def _extract_valid_images(self, kwargs: dict[str, Any]) -> list[torch.Tensor]:
        images: list[torch.Tensor] = []
        for i in range(1, 9):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                images.append(img)
        return images

    def _fallback_empty_canvas(self, target_width: int, target_height: int, r: int = 0, g: int = 0, b: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        import comfy.model_management as mm
        device = mm.get_torch_device()
        dummy_img = torch.zeros((1, target_height, target_width, 3), dtype=torch.float32, device=device)
        dummy_img[:, :, :, 0] = r / 255.0
        dummy_img[:, :, :, 1] = g / 255.0
        dummy_img[:, :, :, 2] = b / 255.0
        dummy_mask = torch.zeros((1, target_height, target_width), dtype=torch.float32, device=device)
        return dummy_img, dummy_mask

    def _evaluate_partitions(
        self, images: list[torch.Tensor], target_width: int, target_height: int, padding: int, normalize_sizing: str
    ) -> tuple[int, float, torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = len(images)
        num_breaks = N - 1
        combinations = 2 ** num_breaks
        
        indices = torch.arange(combinations)
        breaks = (indices.unsqueeze(1) >> torch.arange(num_breaks)) & 1
        
        row_indices = torch.cat([torch.zeros(combinations, 1, dtype=torch.long), breaks], dim=1).cumsum(dim=1)
        num_rows_per_partition = row_indices[:, -1] + 1
        
        widths = torch.tensor([img.shape[2] for img in images], dtype=torch.float32)
        heights = torch.tensor([img.shape[1] for img in images], dtype=torch.float32)
        
        if normalize_sizing == "true":
            max_dims = torch.maximum(widths, heights)
            norm_factors = 1000.0 / torch.clamp(max_dims, min=1.0)
            widths = widths * norm_factors
            heights = heights * norm_factors
        else:
            norm_factors = torch.ones(N, dtype=torch.float32)

        partition_offsets = torch.arange(combinations).unsqueeze(1) * N
        flat_row_indices = (row_indices + partition_offsets).view(-1)
        flat_widths = widths.unsqueeze(0).expand(combinations, N).reshape(-1)
        flat_heights = heights.unsqueeze(0).expand(combinations, N).reshape(-1)

        max_idx = combinations * N
        row_widths_flat = torch.zeros(max_idx, dtype=torch.float32).scatter_add_(0, flat_row_indices, flat_widths)
        
        row_heights_flat = torch.zeros(max_idx, dtype=torch.float32)
        for p in range(combinations):
            for i in range(N):
                r_idx = row_indices[p, i].item()
                glob_idx = p * N + r_idx
                if flat_heights[p * N + i] > row_heights_flat[glob_idx]: # type: ignore
                    row_heights_flat[glob_idx] = flat_heights[p * N + i] # type: ignore
                    
        items_in_row_flat = torch.zeros(max_idx, dtype=torch.float32).scatter_add_(0, flat_row_indices, torch.ones_like(flat_widths))

        row_widths_2d = row_widths_flat.view(combinations, N)
        row_heights_2d = row_heights_flat.view(combinations, N)
        items_in_row_2d = items_in_row_flat.view(combinations, N)

        pad_costs_2d = torch.clamp(items_in_row_2d - 1, min=0) * padding
        valid_rows = items_in_row_2d > 0

        scale_per_row = (target_width - pad_costs_2d) / torch.clamp(row_widths_2d, min=1e-5)
        scale_per_row[~valid_rows] = float('inf')
        min_scale_w = scale_per_row.min(dim=1).values

        total_heights = row_heights_2d.sum(dim=1)
        pad_cost_h = torch.clamp(num_rows_per_partition - 1, min=0) * padding
        scale_h = (target_height - pad_cost_h) / torch.clamp(total_heights, min=1e-5)

        scales = torch.minimum(min_scale_w, scale_h)

        max_scale = scales.max()
        valid_mask = scales >= max_scale * 0.999
        valid_indices = torch.where(valid_mask)[0]

        variances = []
        for idx in valid_indices:
            num_r = int(num_rows_per_partition[idx].item())
            if num_r <= 1:
                variances.append(0.0)
            else:
                row_w = row_widths_2d[idx, :num_r]
                variances.append(row_w.var().item())
                
        best_idx_local = int(torch.tensor(variances).argmin().item()) # type: ignore
        best_partition_idx = int(valid_indices[best_idx_local].item()) # type: ignore
        best_scale = float(scales[best_partition_idx].item())
        best_row_indices = row_indices[best_partition_idx]
        best_num_rows = int(num_rows_per_partition[best_partition_idx].item())

        return (best_partition_idx, best_scale, best_row_indices, best_num_rows, 
                row_widths_2d, row_heights_2d, items_in_row_2d, total_heights, pad_cost_h, norm_factors)

    def _generate_coordinates(
        self, images: list[torch.Tensor], padding: int, layout_data: tuple
    ) -> tuple[int, int, list[tuple[float, float, float, float] | None]]:
        # Unpack layout_data
        (best_partition_idx, best_scale, best_row_indices, best_num_rows, 
         row_widths_2d, row_heights_2d, items_in_row_2d, total_heights, pad_cost_h,
         norm_factors) = layout_data
         
        widths = [img.shape[2] for img in images]
        heights = [img.shape[1] for img in images]
        N = len(images)

        max_used_width = max(
            row_widths_2d[best_partition_idx, r].item() * best_scale + max(0, items_in_row_2d[best_partition_idx, r].item() - 1) * padding 
            for r in range(best_num_rows)
        )
        max_used_height = total_heights[best_partition_idx].item() * best_scale + pad_cost_h[best_partition_idx].item()

        final_w = max(1, math.ceil(max_used_width))
        final_h = max(1, math.ceil(max_used_height))

        coords: list[tuple[float, float, float, float] | None] = [None] * N
        current_y = 0.0
        
        for r in range(best_num_rows):
            mask = (best_row_indices == r)
            row_items_count = items_in_row_2d[best_partition_idx, r].item()
            total_padding = max(0, row_items_count - 1) * padding
            
            row_w_pixels = row_widths_2d[best_partition_idx, r].item() * best_scale + total_padding
            row_h_unscaled = row_heights_2d[best_partition_idx, r].item()
            
            current_x = (final_w - row_w_pixels) / 2.0
            
            item_indices = torch.where(mask)[0].tolist()
            for i in item_indices:
                w_s = float(widths[i]) * float(norm_factors[i]) * best_scale
                h_s = float(heights[i]) * float(norm_factors[i]) * best_scale
                coords[i] = (current_x, current_y, w_s, h_s)
                current_x += w_s + padding
                
            current_y += row_h_unscaled * best_scale + padding
            
        return final_w, final_h, coords

    def _render_tensors(
        self, images: list[torch.Tensor], final_w: int, final_h: int, coords: list[tuple[float, float, float, float] | None], r: int, g: int, b: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = images[0].device
        N = len(images)
        max_batch = max(img.shape[0] for img in images)
        
        padded_images = []
        padded_masks = []
        
        for idx in range(N):
            img = images[idx].to(device)
            coord = coords[idx]
            if coord is None:
                continue
                
            cx, cy, cw, ch = coord
            
            cw_i, ch_i = int(cw), int(ch)
            if cw_i <= 0 or ch_i <= 0:
                padded_images.append(torch.zeros((max_batch, 3, final_h, final_w), device=device))
                padded_masks.append(torch.zeros((max_batch, 1, final_h, final_w), device=device))
                continue
                
            img_nchw = img.permute(0, 3, 1, 2)
            scaled_img = F.interpolate(img_nchw, size=(ch_i, cw_i), mode="bilinear", align_corners=False)
            
            pad_left = max(0, int(cx))
            pad_top = max(0, int(cy))
            pad_right = max(0, final_w - pad_left - cw_i)
            pad_bottom = max(0, final_h - pad_top - ch_i)
            
            padded = F.pad(scaled_img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
            
            if padded.shape[0] < max_batch:
                padded = padded.expand(max_batch, -1, -1, -1)
                
            padded_images.append(padded)
            
            # Generates a clean bounding-box representation of the placement
            ones = torch.ones((1, 1, ch_i, cw_i), dtype=torch.float32, device=device)
            padded_mask = F.pad(ones, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
                
            if padded_mask.shape[0] < max_batch:
                padded_mask = padded_mask.expand(max_batch, -1, -1, -1)
                
            padded_masks.append(padded_mask)

        combined_img_nchw = torch.stack(padded_images, dim=0).sum(dim=0)
        combined_image = combined_img_nchw.permute(0, 2, 3, 1)

        total_mask_nchw = torch.stack(padded_masks, dim=0).sum(dim=0)
        total_mask = total_mask_nchw.permute(0, 2, 3, 1).clamp(0, 1)
        
        bg_color = torch.tensor([r / 255.0, g / 255.0, b / 255.0], dtype=torch.float32, device=device)
        bg_canvas = bg_color.view(1, 1, 1, 3).expand(max_batch, final_h, final_w, 3)
        combined_image = combined_image + bg_canvas * (1.0 - total_mask)

        # Merge individually masked components into multi-batch mask outputs safely
        individual_masks = torch.cat(padded_masks, dim=0).squeeze(1)

        return combined_image, individual_masks

    @torch.inference_mode()
    def pack(self, target_width: int, target_height: int, padding: int, normalize_sizing: str = "true", fill_color: str = "#000000", **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            hex_color = fill_color.lstrip('#')
            fill_R, fill_G, fill_B = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            fill_R, fill_G, fill_B = 0, 0, 0

        images = self._extract_valid_images(kwargs)

        if not images:
            return self._fallback_empty_canvas(target_width, target_height, fill_R, fill_G, fill_B)

        layout_data = self._evaluate_partitions(images, target_width, target_height, padding, normalize_sizing)
        final_w, final_h, coords = self._generate_coordinates(images, padding, layout_data)
        
        return self._render_tensors(images, final_w, final_h, coords, fill_R, fill_G, fill_B)
