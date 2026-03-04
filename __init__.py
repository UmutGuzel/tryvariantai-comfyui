from __future__ import annotations

from .fill_transparency_node import FillTransparencyNode
from .mask_expand_border import MaskExpandBorder, MaskExpandBorderAdvanced
from .mask_from_contour_node import MaskFromCounterOpenCV, MaskFromCounter
from .tranasparency_with_mask import MaskToTransparentNode, DebugMaskNode
from .cutout_transparency_node import WhiteToTransparentNode, SimpleWhiteDetectorNode
from .rgba_to_rgb_node import RGBAtoRGBNode
from .mask_shrink_node import MaskShrinkNode
from .mask_morph_node import MaskMorphNode
from .mask_cleanup_node import MaskCleanupNode
from .base64_decode_node import Base64DecodeNode
from .mask_matting_node import MaskMattingNode
from .birefnet_segmentation_node import BiRefNetSegmentationNode
from .image_mask_convert_node import ImageToMaskNode, MaskToImageNode

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "FillTransparencyNode": FillTransparencyNode,
    "MaskFromContoursOpenCV": MaskFromCounterOpenCV,
    "MaskFromContour": MaskFromCounter,
    "MaskToTransparentNode": MaskToTransparentNode,
    "DebugMaskNode": DebugMaskNode,
    "MaskExpandBorder": MaskExpandBorder,
    "MaskExpandBorderAdvanced": MaskExpandBorderAdvanced,
    "WhiteToTransparentNode": WhiteToTransparentNode,
    "SimpleWhiteDetectorNode": SimpleWhiteDetectorNode,
    "RGBAtoRGBNode": RGBAtoRGBNode,
    "MaskShrinkNode": MaskShrinkNode,
    "MaskMorphNode": MaskMorphNode,
    "MaskCleanupNode": MaskCleanupNode,
    "Base64DecodeNode": Base64DecodeNode,
    "MaskMattingNode": MaskMattingNode,
    "BiRefNetSegmentationNode": BiRefNetSegmentationNode,
    "ImageToMaskNode": ImageToMaskNode,
    "MaskToImageNode": MaskToImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "FillTransparencyNode": "Fill Transparency",
    "MaskExpandBorder": "Mask Expand Border",
    "MaskExpandBorderAdvanced": "Mask Expand Border (Advanced)",
    "MaskToTransparentNode": "Mask to Transparent",
    "MaskFromContour": "Mask From Counter",
    "MaskFromContoursOpenCV": "MaskFromCounterOpenCV",
    "DebugMaskNode": "Debug Mask Visualizer",
    "WhiteToTransparentNode": "White to Transparent",
    "SimpleWhiteDetectorNode": "White Detector",
    "RGBAtoRGBNode": "RGBA to RGB",
    "MaskShrinkNode": "Mask Shrink",
    "MaskMorphNode": "Mask Morph (Expand/Shrink)",
    "MaskCleanupNode": "Mask Cleanup (Morphological)",
    "Base64DecodeNode": "Base64 Decode Image",
    "MaskMattingNode": "Alpha Matting (Mask Refinement)",
    "BiRefNetSegmentationNode": "BiRefNet Segmentation",
    "ImageToMaskNode": "Image to Mask",
    "MaskToImageNode": "Mask to Image",
}


__all__: list[str] = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
