from .fill_transparency_node import FillTransparencyNode
from .mask_expand_border import MaskExpandBorder, MaskExpandBorderAdvanced
from .mask_from_contour_node import MaskFromContoursOpenCV
from .tranasparency_with_mask import MaskToTransparentNode, DebugMaskNode
from .cutout_transparency_node import WhiteToTransparentNode, SimpleWhiteDetectorNode

NODE_CLASS_MAPPINGS = {
    "FillTransparencyNode": FillTransparencyNode,
    "MaskFromContoursOpenCV": MaskFromContoursOpenCV,
    "MaskToTransparentNode": MaskToTransparentNode,
    "DebugMaskNode": DebugMaskNode,
    "MaskExpandBorder": MaskExpandBorder,
    "MaskExpandBorderAdvanced": MaskExpandBorderAdvanced,
    "WhiteToTransparentNode": WhiteToTransparentNode,
    "SimpleWhiteDetectorNode": SimpleWhiteDetectorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillTransparencyNode": "Fill Transparency",
    "MaskExpandBorder": "Mask Expand Border",
    "MaskExpandBorderAdvanced": "Mask Expand Border (Advanced)",
    "MaskToTransparentNode": "Mask to Transparent",
    "DebugMaskNode": "Debug Mask Visualizer",
    "WhiteToTransparentNode": "White to Transparent",
    "SimpleWhiteDetectorNode": "White Detector",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
