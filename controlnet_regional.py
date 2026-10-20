"""
ControlNet-Safe Regional Conditioning  (1-4 mask regions)
=========================================================

Two nodes that let you drive 1 to 4 masked regions, each with its own prompt
and its own ControlNet, while keeping every ControlNet hint aligned to its
region.

    Region 1 :  CLIPTextEncode + Mask + ControlNet + hint  --> [Region] --\
    Region 2 :  CLIPTextEncode + Mask + ControlNet + hint  --> [Region] ---\
    Region 3 :  CLIPTextEncode + Mask + ControlNet + hint  --> [Region] ----> [Combine] --> CONDITIONING --> KSampler +
    Region 4 :  CLIPTextEncode + Mask + ControlNet + hint  --> [Region] ---/

WHY THIS EXISTS
---------------
Stock "Conditioning (Set Mask)" in "mask bounds" mode makes the sampler crop
the latent to the mask bounding box. ControlNet's get_control() then rescales
its FULL-canvas hint to the cropped size but is never told the box OFFSET, so
the control signal is mispositioned (and distorted if the box aspect ratio
differs from the canvas). With several regions the misalignment stacks up.

HOW IT IS FIXED
---------------
The Combine node computes each region's bounding box itself, writes the
conditioning `area` directly (so the sampler does not recompute it), crops that
region's ControlNet hint to the IDENTICAL box, and applies the ControlNet with
the cropped hint. Area and hint come from one bbox per region, so they cannot
drift. ControlNet then only ever does a clean aspect-matched resize.

USAGE
-----
* One "ControlNet-Safe Region" node per region (control_net + control_hint are
  optional -- omit them for a plain regional prompt with no ControlNet).
* Feed the region outputs into "ControlNet-Safe Regions Combine" (region_1 is
  required, region_2..4 are optional).
* Wire Combine's CONDITIONING output to the KSampler's positive input.
* The negative prompt stays normal and full-canvas -- wire it straight to the
  KSampler negative; ControlNet here is applied to the positive regions only.

Install: drop this file into  ComfyUI/custom_nodes/  and restart ComfyUI.
"""

import torch
import node_helpers

from .gpu_ops import to_bhw_mask

# VAE spatial downscale factor (latent px = image px / DOWNSCALE).
# 8 for SD1.5, SDXL, SD3, Flux.
DOWNSCALE = 8


def _bbox_of_mask(mask2d, threshold=1e-6):
    """Half-open pixel bbox (x0, y0, x1, y1) of the nonzero region, or None."""
    nz = torch.nonzero(mask2d > threshold)
    if nz.numel() == 0:
        return None
    ys, xs = nz[:, 0], nz[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ---------------------------------------------------------------------------
# Node 1 -- bundle a single region
# ---------------------------------------------------------------------------
class ControlNetSafeRegion:
    """Pack one masked region (prompt + mask + optional ControlNet) into a REGION."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "control_hint": ("IMAGE",),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "cn_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "cn_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("CN_SAFE_REGION",)
    RETURN_NAMES = ("region",)
    FUNCTION = "build"
    CATEGORY = "conditioning/regional"

    def build(self, conditioning, mask, mask_strength, padding,
              control_net=None, control_hint=None, cn_strength=1.0,
              cn_start_percent=0.0, cn_end_percent=1.0):
        region = {
            "conditioning": conditioning,
            "mask": mask,
            "mask_strength": mask_strength,
            "padding": int(padding),
            "control_net": control_net,
            "control_hint": control_hint,
            "cn_strength": cn_strength,
            "cn_start": cn_start_percent,
            "cn_end": cn_end_percent,
        }
        return (region,)


# ---------------------------------------------------------------------------
# Node 2 -- crop hints, set areas, apply ControlNets, combine
# ---------------------------------------------------------------------------
class ControlNetSafeRegionsCombine:
    """Combine 1-4 regions into one CONDITIONING with aligned ControlNet hints."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "region_1": ("CN_SAFE_REGION",),
            },
            "optional": {
                "region_2": ("CN_SAFE_REGION",),
                "region_3": ("CN_SAFE_REGION",),
                "region_4": ("CN_SAFE_REGION",),
                "vae": ("VAE",),  # only needed for ControlNets that require a VAE
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "combine"
    CATEGORY = "conditioning/regional"

    def combine(self, region_1, region_2=None, region_3=None, region_4=None, vae=None):
        regions = [r for r in (region_1, region_2, region_3, region_4) if r is not None]
        out = []
        for r in regions:
            out.extend(self._process_region(r, vae))
        return (out,)

    # -- per region -----------------------------------------------------------
    def _process_region(self, r, vae):
        conditioning = r["conditioning"]
        mask = to_bhw_mask(r["mask"])
        m = mask[0]
        mh, mw = int(m.shape[-2]), int(m.shape[-1])

        bbox = _bbox_of_mask(m)

        # --- empty mask: behave as a plain full-canvas conditioning ---
        if bbox is None:
            c = node_helpers.conditioning_set_values(
                conditioning,
                {"mask": mask, "mask_strength": r["mask_strength"], "set_area_to_bounds": False},
            )
            return self._apply_controlnet(c, r, None, vae)

        x0, y0, x1, y1 = bbox
        pad = r["padding"]
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(mw, x1 + pad)
        y1 = min(mh, y1 + pad)

        # --- canonical rectangle in LATENT units ---
        Lw, Lh = max(1, mw // DOWNSCALE), max(1, mh // DOWNSCALE)
        X = x0 // DOWNSCALE                       # floor
        Y = y0 // DOWNSCALE
        x1_l = -(-x1 // DOWNSCALE)                # ceil
        y1_l = -(-y1 // DOWNSCALE)
        W = max(1, min(Lw, x1_l) - X)
        H = max(1, min(Lh, y1_l) - Y)
        area = (int(H), int(W), int(Y), int(X))

        # area written directly so the sampler does NOT recompute it;
        # full-resolution mask kept for soft-edge weighting inside the box.
        c = node_helpers.conditioning_set_values(
            conditioning,
            {
                "area": area,
                "set_area_to_bounds": False,
                "mask": mask,
                "mask_strength": r["mask_strength"],
            },
        )

        # --- crop this region's hint to the IDENTICAL rectangle ---
        # VRAM NOTE: keep the cropped hint on the CPU. set_cond_hint() (called
        # later by get_control during sampling) moves the hint to the GPU
        # itself, just-in-time. Holding N region hints in VRAM here would
        # inflate ComfyUI's pre-sampling memory estimate and can trigger it to
        # evict other models. .contiguous() (not .clone()) detaches the slice
        # from the full hint tensor so the big original can be freed.
        cropped_hint = None
        hint = r["control_hint"]
        if hint is not None:
            ih, iw = int(hint.shape[1]), int(hint.shape[2])
            hx0 = max(0, min(round(X / Lw * iw), iw - 1))
            hy0 = max(0, min(round(Y / Lh * ih), ih - 1))
            hx1 = max(hx0 + 1, min(round((X + W) / Lw * iw), iw))
            hy1 = max(hy0 + 1, min(round((Y + H) / Lh * ih), ih))
            cropped_hint = hint[:, hy0:hy1, hx0:hx1, :].to("cpu").contiguous()

        return self._apply_controlnet(c, r, cropped_hint, vae)

    # -- attach the region's ControlNet using the cropped hint ---------------
    def _apply_controlnet(self, conditioning, r, cropped_hint, vae):
        cn = r["control_net"]
        if cn is None or cropped_hint is None or r["cn_strength"] == 0:
            return conditioning  # plain regional prompt, no ControlNet

        hint = cropped_hint.movedim(-1, 1)        # [B,H,W,C] -> [B,C,H,W]
        rng = (r["cn_start"], r["cn_end"])

        out = []
        cnets = {}
        for t in conditioning:
            d = t[1].copy()
            prev = d.get("control", None)
            if prev in cnets:
                cnet = cnets[prev]
            else:
                # cn.copy() is required: each region needs its own cond_hint.
                # set_cond_hint stores the hint on the CPU copy; get_control
                # moves it to VRAM just-in-time during sampling, so no extra
                # hint tensor sits in VRAM before the run.
                cnet = cn.copy()
                try:
                    cnet.set_cond_hint(hint, r["cn_strength"], rng, vae=vae)
                except TypeError:
                    # older ComfyUI: set_cond_hint had no vae parameter
                    cnet.set_cond_hint(hint, r["cn_strength"], rng)
                cnet.set_previous_controlnet(prev)
                cnets[prev] = cnet
            d["control"] = cnet
            d["control_apply_to_uncond"] = False
            out.append([t[0], d])

        # drop our local references so the CPU-side hint copies are collectible
        del hint, cropped_hint, cnets
        return out


# NODE_CLASS_MAPPINGS = {
#     "ControlNetSafeRegion": ControlNetSafeRegion,
#     "ControlNetSafeRegionsCombine": ControlNetSafeRegionsCombine,
# }
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "ControlNetSafeRegion": "ControlNet-Safe Region",
#     "ControlNetSafeRegionsCombine": "ControlNet-Safe Regions Combine (1-4)",
# }