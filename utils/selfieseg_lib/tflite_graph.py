"""TFLite flatbuffer -> trainable torch module, for the op subset small MediaPipe
vision models use.

Two halves, split so a converted model never needs the .tflite (or the flatbuffer
bindings) again:

  parse_tflite(path) -> (spec, weights)
      Reads the flatbuffer with the `tflite` schema bindings (pure python, tiny).
      `spec` is a JSON-serializable description of the graph — one entry per op with
      everything needed to rebuild it (kernel, stride, padding mode, ...) but no
      weights; `weights` is a state_dict-shaped {name: tensor}. fp16 constants behind
      DEQUANTIZE are folded into fp32 parameters.

  TFLiteNet(spec)
      A torch module that executes the spec in graph order. Every CONV_2D /
      DEPTHWISE_CONV_2D / transpose conv becomes a real nn.Conv2d-family module, so
      the weights are ordinary trainable parameters and the whole thing fine-tunes,
      exports to ONNX, etc. like any hand-written network.

Layout: TFLite is NHWC, torch is NCHW. The caller feeds NCHW; constants are permuted
once at parse time. TF "SAME" padding is asymmetric (the odd pixel goes bottom/right)
and is reproduced exactly by padding at runtime from the actual input size — which
also makes the module resolution-agnostic: global average pools are detected as such
(filter == input extent) and rebuilt as adaptive pools, and bilinear resizes are
stored as scale factors rather than absolute sizes.

Unsupported ops raise NotImplementedError naming the op — extend `_parse_node` /
`_build_layer` rather than special-casing callers.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SPEC_VERSION = 1

# fused-activation enum (tflite.ActivationFunctionType) -> name used in the spec
_FUSED_ACT = {0: None, 1: "relu", 2: "relu_n1_to_1", 3: "relu6", 4: "tanh"}
_PADDING = {0: "same", 1: "valid"}
_NP_DTYPES = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64,
              6: np.bool_, 9: np.int8}

# ------------------------------------------------------------------ TF SAME padding

def same_pad_amounts(size: int, kernel: int, stride: int, dilation: int = 1) -> tuple[int, int]:
    """(before, after) padding along one axis for TF/TFLite "SAME": output is
    ceil(size / stride), and when the total padding is odd the extra pixel goes
    AFTER (bottom/right) — the opposite of what a symmetric torch padding would do."""
    effective = (kernel - 1) * dilation + 1
    out = -(-size // stride)
    total = max((out - 1) * stride + effective - size, 0)
    return total // 2, total - total // 2


def _same_pad(x: torch.Tensor, kernel: tuple[int, int], stride: tuple[int, int],
              dilation: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad x for a SAME conv. Returns (x, symmetric_padding): when the required padding
    is symmetric, x is returned untouched and the conv applies it natively (no copy);
    otherwise x is explicitly padded and the conv pads nothing."""
    pt, pb = same_pad_amounts(x.shape[-2], kernel[0], stride[0], dilation[0])
    pl, pr = same_pad_amounts(x.shape[-1], kernel[1], stride[1], dilation[1])
    if pt == pb and pl == pr:
        return x, (pt, pl)
    return F.pad(x, (pl, pr, pt, pb)), (0, 0)


class Conv2dTF(nn.Conv2d):
    """nn.Conv2d with TFLite padding semantics ("same" = asymmetric TF SAME, "valid")."""

    def __init__(self, *args: Any, tf_padding: str = "valid", **kwargs: Any):
        super().__init__(*args, padding=0, **kwargs)
        if tf_padding not in ("same", "valid"):
            raise ValueError(f"tf_padding must be 'same' or 'valid', got {tf_padding!r}")
        self.tf_padding = tf_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tf_padding == "same":
            x, pad = _same_pad(x, self.kernel_size, self.stride, self.dilation)
        else:
            pad = (0, 0)
        return F.conv2d(x, self.weight, self.bias, self.stride, pad, self.dilation, self.groups)


def _act(name: str | None, x: torch.Tensor) -> torch.Tensor:
    if name is None:
        return x
    if name == "relu":
        return F.relu(x)
    if name == "relu6":
        return F.relu6(x)
    if name == "relu_n1_to_1":
        return x.clamp(-1.0, 1.0)
    if name == "tanh":
        return torch.tanh(x)
    if name == "hardswish":
        return F.hardswish(x)
    if name == "logistic":
        return torch.sigmoid(x)
    raise ValueError(f"unknown activation {name!r}")


# ------------------------------------------------------------------------- parsing

class _Reader:
    """Thin view over the flatbuffer: tensors, constants (DEQUANTIZE folded), options."""

    def __init__(self, buf: bytes):
        import tflite  # lazy: only `convert` needs the flatbuffer bindings

        self.tflite = tflite
        self.model = tflite.Model.GetRootAsModel(buf, 0)
        if self.model.SubgraphsLength() != 1:
            raise NotImplementedError(f"expected 1 subgraph, got {self.model.SubgraphsLength()}")
        self.graph = self.model.Subgraphs(0)
        self.op_names: list[str] = []
        self.custom_names: list[str | None] = []
        by_value = {v: k for k, v in vars(tflite.BuiltinOperator).items() if isinstance(v, int)}
        for i in range(self.model.OperatorCodesLength()):
            code = self.model.OperatorCodes(i)
            builtin = code.BuiltinCode() or code.DeprecatedBuiltinCode()
            self.op_names.append(by_value.get(builtin, str(builtin)))
            custom = code.CustomCode()
            self.custom_names.append(custom.decode() if custom else None)
        # tensor id -> fp32 numpy constant; DEQUANTIZE outputs are added during parsing
        self.constants: dict[int, np.ndarray] = {}
        for t in range(self.graph.TensorsLength()):
            arr = self._buffer(t)
            if arr is not None:
                self.constants[t] = arr

    def shape(self, t: int) -> list[int]:
        tensor = self.graph.Tensors(t)
        return [int(v) for v in tensor.ShapeAsNumpy()] if tensor.ShapeLength() else []

    def name(self, t: int) -> str:
        return self.graph.Tensors(t).Name().decode()

    def _buffer(self, t: int) -> np.ndarray | None:
        tensor = self.graph.Tensors(t)
        buf = self.model.Buffers(tensor.Buffer())
        if not buf.DataLength():
            return None
        dtype = _NP_DTYPES.get(tensor.Type())
        if dtype is None:
            raise NotImplementedError(f"tensor {t} ({self.name(t)}): unsupported dtype {tensor.Type()}")
        arr = np.frombuffer(buf.DataAsNumpy().tobytes(), dtype=dtype).reshape(self.shape(t))
        if dtype == np.float16:
            arr = arr.astype(np.float32)
        return arr

    def const(self, t: int) -> np.ndarray:
        if t not in self.constants:
            raise ValueError(f"tensor {t} ({self.name(t)}) is not a constant")
        return self.constants[t]

    def options(self, op, cls_name: str):
        opts = getattr(self.tflite, cls_name)()
        table = op.BuiltinOptions()
        opts.Init(table.Bytes, table.Pos)
        return opts


def _nhwc_to_nchw_const(arr: np.ndarray) -> np.ndarray:
    """Constants that broadcast against activations: [1,1,1,C] / [N,H,W,C] -> NCHW,
    [C] -> [1,C,1,1]. Scalars pass through."""
    if arr.ndim == 4:
        return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
    if arr.ndim == 1:
        return arr.reshape(1, -1, 1, 1)
    if arr.ndim == 0:
        return arr
    raise NotImplementedError(f"cannot map a {arr.ndim}-D broadcast constant to NCHW")


def parse_tflite(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Parse a .tflite file into (spec, weights). See the module docstring."""
    path = Path(path)
    buf = path.read_bytes()
    r = _Reader(buf)
    g = r.graph
    inputs = [int(g.Inputs(i)) for i in range(g.InputsLength())]
    outputs = [int(g.Outputs(i)) for i in range(g.OutputsLength())]
    if len(inputs) != 1 or len(outputs) != 1:
        raise NotImplementedError(f"expected 1 input / 1 output, got {inputs} / {outputs}")

    nodes: list[dict[str, Any]] = []
    weights: dict[str, torch.Tensor] = {}
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        name = r.op_names[op.OpcodeIndex()]
        if name == "DEQUANTIZE":
            # fold: the output tensor id resolves to the (already fp32) constant
            r.constants[int(op.Outputs(0))] = r.const(int(op.Inputs(0)))
            continue
        nodes.append(_parse_node(r, i, name, op, weights))

    spec = {
        "spec_version": SPEC_VERSION,
        "source": {"file": path.name, "sha256": hashlib.sha256(buf).hexdigest(),
                   "description": (r.model.Description() or b"").decode()},
        "input": inputs[0],
        "input_shape": r.shape(inputs[0]),        # NHWC, as in the file
        "output": outputs[0],
        "nodes": nodes,
    }
    return spec, weights


def _parse_node(r: _Reader, i: int, name: str, op, weights: dict[str, torch.Tensor]) -> dict[str, Any]:
    ins = [int(op.Inputs(j)) for j in range(op.InputsLength())]
    outs = [int(op.Outputs(j)) for j in range(op.OutputsLength())]
    key = f"n{i:03d}"
    node: dict[str, Any] = {"id": key, "op": name, "inputs": ins, "outputs": outs}

    if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
        depthwise = name == "DEPTHWISE_CONV_2D"
        o = r.options(op, "DepthwiseConv2DOptions" if depthwise else "Conv2DOptions")
        w = r.const(ins[1])
        bias = r.const(ins[2]) if len(ins) > 2 and ins[2] >= 0 else None
        if depthwise:
            # [1, kh, kw, ic*dm] -> [ic*dm, 1, kh, kw], groups = ic
            in_ch = r.shape(ins[0])[-1]
            out_ch = w.shape[-1]
            if out_ch != in_ch * o.DepthMultiplier():
                raise ValueError(f"{key}: depthwise channel mismatch {out_ch} vs {in_ch}x{o.DepthMultiplier()}")
            torch_w = np.ascontiguousarray(w.transpose(3, 0, 1, 2))
            groups = in_ch
        else:
            # [oc, kh, kw, ic] -> [oc, ic, kh, kw]
            torch_w = np.ascontiguousarray(w.transpose(0, 3, 1, 2))
            in_ch, out_ch, groups = w.shape[3], w.shape[0], 1
        node.update(
            layer="conv", in_ch=int(in_ch), out_ch=int(out_ch), groups=int(groups),
            kernel=[int(w.shape[1]), int(w.shape[2])],
            stride=[int(o.StrideH()), int(o.StrideW())],
            dilation=[int(o.DilationHFactor()), int(o.DilationWFactor())],
            padding=_PADDING[o.Padding()], bias=bias is not None,
            act=_FUSED_ACT[o.FusedActivationFunction()],
        )
        weights[f"layers.{key}.conv.weight"] = torch.from_numpy(torch_w)
        if bias is not None:
            weights[f"layers.{key}.conv.bias"] = torch.from_numpy(np.ascontiguousarray(bias))
        node["inputs"] = ins[:1]
        return node

    if name == "CUSTOM":
        custom = r.custom_names[op.OpcodeIndex()]
        if custom != "Convolution2DTransposeBias":
            raise NotImplementedError(f"{key}: custom op {custom!r} is not supported")
        # MediaPipe's transpose conv + bias: custom_options = TfLiteTransposeConvParams
        # {padding (1=SAME, 2=VALID), stride_w, stride_h}; weights [oc, kh, kw, ic]
        raw = bytes(op.CustomOptionsAsNumpy())
        padding_code, stride_w, stride_h = struct.unpack("<3i", raw[:12])
        w = r.const(ins[1])
        bias = r.const(ins[2]) if len(ins) > 2 and ins[2] >= 0 else None
        kh, kw = int(w.shape[1]), int(w.shape[2])
        if (stride_h, stride_w) != (kh, kw):
            raise NotImplementedError(
                f"{key}: transpose conv with kernel {kh}x{kw} != stride {stride_h}x{stride_w} "
                "needs output-padding handling that is not implemented"
            )
        node.update(
            layer="conv_transpose", in_ch=int(w.shape[3]), out_ch=int(w.shape[0]),
            kernel=[kh, kw], stride=[int(stride_h), int(stride_w)], bias=bias is not None,
            padding={1: "same", 2: "valid"}.get(padding_code, "same"), act=None,
        )
        # [oc, kh, kw, ic] -> torch ConvTranspose2d [ic, oc, kh, kw]
        weights[f"layers.{key}.conv.weight"] = torch.from_numpy(np.ascontiguousarray(w.transpose(3, 0, 1, 2)))
        if bias is not None:
            weights[f"layers.{key}.conv.bias"] = torch.from_numpy(np.ascontiguousarray(bias))
        node["inputs"] = ins[:1]
        return node

    if name in ("ADD", "MUL", "SUB"):
        o = r.options(op, {"ADD": "AddOptions", "MUL": "MulOptions", "SUB": "SubOptions"}[name])
        node.update(layer="binary", fn=name.lower(), act=_FUSED_ACT[o.FusedActivationFunction()])
        # a constant operand becomes a buffer; its position is remembered (SUB is ordered)
        const_slots = [j for j, t in enumerate(ins) if t in r.constants]
        if len(const_slots) == 2:
            raise NotImplementedError(f"{key}: {name} of two constants")
        if const_slots:
            j = const_slots[0]
            const = _nhwc_to_nchw_const(r.const(ins[j])).astype(np.float32)
            weights[f"layers.{key}.const"] = torch.from_numpy(np.ascontiguousarray(const))
            node["const_slot"] = j
            node["const_shape"] = list(const.shape)
            node["inputs"] = [t for jj, t in enumerate(ins) if jj != j]
        return node

    if name in ("RELU", "RELU6", "HARD_SWISH", "LOGISTIC", "TANH"):
        node.update(layer="act", act={"RELU": "relu", "RELU6": "relu6", "HARD_SWISH": "hardswish",
                                      "LOGISTIC": "logistic", "TANH": "tanh"}[name])
        return node

    if name in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
        o = r.options(op, "Pool2DOptions")
        in_shape = r.shape(ins[0])
        fh, fw = int(o.FilterHeight()), int(o.FilterWidth())
        node.update(layer="pool", fn="avg" if name == "AVERAGE_POOL_2D" else "max",
                    act=_FUSED_ACT[o.FusedActivationFunction()])
        if len(in_shape) == 4 and (fh, fw) == (in_shape[1], in_shape[2]):
            node["global"] = True   # the whole extent: rebuild as an adaptive pool
        else:
            if _PADDING[o.Padding()] != "valid":
                raise NotImplementedError(f"{key}: non-global {name} with SAME padding")
            node.update(**{"global": False}, kernel=[fh, fw], stride=[int(o.StrideH()), int(o.StrideW())])
        return node

    if name in ("RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"):
        o = r.options(op, "ResizeBilinearOptions" if name == "RESIZE_BILINEAR" else "ResizeNearestNeighborOptions")
        in_shape = r.shape(ins[0])
        size = [int(v) for v in r.const(ins[1])]
        if len(in_shape) != 4:
            raise NotImplementedError(f"{key}: resize of a {len(in_shape)}-D tensor")
        if name == "RESIZE_BILINEAR" and not o.HalfPixelCenters() and not o.AlignCorners():
            raise NotImplementedError(f"{key}: legacy TF resize (neither half-pixel nor align-corners)")
        node.update(layer="resize", mode="bilinear" if name == "RESIZE_BILINEAR" else "nearest",
                    scale=[size[0] / in_shape[1], size[1] / in_shape[2]],
                    align_corners=bool(o.AlignCorners()))
        node["inputs"] = ins[:1]
        return node

    raise NotImplementedError(
        f"{key}: tflite op {name} is not supported by the converter (see tflite_graph._parse_node)"
    )


# ------------------------------------------------------------------------ building

class _Binary(nn.Module):
    def __init__(self, fn: str, act: str | None, const_slot: int | None, const_shape: list[int] | None):
        super().__init__()
        self.fn, self.act, self.const_slot = fn, act, const_slot
        if const_shape is not None:
            self.register_buffer("const", torch.zeros(const_shape))

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        operands = list(xs)
        if self.const_slot is not None:
            operands.insert(self.const_slot, self.const)
        a, b = operands
        if self.fn == "add":
            y = a + b
        elif self.fn == "mul":
            y = a * b
        else:
            y = a - b
        return _act(self.act, y)


class _Act(nn.Module):
    def __init__(self, act: str):
        super().__init__()
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _act(self.act, x)


class _Pool(nn.Module):
    def __init__(self, node: dict[str, Any]):
        super().__init__()
        self.fn, self.act, self.is_global = node["fn"], node.get("act"), node["global"]
        if not self.is_global:
            self.kernel, self.stride = tuple(node["kernel"]), tuple(node["stride"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_global:
            y = F.adaptive_avg_pool2d(x, 1) if self.fn == "avg" else F.adaptive_max_pool2d(x, 1)
        elif self.fn == "avg":
            y = F.avg_pool2d(x, self.kernel, self.stride)
        else:
            y = F.max_pool2d(x, self.kernel, self.stride)
        return _act(self.act, y)


class _Resize(nn.Module):
    def __init__(self, node: dict[str, Any]):
        super().__init__()
        self.mode, self.scale, self.align = node["mode"], tuple(node["scale"]), node["align_corners"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kw = {"align_corners": self.align} if self.mode == "bilinear" else {}
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode, recompute_scale_factor=False, **kw)


class _ConvAct(nn.Module):
    def __init__(self, conv: nn.Module, act: str | None):
        super().__init__()
        self.conv, self.act = conv, act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _act(self.act, self.conv(x))


def _build_layer(node: dict[str, Any]) -> nn.Module:
    layer = node["layer"]
    if layer == "conv":
        conv = Conv2dTF(node["in_ch"], node["out_ch"], tuple(node["kernel"]), stride=tuple(node["stride"]),
                        dilation=tuple(node["dilation"]), groups=node["groups"], bias=node["bias"],
                        tf_padding=node["padding"])
        return _ConvAct(conv, node.get("act"))
    if layer == "conv_transpose":
        conv = nn.ConvTranspose2d(node["in_ch"], node["out_ch"], tuple(node["kernel"]),
                                  stride=tuple(node["stride"]), bias=node["bias"])
        return _ConvAct(conv, node.get("act"))
    if layer == "binary":
        return _Binary(node["fn"], node.get("act"), node.get("const_slot"), node.get("const_shape"))
    if layer == "act":
        return _Act(node["act"])
    if layer == "pool":
        return _Pool(node)
    if layer == "resize":
        return _Resize(node)
    raise ValueError(f"unknown layer kind {layer!r}")


class TFLiteNet(nn.Module):
    """Executes a parsed spec. Input/output are NCHW float tensors; the graph's own
    input scaling (e.g. /255) is NOT part of the spec — callers apply it."""

    def __init__(self, spec: dict[str, Any]):
        super().__init__()
        if spec.get("spec_version") != SPEC_VERSION:
            raise ValueError(f"spec version {spec.get('spec_version')} != {SPEC_VERSION}")
        self.spec = spec
        self.layers = nn.ModuleDict({n["id"]: _build_layer(n) for n in spec["nodes"]})
        # last use of every tensor, so intermediates are dropped as soon as possible
        last_use: dict[int, int] = {}
        for idx, n in enumerate(spec["nodes"]):
            for t in n["inputs"]:
                last_use[t] = idx
        self._free_after: list[list[int]] = [[] for _ in spec["nodes"]]
        for t, idx in last_use.items():
            self._free_after[idx].append(t)

    @property
    def input_hw(self) -> tuple[int, int]:
        """The resolution the graph was exported at (from the NHWC input shape)."""
        s = self.spec["input_shape"]
        return int(s[1]), int(s[2])

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        """logits=True stops before a trailing LOGISTIC node (train with BCE-with-logits)."""
        nodes = self.spec["nodes"]
        stop = len(nodes)
        if logits and nodes and nodes[-1]["op"] == "LOGISTIC":
            stop -= 1
        tensors: dict[int, torch.Tensor] = {self.spec["input"]: x}
        y = x
        for idx in range(stop):
            n = nodes[idx]
            y = self.layers[n["id"]](*(tensors[t] for t in n["inputs"]))
            for t in n["outputs"]:
                tensors[t] = y
            for t in self._free_after[idx]:
                tensors.pop(t, None)
        return y
