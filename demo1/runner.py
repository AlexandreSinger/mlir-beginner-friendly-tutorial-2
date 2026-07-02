import ctypes
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from inspect_sig import inspect_signature


def _make_descriptor_type(ndim: int):
    class Desc(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned",   ctypes.c_void_p),
            ("offset",    ctypes.c_int64),
            ("sizes",     ctypes.c_int64 * ndim),
            ("strides",   ctypes.c_int64 * ndim),
        ]
    return Desc


def _tensor_to_descriptor(t: torch.Tensor):
    t = t.contiguous()
    Desc = _make_descriptor_type(t.ndim)
    desc = Desc()
    ptr  = ctypes.c_void_p(t.data_ptr())
    desc.allocated = ptr
    desc.aligned   = ptr
    desc.offset    = 0
    for i, (s, st) in enumerate(zip(t.shape, t.stride())):
        desc.sizes[i]   = s
        desc.strides[i] = st
    return desc, t  # return t to keep it alive


def _make_packed_output_type(outputs: list[tuple[tuple, torch.dtype]]):
    """Build a ctypes struct that packs all output descriptors sequentially."""
    fields = [(f"out{i}", _make_descriptor_type(len(shape)))
              for i, (shape, _) in enumerate(outputs)]
    class PackedOutputs(ctypes.Structure):
        _fields_ = fields
    return PackedOutputs


_TORCH_TO_CTYPES = {
    torch.float16: ctypes.c_int16,
    torch.float32: ctypes.c_float,
    torch.float64: ctypes.c_double,
    torch.int8:    ctypes.c_int8,
    torch.int16:   ctypes.c_int16,
    torch.int32:   ctypes.c_int32,
    torch.int64:   ctypes.c_int64,
    torch.uint8:   ctypes.c_uint8,
    torch.bool:    ctypes.c_bool,
}


def _descriptor_to_tensor(desc, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    total   = 1
    for s in shape:
        total *= s
    c_dtype  = _TORCH_TO_CTYPES[dtype]
    arr_type = c_dtype * total
    arr      = arr_type.from_address(desc.aligned)
    return torch.frombuffer(arr, dtype=dtype).reshape(shape).clone()


class MLIRRunner:
    """
    Calls the compiled model.so produced by the pipeline.
    Automatically detects input/output shapes from the MLIR signature.
    """

    def __init__(self, out_dir: str | Path, func_name: str = "main"):
        out_dir = Path(out_dir)
        so_path = out_dir / "model.so"
        if not so_path.exists():
            raise FileNotFoundError(f"No compiled model at {so_path}")

        self._lib       = ctypes.CDLL(str(so_path))
        self._func_name = func_name
        self._inputs, self._outputs = inspect_signature(out_dir)

    def run(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        fn = getattr(self._lib, f"_mlir_ciface_{self._func_name}")
        fn.restype  = None
        fn.argtypes = None

        PackedOutput = _make_packed_output_type(self._outputs)
        packed_out   = PackedOutput()
        descs_and_refs = [_tensor_to_descriptor(t) for t in inputs]
        in_descs = [d for d, _ in descs_and_refs]

        args = [ctypes.byref(packed_out)] + [ctypes.byref(d) for d in in_descs]
        fn(*args)

        return [
            _descriptor_to_tensor(getattr(packed_out, f"out{i}"), shape, dtype)
            for i, (shape, dtype) in enumerate(self._outputs)
        ]

    @property
    def input_shapes(self):
        return [(s, d) for s, d in self._inputs]

    @property
    def output_shapes(self):
        return [(s, d) for s, d in self._outputs]


class PyTorchRunner:
    """Runs the original PyTorch model as the golden reference."""

    def __init__(self, model: nn.Module):
        self._model = model
        self._model.eval()

    def run(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        with torch.no_grad():
            out = self._model(*inputs)
        if isinstance(out, torch.Tensor):
            return [out]
        return list(out)

    def save(self, inputs: list[torch.Tensor], out_dir: str | Path) -> list[torch.Tensor]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        results = self.run(inputs)
        for i, r in enumerate(results):
            np.save(out / f"golden_output_{i}.npy", r.numpy())
        for i, inp in enumerate(inputs):
            np.save(out / f"input_{i}.npy", inp.numpy())
        return results


def compare(
    golden: list[torch.Tensor],
    actual: list[torch.Tensor],
    atol: float = 1e-4,
) -> bool:
    all_match = True
    for i, (g, a) in enumerate(zip(golden, actual)):
        match   = torch.allclose(g, a, atol=atol)
        max_err = float((g - a).abs().max())
        print(f"  output[{i}]  match={match}  max_err={max_err:.6e}  atol={atol:.6e}")
        all_match = all_match and match
    return all_match