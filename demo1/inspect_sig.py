import re
from pathlib import Path

import torch

DTYPE_MAP = {
    "f16":  torch.float16,
    "f32":  torch.float32,
    "f64":  torch.float64,
    "i1":   torch.bool,
    "i8":   torch.int8,
    "i16":  torch.int16,
    "i32":  torch.int32,
    "i64":  torch.int64,
    "si32": torch.int32,
    "si64": torch.int64,
    "ui8":  torch.uint8,
}


def _parse_memref(type_str: str) -> tuple[tuple, torch.dtype] | None:
    m = re.match(r"memref<([^>]+)>", type_str.strip())
    if not m:
        return None
    inner = m.group(1).split(",")[0].strip()
    parts = inner.split("x")
    dtype_str = parts[-1]
    if dtype_str not in DTYPE_MAP:
        return None
    shape = tuple(int(d) if d != "?" else -1 for d in parts[:-1])
    return shape, DTYPE_MAP[dtype_str]


def inspect_signature(out_dir: str | Path) -> tuple[list, list]:
    """
    Parse 04_bufferized.mlir to extract the main function's
    input and output (shape, dtype) pairs.

    Returns:
        inputs:  list of (shape, torch.dtype)
        outputs: list of (shape, torch.dtype)
    """
    mlir_text = (Path(out_dir) / "04_bufferized.mlir").read_text()

    func_re = re.compile(
        r"func\.func\s+@\w+\(([^)]*)\)\s*"
        r"(?:->\s*([\w<>\[\](),\s\?:!]+?))?\s*(?:attributes)?\s*\{",
        re.DOTALL,
    )
    m = func_re.search(mlir_text)
    if not m:
        raise ValueError("Could not find function signature in 04_bufferized.mlir")

    args_str    = m.group(1) or ""
    returns_str = (m.group(2) or "").strip().strip("()")

    inputs = []
    for tok in re.finditer(r"memref<[^>]+>", args_str):
        parsed = _parse_memref(tok.group(0))
        if parsed:
            inputs.append(parsed)

    outputs = []
    for tok in re.finditer(r"memref<[^>]+>", returns_str):
        parsed = _parse_memref(tok.group(0))
        if parsed:
            outputs.append(parsed)

    return inputs, outputs