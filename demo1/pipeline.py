import re
import subprocess
import tempfile
from pathlib import Path

BUILD_BIN        = Path(__file__).parent.parent / "build/bin"
TORCH_MLIR_OPT   = BUILD_BIN / "torch-mlir-opt"
MLIR_OPT         = BUILD_BIN / "mlir-opt"
MLIR_TRANSLATE   = BUILD_BIN / "mlir-translate"


def _run(binary: Path, mlir_text: str, *flags: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(mlir_text)
        tmp_path = f.name

    result = subprocess.run(
        [str(binary), tmp_path, *flags],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{binary.name} failed:\n{result.stderr}")
    return result.stdout


def _dump(mlir_text: str, out_dir: Path | None, filename: str) -> None:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / filename).write_text(mlir_text)


def torch_to_tosa(mlir_text: str) -> str:
    mlir_text = _run(
        TORCH_MLIR_OPT,
        mlir_text,
        "--pass-pipeline=builtin.module(torchdynamo-export-to-torch-backend-pipeline)",
    )
    return _run(
        TORCH_MLIR_OPT,
        mlir_text,
        "--pass-pipeline=builtin.module(torch-backend-to-tosa-backend-pipeline)",
    )


def tosa_to_linalg(mlir_text: str) -> str:
    return _run(
        MLIR_OPT,
        mlir_text,
        "--pass-pipeline=builtin.module("
        "  func.func(tosa-to-linalg-named),"
        "  func.func(tosa-to-linalg),"
        "  func.func(tosa-to-arith),"
        "  func.func(tosa-to-scf),"
        "  func.func(tosa-to-tensor)"
        ")",
    )


def bufferize(mlir_text: str) -> str:
    return _run(
        MLIR_OPT,
        mlir_text,
        "--pass-pipeline=builtin.module("
        "  one-shot-bufferize{bufferize-function-boundaries=true},"
        "  func.func(buffer-deallocation-pipeline)"
        ")",
    )


def linalg_to_scf(mlir_text: str) -> str:
    return _run(
        MLIR_OPT,
        mlir_text,
        "--pass-pipeline=builtin.module("
        "  func.func(convert-linalg-to-loops,"
        "            lower-affine,"
        "            convert-scf-to-cf)"
        ")",
    )


def _inject_c_interface(mlir_text: str) -> str:
    """Add llvm.emit_c_interface to every func.func so convert-func-to-llvm
    generates _mlir_ciface_* wrappers with pointer-to-struct ABI."""
    lines = []
    for line in mlir_text.splitlines():
        if re.match(r"\s*func\.func\s+@", line) and line.rstrip().endswith("{"):
            if "attributes {" in line:
                line = line.replace("attributes {", "attributes {llvm.emit_c_interface, ", 1)
            else:
                line = line.rstrip()[:-1].rstrip() + " attributes {llvm.emit_c_interface} {"
        lines.append(line)
    return "\n".join(lines)


def to_llvm_dialect(mlir_text: str) -> str:
    mlir_text = _inject_c_interface(mlir_text)
    return _run(
        MLIR_OPT,
        mlir_text,
        "--pass-pipeline=builtin.module("
        "  expand-strided-metadata,"
        "  lower-affine,"
        "  convert-math-to-llvm,"
        "  convert-arith-to-llvm,"
        "  convert-cf-to-llvm,"
        "  finalize-memref-to-llvm,"
        "  convert-func-to-llvm,"
        "  convert-index-to-llvm,"
        "  reconcile-unrealized-casts"
        ")",
    )


def to_llvmir(mlir_text: str) -> str:
    return _run(MLIR_TRANSLATE, mlir_text, "--mlir-to-llvmir")


def compile_so(llvmir_text: str, out_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(llvmir_text)
        tmp_ll = f.name

    tmp_obj = tmp_ll.replace(".ll", ".o")

    # Use the build's llc to avoid LLVM IR attribute version mismatch
    llc = BUILD_BIN / "llc"
    r = subprocess.run(
        [str(llc), "-O2", "-filetype=obj", "--relocation-model=pic",
         tmp_ll, "-o", tmp_obj],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"llc failed:\n{r.stderr}")

    build_lib = BUILD_BIN.parent / "lib"
    r = subprocess.run(
        [
            "clang", "-shared", "-fPIC", "-o", str(out_path), tmp_obj,
            f"-L{build_lib}",
            "-lmlir_runner_utils",
            "-lmlir_c_runner_utils",
            f"-Wl,-rpath,{build_lib}",
        ],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"clang link failed:\n{r.stderr}")

    return out_path


def compile(
    mlir_text: str,
    out_dir: str | Path | None = None,
) -> str:
    out = Path(out_dir) if out_dir else None

    _dump(mlir_text, out, "01_torch.mlir")

    tosa = torch_to_tosa(mlir_text)
    _dump(tosa, out, "02_tosa.mlir")

    linalg = tosa_to_linalg(tosa)
    _dump(linalg, out, "03_linalg.mlir")

    buffered = bufferize(linalg)
    _dump(buffered, out, "04_bufferized.mlir")

    scf = linalg_to_scf(buffered)
    _dump(scf, out, "05_scf_arith_memref.mlir")

    llvm = to_llvm_dialect(scf)
    _dump(llvm, out, "06_llvm_dialect.mlir")

    llvmir = to_llvmir(llvm)
    _dump(llvmir, out, "07_llvmir.ll")

    if out:
        return compile_so(llvmir, out / "model.so")

    return None