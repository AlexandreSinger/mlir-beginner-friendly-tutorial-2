import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from frontend import export_to_mlir
from pipeline import (
    torch_to_tosa, tosa_to_linalg,
    bufferize, linalg_to_scf,
    to_llvm_dialect, to_llvmir, compile_so,
    compile as lower,
)
from runner import MLIRRunner, PyTorchRunner, compare
from mlp import SampleMLP

OUTPUT_ROOT = Path(__file__).parent / "models_mlir"
NUM_TEST_INPUTS = 5

def make_random_inputs(example_inputs: tuple) -> list[torch.Tensor]:
    result = []
    for t in example_inputs:
        if t.is_floating_point():
            result.append(torch.randn_like(t))
        else:
            # Preserve vocab range
            high = int(t.max().item()) + 1
            result.append(torch.randint_like(t, low=0, high=max(high, 1)))
    return result


BATCH_SIZE = 8
mlp   = SampleMLP()
# conv  = SampleConv()
# rn18  = ResNet18Wrapper(pretrained=False)
# rn50  = ResNet50Wrapper(pretrained=False)
# gpt2 = GPT2Wrapper(pretrained=False)
# llama3 = Llama3Wrapper(pretrained=False)

models = {
    "mlp":              (mlp,              mlp.inputs(BATCH_SIZE)),
    # "conv":             (conv,             conv.inputs(BATCH_SIZE)),
    # "resnet18":         (rn18,             rn18.inputs(BATCH_SIZE)),
    # "resnet50":         (rn50,             rn50.inputs(BATCH_SIZE)),
    # "gpt2_no_kv":       (gpt2.no_kv,       gpt2.no_kv.inputs(BATCH_SIZE)),
    # "gpt2_prefill":     (gpt2.prefill,     gpt2.prefill.inputs(BATCH_SIZE)),
    # "gpt2_decode":      (gpt2.decode,      gpt2.decode.inputs(BATCH_SIZE)),
    # "llama3_no_kv":     (llama3.no_kv,     llama3.no_kv.inputs(BATCH_SIZE)),
    # "llama3_prefill":   (llama3.prefill,   llama3.prefill.inputs(BATCH_SIZE)),
    # "llama3_decode":    (llama3.decode,    llama3.decode.inputs(BATCH_SIZE)),
}

for name, (model, example_inputs) in models.items():
    print(f"\n=== {name} ===")
    out_dir = f"{OUTPUT_ROOT}/{name}"

    try:
        # Compile once using the example inputs
        torch_mlir = export_to_mlir(model, example_inputs)
        lower(torch_mlir, out_dir=out_dir)
        print(f"  compiled → {out_dir}/model.so")

        ref = PyTorchRunner(model)
        mlir_runner = MLIRRunner(out_dir)
        print(f"  input shapes: {mlir_runner.input_shapes}, output shapes: {mlir_runner.output_shapes}")

        all_pass = True
        for i in range(NUM_TEST_INPUTS):
            inputs = make_random_inputs(example_inputs)
            golden = ref.run(inputs)
            mlir_outputs = mlir_runner.run(inputs)
            match = compare(golden, mlir_outputs)
            print(f"  input[{i}]: {'PASS' if match else 'FAIL'}")
            all_pass = all_pass and match

        print(f"  overall: {'PASS' if all_pass else 'FAIL'}")

    except RuntimeError as e:
        print(f"  FAILED: {e}")