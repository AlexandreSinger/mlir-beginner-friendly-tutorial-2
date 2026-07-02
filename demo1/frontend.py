import torch
from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_decomp_util import get_decomposition_table
from decompositions import CUSTOM_DECOMPOSITIONS


def _inline_buffers(exported: torch.export.ExportedProgram) -> None:
    """Replace buffer placeholder nodes with get_attr constant nodes.

    torch.export keeps registered buffers (e.g. BN running_mean/running_var/
    num_batches_tracked) as function inputs rather than inlining them, because
    they are mutable state. FxImporter preserves this, so they appear as extra
    memref arguments in the compiled function.

    This function replaces each buffer placeholder with a get_attr node pointing
    to the buffer's actual tensor value. FxImporter will then inline those as
    torch.vtensor.literal constants, removing them from the function signature.

    Mutates exported.graph in-place.
    """
    sig = exported.graph_signature
    if not sig.inputs_to_buffers:
        return

    graph = exported.graph
    gm = exported.graph_module

    buf_nodes = [
        node for node in graph.nodes
        if node.op == "placeholder" and node.name in sig.inputs_to_buffers
    ]

    for node in buf_nodes:
        buf_name = sig.inputs_to_buffers[node.name]
        # Non-persistent buffers (e.g. LLaMA's rotary_emb.inv_freq) are absent
        # from state_dict and land in exported.constants instead.
        if buf_name in exported.state_dict:
            tensor = exported.state_dict[buf_name].detach()
        else:
            tensor = exported.constants[buf_name].detach()

        attr_name = f"_frozen_{node.name}"
        gm.register_buffer(attr_name, tensor)

        with graph.inserting_after(node):
            const_node = graph.get_attr(attr_name)
            const_node.meta = dict(node.meta)

        node.replace_all_uses_with(const_node)
        graph.erase_node(node)

    graph.lint()
    gm.recompile()


def export_to_mlir(
    model: torch.nn.Module,
    example_inputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
    output_path: str | None = None,
) -> str:
    model.eval()

    decomp_table = {**get_decomposition_table(), **CUSTOM_DECOMPOSITIONS}

    exported = torch.export.export(model, example_inputs)
    exported = exported.run_decompositions(decomp_table)

    _inline_buffers(exported)

    context = ir.Context()
    torch_d.register_dialect(context)
    importer = FxImporter(context=context)
    importer.import_frozen_program(exported)

    mlir_text = str(importer.module)

    if output_path:
        with open(output_path, "w") as f:
            f.write(mlir_text)

    return mlir_text