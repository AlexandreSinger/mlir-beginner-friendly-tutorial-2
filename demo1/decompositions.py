import torch

aten = torch.ops.aten


def diff_decomp(
    input: torch.Tensor,
    n: int = 1,
    dim: int = -1,
    prepend: torch.Tensor | None = None,
    append: torch.Tensor | None = None,
) -> torch.Tensor:
    combined = input
    if prepend is not None:
        combined = torch.cat([prepend, combined], dim=dim)
    if append is not None:
        combined = torch.cat([combined, append], dim=dim)
    for _ in range(n):
        size = combined.shape[dim]
        a = torch.narrow(combined, dim, 1, size - 1)
        b = torch.narrow(combined, dim, 0, size - 1)
        combined = a - b
    return combined


def and_tensor_decomp(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    # __and__.Tensor has no lowering in TOSA/stablehlo; remap to bitwise_and
    return aten.bitwise_and.Tensor(self, other)


def scaled_dot_product_attention_decomp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    import math
    head_dim = query.size(-1)
    s = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    # QK^T scaled
    attn = torch.matmul(query, key.transpose(-2, -1)) * s

    if is_causal:
        L, S = query.size(-2), key.size(-2)
        mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
        attn = attn.masked_fill(~mask, float("-inf"))
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask

    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, value)


# Neither SampleMLP nor FullyConnected need any of these — they only use
# Linear and ReLU, which torch-mlir/TOSA already lowers natively. These three
# decompositions are kept as worked examples for extending the demo to
# sequence/attention-based models: if you add a model and torch.export's
# run_decompositions() fails because TOSA has no lowering for some aten op,
# this is where you'd add a rewrite (in terms of ops TOSA does support) to
# fix it. These particular decompositions come from
# [Robert's Waferscape Project](https://github.com/robluo/WaferScapeMapper),
# which this demo's pipeline code was originally borrowed from.
CUSTOM_DECOMPOSITIONS = {
    aten.diff.default:                          diff_decomp,
    aten.__and__.Tensor:                        and_tensor_decomp,
    aten.scaled_dot_product_attention.default:  scaled_dot_product_attention_decomp,
}