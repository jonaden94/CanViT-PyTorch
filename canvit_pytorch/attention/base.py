from typing import override

import torch.nn.functional as F
from torch import Tensor, nn

from canvit_pytorch.rope import RoPE, rope_apply_with_prefix


def to_multihead(x: Tensor, num_heads: int) -> Tensor:
    """[B, N, D] -> [B, H, N, head_dim]."""
    B, N, D = x.shape
    return x.view(B, N, num_heads, D // num_heads).transpose(1, 2)


def from_multihead(x: Tensor) -> Tensor:
    """[B, H, N, head_dim] -> [B, N, D]."""
    B, H, N, hd = x.shape
    return x.transpose(1, 2).reshape(B, N, H * hd)


class CanvasAttention(nn.Module):
    """Base class for asymmetric canvas cross-attention.

    Subclasses (CanvasReadAttention, CanvasWriteAttention) configure which
    transforms are dense (Linear) vs Identity.

    All attention happens in canvas_dim space. The output is projected to out_dim.
    """

    def __init__(
        self,
        *,
        q_in_dim: int,
        kv_in_dim: int,
        canvas_dim: int,
        out_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        assert canvas_dim % num_heads == 0
        self.canvas_dim: int = canvas_dim
        self.out_dim: int = out_dim
        self.num_heads: int = num_heads

        # Overridden by subclasses
        self.q_proj: nn.Module = nn.Identity()
        self.k_proj: nn.Module = nn.Identity()
        self.v_proj: nn.Module = nn.Identity()
        self.out_proj: nn.Module = nn.Identity()

        self.q_norm = nn.LayerNorm(q_in_dim)
        self.kv_norm = nn.LayerNorm(kv_in_dim)

    # Which side carries the (local/glimpse) tokens that may be modulated:
    # "q" for reads (query=local; γ,β on q + an α output gate), "kv" for writes
    # (kv=local; γ,β on kv, no gate). None = not modulatable. Subclasses set it.
    _mod_side: str | None = None

    def _apply_local_mod(
        self, qn: Tensor, kvn: Tensor, mod: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Apply local-side γ=1+raw, β=raw to the normed q (read) or kv (write).

        Returns the (possibly) modulated ``(qn, kvn)`` and, for reads, the raw
        α output-gate term (else ``None``). No-op when ``mod is None``.
        """
        if mod is None or self._mod_side is None:
            return qn, kvn, None
        if self._mod_side == "q":
            g, b, a = mod.chunk(3, dim=-1)  # [n_local, D] each
            return (1.0 + g) * qn + b, kvn, a
        g, b = mod.chunk(2, dim=-1)
        return qn, (1.0 + g) * kvn + b, None

    @override
    def forward(
        self,
        *,
        query: Tensor,
        kv: Tensor,
        query_rope: RoPE,
        kv_rope: RoPE,
        mod: Tensor | None = None,
    ) -> Tensor:
        """Cross-attention from query to kv.

        Args:
            query: [B, N_q, q_in_dim]
            kv: [B, N_kv, kv_in_dim] - source for keys and values
            query_rope: RoPE for query positions
            kv_rope: RoPE for key positions
            mod: optional per-local-token modulation (None = unmodulated).

        Returns:
            [B, N_q, out_dim]
        """
        qn, kvn, gate_a = self._apply_local_mod(self.q_norm(query), self.kv_norm(kv), mod)
        q: Tensor = to_multihead(self.q_proj(qn), self.num_heads)
        k: Tensor = to_multihead(self.k_proj(kvn), self.num_heads)
        v: Tensor = to_multihead(self.v_proj(kvn), self.num_heads)

        q = rope_apply_with_prefix(x=q, rope=query_rope)
        k = rope_apply_with_prefix(x=k, rope=kv_rope)

        # Cast K/V to Q's dtype for AMP compatibility (e.g. TPU autocast
        # promotes Q to bf16 via backbone but K/V stay f32 from canvas projections).
        out: Tensor = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype))
        out = self.out_proj(from_multihead(out))
        if gate_a is not None:
            out = (1.0 + gate_a) * out  # read residual gate (α0 = 1)
        return out
