import torch
from vllm.model_executor.layers.mamba.gdn_linear_attn import (
    GatedDeltaNetAttention,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.distributed import tensor_model_parallel_all_gather

# Save original forwards before patching
_orig_qwen3next_attention_forward = Qwen3NextAttention.forward
_orig_gdn_forward_cuda = GatedDeltaNetAttention.forward_cuda


# ====================================================================
# Qwen3NextAttention.forward  (full-attention layers)
# Patch any 3D layout (decode or bucketed prefill with BS > 1):
#   hidden_states: [B, L, H],  output: [B, L, H_out]
# ====================================================================
def _hpu_qwen3next_attention_forward(self, positions, output, hidden_states):

    # Patch any 3D layout (BS > 1):
    #   Decode:  hidden_states [B, 1, H],  output [B, 1, H_out]
    #   Prefill: hidden_states [B, L, H],  output [B, L, H_out]
    #
    # Upstream forward assumes 2D (tokens, dim) for attn_output but
    # preserves 3D for gate when hidden_states is 3D, causing a shape
    # mismatch in `attn_output * gate`.  We flatten both to 2D.
    is_3d = (hidden_states is not None and output is not None and hidden_states.dim() == 3 and output.dim() == 3)
    if not is_3d:
        return _orig_qwen3next_attention_forward(self, positions, output, hidden_states)

    qkv, _ = self.qkv_proj(hidden_states)

    gate = None
    if self.attn_output_gate:
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        orig_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)

        q = q.reshape(*orig_shape, -1)
        gate = gate.reshape(*orig_shape, -1)
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
    k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

    q, k = self.rotary_emb(positions, q, k)

    # Normalize attention output to 2D token-major layout.
    attn_output = self.attn(q, k, v)
    attn_output_2d = attn_output.view(-1, attn_output.shape[-1])

    if self.attn_output_gate:
        assert gate is not None
        gate_2d = torch.sigmoid(gate).view(-1, gate.shape[-1])
        attn_output_2d = attn_output_2d * gate_2d

    proj_out, _ = self.o_proj(attn_output_2d)

    # Output buffer may be [B, 1, H_out] in decode.
    output_2d = output.view(-1, output.shape[-1])
    proj_out_2d = proj_out.view(-1, proj_out.shape[-1])
    output_2d[:proj_out_2d.shape[0]].copy_(proj_out_2d)


# ====================================================================
# 2. Qwen3NextSparseMoeBlock.forward  (MoE layers)
#    Upstream assumes 2-D input (num_tokens, hidden_dim).  On HPU the
#    hidden_states may arrive as 3-D [B, seq, H] during decode, so we
#    reshape to 2-D first and restore the original shape on output.
# ====================================================================
def _hpu_qwen3next_sparse_moe_forward(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    orig_shape = hidden_states.shape
    hidden_dim = orig_shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    num_tokens = hidden_states.shape[0]

    if self.is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)

    if self.experts.is_internal_router:
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
    else:
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)

    if self.shared_expert is not None:
        final_hidden_states = (final_hidden_states[0] + final_hidden_states[1])

    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]
    elif self.tp_size > 1:
        final_hidden_states = (self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states))

    return final_hidden_states.reshape(orig_shape)


# ====================================================================
# 3. GatedDeltaNetAttention.forward_cuda  (linear-attention / GDN layers)
#    Shared by Qwen3-Next and Qwen3.5. Upstream assumes 2-D
#    (num_tokens, H) inputs and allocates core_attn_out using
#    hidden_states.size(0); on HPU the bucketed input is 3-D
#    (B, L, H), which makes core_attn_out come out as (B, ...) while
#    z keeps the (B, L, ...) prefix, causing a broadcast mismatch in
#    self.norm(core_attn_out, z).
#
#    We flatten both hidden_states and output to 2-D before delegating
#    to upstream. output.view(...) shares storage so in-place writes
#    inside upstream propagate back to the caller's 3-D buffer.
# ====================================================================
def _hpu_gdn_forward_cuda(self, hidden_states, output):
    is_3d = (hidden_states is not None and output is not None and hidden_states.dim() == 3 and output.dim() == 3)
    if not is_3d:
        return _orig_gdn_forward_cuda(self, hidden_states, output)

    B, L, H = hidden_states.shape
    hs_2d = hidden_states.reshape(B * L, H)
    out_2d = output.view(B * L, output.shape[-1])
    _orig_gdn_forward_cuda(self, hs_2d, out_2d)


# ====================================================================
# Apply all patches
# ====================================================================
Qwen3NextAttention.forward = _hpu_qwen3next_attention_forward
Qwen3NextSparseMoeBlock.forward = _hpu_qwen3next_sparse_moe_forward
GatedDeltaNetAttention.forward_cuda = _hpu_gdn_forward_cuda
