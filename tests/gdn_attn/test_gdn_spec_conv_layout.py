"""Spec-decode conv-state layout of torch.ops._xpu_C.gdn_attention (nsr2-55).

vLLM 0.26's Mamba align-mode copies treat a GDN conv state as one sliding
window in block-table column 0, read at row num_accepted - 1. The kernel must
read and write that layout. Before the vllm-xpu-kernels#544 backport it wrote a
3-row checkpoint into every column instead.

Run inside the vllm-b65 image on an idle XPU (the image has no pytest):
    python /work/tests/test_gdn_spec_conv_layout.py
_extract_qkv_b_a_z is copied from Intel's tests/gdn_attn/test_gdn_attn.py.
"""
import math
import random
import sys

import itertools

import torch
import torch.nn.functional as F

import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C)


def _extract_qkv_b_a_z(projected_states_qkvz, projected_states_ba,
                       num_actual_tokens, num_k_heads, num_v_heads,
                       head_k_dim, head_v_dim, tp_size, reorder_input):
    """Replicate the qkv/ba split done in `ref_gdn_attention` and return
    the post-reorder qkv (for conv1d), b, a, z tensors plus split sizes."""
    if reorder_input:
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        q_size = key_dim // tp_size
        k_size = q_size
        v_size = value_dim // tp_size
        z_size = v_size
        q_tmp, k_tmp, v_tmp, z_tmp = projected_states_qkvz.split(
            [q_size, k_size, v_size, z_size], dim=-1)
        q_tmp = q_tmp.reshape(q_tmp.size(0), -1, head_k_dim)
        k_tmp = k_tmp.reshape(k_tmp.size(0), -1, head_k_dim)
        v_tmp = v_tmp.reshape(v_tmp.size(0), -1,
                              num_v_heads // num_k_heads * head_v_dim)
        z_tmp = z_tmp.reshape(z_tmp.size(0), -1,
                              num_v_heads // num_k_heads * head_v_dim)
        projected_states_qkvz = torch.cat(
            [q_tmp, k_tmp, v_tmp, z_tmp],
            dim=-1).reshape(q_tmp.size(0), -1).contiguous()

        b, a = projected_states_ba.chunk(2, dim=-1)
        b = b.reshape(b.size(0), -1, num_v_heads // num_k_heads)
        a = a.reshape(a.size(0), -1, num_v_heads // num_k_heads)
        projected_states_ba = torch.cat(
            [b, a], dim=-1).reshape(b.size(0), -1).contiguous()

    projected_states_ba = projected_states_ba.reshape(
        num_actual_tokens, num_k_heads // tp_size,
        (2 * num_v_heads // num_k_heads))
    b, a = torch.split(
        projected_states_ba,
        [num_v_heads // num_k_heads, num_v_heads // num_k_heads],
        dim=-1)
    b = b.reshape(num_actual_tokens, num_v_heads // tp_size)
    a = a.reshape(num_actual_tokens, num_v_heads // tp_size)

    split_qkvz = [
        head_k_dim,
        head_k_dim,
        num_v_heads // num_k_heads * head_v_dim,
        num_v_heads // num_k_heads * head_v_dim,
    ]
    projected_states_qkvz = projected_states_qkvz.reshape(
        num_actual_tokens, num_k_heads // tp_size,
        (2 * head_k_dim + 2 * num_v_heads // num_k_heads * head_v_dim))
    q_split, k_split, v_split, z_split = torch.split(
        projected_states_qkvz, split_qkvz, dim=-1)
    q_split = q_split.reshape(num_actual_tokens,
                              num_k_heads // tp_size * head_k_dim)
    k_split = k_split.reshape(num_actual_tokens,
                              num_k_heads // tp_size * head_k_dim)
    v_split = v_split.reshape(
        num_actual_tokens,
        num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim)
    qkv = torch.cat((q_split, k_split, v_split), dim=-1).reshape(
        num_actual_tokens,
        num_k_heads // tp_size *
        (2 * head_k_dim + num_v_heads // num_k_heads * head_v_dim))
    z_global = z_split.reshape(num_actual_tokens, num_v_heads // tp_size,
                               head_v_dim)
    return qkv, b, a, z_global





def ref_spec(qkvz, ba, conv_state, ssm_state, conv_w, conv_b, A_log, dt_bias,
             query_start_loc, token_indx, state_idx, num_accepted, H, HV, D,
             reorder_input):
    """Reference for one spec-decode call with the sliding-window conv line."""
    eps, scale = 1e-6, 1.0 / math.sqrt(D)
    width = conv_w.shape[-1]
    K = state_idx.shape[1]
    state_len = conv_state.shape[1]
    hist_rows = state_len - K
    n_tok = qkvz.shape[0]
    qkv_elems = H * (2 * D + HV // H * D)
    qkv, b, a, z_glob = _extract_qkv_b_a_z(qkvz, ba, n_tok, H, HV, D, D, 1,
                                           reorder_input)
    out = torch.zeros(n_tok, HV, D, dtype=qkvz.dtype, device=qkvz.device)
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)
    split = [H * D, H * D, HV * D]
    rep = HV // H
    for n in range(state_idx.shape[0]):
        s, e = int(query_start_loc[n]), int(query_start_loc[n + 1])
        glob = token_indx[s:e].to(torch.long)
        seq = e - s
        init_row = max(int(num_accepted[n]) - 1, 0)
        ssm_slot = int(state_idx[n, init_row])
        line_slot = int(state_idx[n, 0])
        line = conv_state[line_slot]
        prior = line[init_row:init_row + width - 1].clone()
        x = qkv[glob]
        conv_in = torch.cat([prior, x], 0)
        new_line = line.clone()
        new_line[:hist_rows] = line[init_row + 1:init_row + 1 + hist_rows]
        new_line[hist_rows:hist_rows + seq] = x
        conv_state[line_slot] = new_line
        y = F.conv1d(conv_in.T.unsqueeze(0).float(), conv_w.unsqueeze(1).float(),
                     None if conv_b is None else conv_b.float(),
                     groups=qkv_elems)
        y = F.silu(y).to(qkvz.dtype).transpose(-2, -1).reshape(seq, qkv_elems)
        q, k, v = torch.split(y, split, -1)
        q = q.reshape(seq, H, D).float()
        k = k.reshape(seq, H, D).float()
        v = v.reshape(seq, HV, D).float()
        q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + eps) * scale
        k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + eps)
        q, k = q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)
        h = ssm_state[ssm_slot].float().clone()
        beta = torch.sigmoid(b[glob].float())
        g = torch.exp(-torch.exp(A_log) * softplus(a[glob].float() + dt_bias))
        for t in range(seq):
            h *= g[t][:, None, None]
            delta = (v[t] - torch.einsum("vhk,vk->vh", h, k[t])) * beta[t][:, None]
            h += torch.einsum("vh,vk->vhk", delta, k[t])
            out[glob[t]] = torch.einsum("vhk,vk->vh", h, q[t]).to(out.dtype)
            ssm_state[int(state_idx[n, t])] = h.to(ssm_state.dtype)
    return out


@torch.inference_mode()
def test_spec_conv_line_layout(dtype, reorder_input, ragged):
    # Qwen3.8-27B GDN shapes on one B65: 16 k-heads, 48 v-heads, head dim 128.
    H, HV, D, width, num_spec = 16, 48, 128, 4, 3
    K = num_spec + 1
    n_seq, slots = 5, 64
    device = "xpu"
    random.seed(0)
    torch.manual_seed(0)
    lens = [K] * n_seq
    if ragged:
        lens[-1] -= 1
    n_tok = sum(lens)
    qkvz = torch.randn(n_tok, H * (2 * D + 2 * D * HV // H), dtype=dtype, device=device)
    ba = torch.randn(n_tok, 2 * HV, dtype=dtype, device=device)
    conv_dim = H * (2 * D + D * HV // H)
    conv_state = torch.randn(slots, width - 1 + num_spec, conv_dim, dtype=dtype, device=device)
    ssm_state = torch.randn(slots, HV, D, D, dtype=dtype, device=device) * 0.1
    conv_w = torch.randn(conv_dim, width, dtype=dtype, device=device)
    conv_b = torch.randn(conv_dim, dtype=dtype, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=dtype, device=device)
    state_idx = torch.tensor(random.sample(range(1, slots), n_seq * K),
                             dtype=torch.int32, device=device).reshape(n_seq, K)
    # Cover acceptance 0..K, including num_accepted > 1 (the failing case).
    num_accepted = torch.tensor([n % (K + 1) for n in range(n_seq)],
                                dtype=torch.int32, device=device)
    token_indx = torch.randperm(n_tok, device=device).to(torch.int32)
    qsl = torch.tensor([0] + lens, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)

    ref_conv, ref_ssm = conv_state.clone(), ssm_state.clone()
    before_conv = conv_state.clone()
    out = torch.zeros(n_tok, HV, D, dtype=dtype, device=device)
    z = torch.zeros_like(out)
    torch.ops._xpu_C.gdn_attention(
        out, z, qkvz, ba, H, HV, D, D,
        conv_state=conv_state, ssm_state=ssm_state, conv_weights=conv_w,
        conv_bias=conv_b, activation="silu", A_log=A_log, dt_bias=dt_bias,
        num_prefills=0, num_decodes=0, num_spec_decodes=n_seq,
        has_initial_state=None, non_spec_query_start_loc=None,
        non_spec_token_indx=None, non_spec_state_indices_tensor=None,
        spec_query_start_loc=qsl, spec_token_indx=token_indx,
        spec_state_indices_tensor=state_idx, num_accepted_tokens=num_accepted,
        num_actual_tokens=n_tok, tp_size=1, reorder_input=reorder_input)
    ref_out = ref_spec(qkvz, ba, ref_conv, ref_ssm, conv_w, conv_b, A_log, dt_bias,
                       qsl, token_indx, state_idx, num_accepted, H, HV, D,
                       reorder_input)

    torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)
    for n in range(n_seq):
        line = int(state_idx[n, 0])
        rows = (width - 1 - 1) + lens[n]  # history rows + this sequence's inputs
        torch.testing.assert_close(conv_state[line, :rows], ref_conv[line, :rows],
                                   atol=5e-2, rtol=5e-2)
        # Columns 1..K-1 are SSM rollback slots only; their conv state is untouched.
        for t in range(1, K):
            slot = int(state_idx[n, t])
            assert torch.equal(conv_state[slot], before_conv[slot]), (n, t)


if __name__ == "__main__":
    failed = 0
    for dtype, reorder_input, ragged in itertools.product(
            [torch.bfloat16, torch.float16], [True, False], [False, True]):
        name = f"dtype={dtype} reorder_input={reorder_input} ragged={ragged}"
        try:
            test_spec_conv_line_layout(dtype, reorder_input, ragged)
            print("PASS", name)
        except AssertionError as e:
            failed += 1
            print("FAIL", name, str(e).splitlines()[0][:200])
    print(f"{8 - failed}/8 passed")
    sys.exit(1 if failed else 0)
