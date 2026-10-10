"""Frozen timing contract: mixed per-sequence causal call versus split baseline."""
import json
import statistics
import time

import torch

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func


def sync_time(fn):
    torch.xpu.synchronize()
    start = time.perf_counter_ns()
    output = fn()
    torch.xpu.synchronize()
    return (time.perf_counter_ns() - start) / 1e6, output


def main():
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")
    torch.manual_seed(20260807)
    q_lens, kv_lens, causal = [3, 4, 2], [9, 10, 8], [True, False, True]
    q = torch.randn(sum(q_lens), 8, 64, dtype=torch.bfloat16)
    k = torch.randn(32, 16, 2, 64, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu = torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(0,
                                                               dtype=torch.int32)
    used = torch.tensor(kv_lens, dtype=torch.int32)
    tables = torch.randint(0, 32, (3, 1), dtype=torch.int32)
    mask = torch.tensor(causal, dtype=torch.bool)

    def mixed():
        return flash_attn_varlen_func(q, k, v, 4, cu, 10,
                                      seqused_k=used, softmax_scale=64**-0.5,
                                      causal=True, block_table=tables,
                                      window_size=(4, 2),
                                      per_seq_causal=mask)

    def split():
        outputs, offset = [], 0
        for row, (ql, kl, is_causal) in enumerate(zip(q_lens, kv_lens, causal)):
            outputs.append(flash_attn_varlen_func(
                q[offset:offset + ql], k, v, ql,
                torch.tensor([0, ql], dtype=torch.int32), kl,
                seqused_k=torch.tensor([kl], dtype=torch.int32),
                softmax_scale=64**-0.5, causal=is_causal,
                block_table=tables[row:row + 1],
                window_size=(4, 2 if is_causal else 4)))
            offset += ql
        return torch.cat(outputs)

    _, candidate_out = sync_time(mixed)
    _, split_out = sync_time(split)
    torch.testing.assert_close(candidate_out, split_out, atol=1.5e-2,
                               rtol=1.5e-2)
    for _ in range(10):
        sync_time(split)
        sync_time(mixed)
    split_ms = [sync_time(split)[0] for _ in range(30)]
    mixed_ms = [sync_time(mixed)[0] for _ in range(30)]
    print(json.dumps({"baseline_split_ms": split_ms, "candidate_mixed_ms": mixed_ms,
                      "baseline_median_ms": statistics.median(split_ms),
                      "candidate_median_ms": statistics.median(mixed_ms),
                      "ratio": statistics.median(mixed_ms) / statistics.median(split_ms)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
