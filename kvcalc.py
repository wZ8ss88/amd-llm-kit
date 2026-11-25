#!/usr/bin/env python3
"""KV-cache size calculator for transformer inference on AMD GPUs.

Answers one question before you launch a server: how much VRAM will the
KV cache eat for this model, this context length and this batch?

The formula is the one every serving engine uses:

    bytes = 2 * layers * kv_heads * head_dim * bytes_per_element
            * seq_len * batch

`2` is for K and V, `kv_heads` is the *key/value* head count, not the
attention head count, so grouped-query attention (GQA) and multi-query
attention (MQA) are handled by construction: a model with 32 attention
heads and 8 KV heads only pays for 8.

Weights are NOT included -- this is the cache only. Add them separately
(see --total).

Stdlib only. No ROCm needed to run it.

Examples
--------
    # 70B-class GQA model, 8k context, batch 4, on a 192 GiB card
    python kvcalc.py --layers 80 --kv-heads 8 --head-dim 128 \
        --seq 8192 --batch 4 --vram 192

    # how many concurrent 32k sequences fit into 120 GiB of free VRAM?
    python kvcalc.py --layers 80 --kv-heads 8 --head-dim 128 \
        --seq 32768 --fit 120
"""

from __future__ import annotations

import argparse
import sys

# bytes per element for the dtypes that actually show up in KV caches
DTYPES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
}

GIB = 1024 ** 3


def kv_cache_bytes(
    layers: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    dtype: str = "fp16",
) -> int:
    """Return the KV-cache size in bytes for one forward pass of `batch`
    sequences of `seq_len` tokens each."""
    if layers <= 0:
        raise ValueError("layers must be > 0")
    if kv_heads <= 0:
        raise ValueError("kv_heads must be > 0 (use 1 for MQA)")
    if head_dim <= 0:
        raise ValueError("head_dim must be > 0")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if batch <= 0:
        raise ValueError("batch must be > 0")
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; pick one of {sorted(DTYPES)}")
    return 2 * layers * kv_heads * head_dim * DTYPES[dtype] * seq_len * batch


def human(n: float) -> str:
    """Format a byte count with binary units."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.2f} {unit}" if unit != "B" else f"{int(n):,} B"
        n /= 1024
    return f"{n:,.2f} TiB"  # pragma: no cover


def max_sequences(
    layers: int, kv_heads: int, head_dim: int, seq_len: int,
    dtype: str, budget_bytes: int,
) -> int:
    """How many concurrent sequences of `seq_len` fit into `budget_bytes`."""
    per_seq = kv_cache_bytes(layers, kv_heads, head_dim, seq_len, 1, dtype)
    return budget_bytes // per_seq


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Estimate KV-cache VRAM for a transformer on AMD hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--layers", type=int, required=True,
                   help="decoder layer count (num_hidden_layers)")
    p.add_argument("--kv-heads", type=int, required=True,
                   help="key/value head count (num_key_value_heads; use 1 for MQA)")
    p.add_argument("--head-dim", type=int, default=128,
                   help="dimension per head (hidden_size / num_attention_heads)")
    p.add_argument("--heads", type=int, default=None,
                   help="optional attention head count, shown as the GQA ratio")
    p.add_argument("--seq", type=int, default=8192,
                   help="context length in tokens (max_model_len)")
    p.add_argument("--batch", type=int, default=1,
                   help="concurrent sequences")
    p.add_argument("--dtype", default="fp16", choices=sorted(DTYPES),
                   help="KV-cache dtype")
    p.add_argument("--vram", type=float, default=None,
                   help="VRAM budget in GiB; prints how much of it the cache takes")
    p.add_argument("--weights", type=float, default=None,
                   help="model weights in GiB, added to the total for a sanity check")
    p.add_argument("--fit", type=float, default=None, metavar="GIB",
                   help="instead of --vram: how many sequences fit in this many GiB")
    args = p.parse_args(argv)

    try:
        total = kv_cache_bytes(args.layers, args.kv_heads, args.head_dim,
                               args.seq, args.batch, args.dtype)
    except ValueError as exc:
        p.error(str(exc))

    per_seq = kv_cache_bytes(args.layers, args.kv_heads, args.head_dim,
                             args.seq, 1, args.dtype)

    print(f"layers          {args.layers}")
    print(f"kv heads        {args.kv_heads}"
          + (f"  (GQA ratio {args.heads / args.kv_heads:.1f}:1 vs {args.heads} attention heads)"
             if args.heads else ""))
    print(f"head dim        {args.head_dim}")
    print(f"context         {args.seq:,} tokens")
    print(f"batch           {args.batch}")
    print(f"dtype           {args.dtype} ({DTYPES[args.dtype]} B/element)")
    print("-" * 46)
    print(f"per sequence    {human(per_seq)}")
    print(f"total cache     {human(total)}")

    if args.weights is not None:
        served = total + args.weights * GIB
        print(f"weights         {args.weights:.2f} GiB")
        print(f"weights + cache {human(served)}")

    if args.vram is not None:
        budget = args.vram * GIB
        pct = 100.0 * total / budget
        print(f"share of {args.vram:g} GiB   {pct:.1f}%")
        if pct > 100:
            print("VERDICT: does not fit -- lower the context, the batch or the dtype")

    if args.fit is not None:
        n = max_sequences(args.layers, args.kv_heads, args.head_dim,
                          args.seq, args.dtype, int(args.fit * GIB))
        print(f"fits in {args.fit:g} GiB {n} sequence(s) of {args.seq:,} tokens")

    return 0


if __name__ == "__main__":
    sys.exit(main())
