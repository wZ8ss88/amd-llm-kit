#!/usr/bin/env python3
"""Map an AMD GPU target (gfx arch) to what it actually is.

Handy when a log line, a container tag or a support thread says `gfx1030`
and you need to know whether the box can run a given precision, what its
wavefront size is, and whether mainstream serving stacks target it.

Data is descriptive, not aspirational: CDNA parts are the ones with
matrix cores (MFMA) and the ones vLLM/ROCm ship kernels for; RDNA parts
are consumer/graphics oriented and mostly lack MFMA.

Stdlib only.

    python gfxmap.py gfx942
    python gfxmap.py gfx1100 gfx1201
    python gfxmap.py --all
"""

from __future__ import annotations

import argparse
import sys

# target -> (family, product examples, wavefront, matrix cores, precisions, note)
ARCH = {
    "gfx900": ("Vega 10", "Vega 56/64", 64, "no MFMA",
               "fp32, fp16", "legacy, ROCm support frozen"),
    "gfx906": ("Vega 20", "Radeon VII, MI50, MI60", 64, "MFMA (fp32/fp16)",
               "fp32, fp16", "legacy HPC, dropped from recent ROCm"),
    "gfx908": ("CDNA 1", "MI100", 64, "MFMA",
               "fp32, fp16, bf16", "first CDNA; no fp8"),
    "gfx90a": ("CDNA 2", "MI210, MI250, MI250X", 64, "MFMA",
               "fp32, fp16, bf16", "bf16 support; still widely deployed"),
    "gfx940": ("CDNA 3", "MI300A", 64, "MFMA",
               "fp32, fp16, bf16, fp8, fp6", "APU variant (shared HBM)"),
    "gfx941": ("CDNA 3", "MI300 series", 64, "MFMA",
               "fp32, fp16, bf16, fp8, fp6", "8-stack variant"),
    "gfx942": ("CDNA 3", "MI300X", 64, "MFMA",
               "fp32, fp16, bf16, fp8, fp6", "the MI300X target; fp8 KV cache works"),
    "gfx950": ("CDNA 4", "MI325X, MI355X", 64, "MFMA",
               "fp32, fp16, bf16, fp8, fp6, fp4", "fp4 tensor ops; 256 GiB class HBM"),
    "gfx1010": ("RDNA 1", "RX 5700", 32, "no MFMA",
               "fp32, fp16", "graphics first"),
    "gfx1030": ("RDNA 2", "RX 6800/6900", 32, "no MFMA",
               "fp32, fp16", "consumer; ROCm works, no matrix cores"),
    "gfx1031": ("RDNA 2", "RX 6700", 32, "no MFMA", "fp32, fp16", "consumer"),
    "gfx1032": ("RDNA 2", "RX 6600", 32, "no MFMA", "fp32, fp16", "consumer"),
    "gfx1100": ("RDNA 3", "RX 7900 XTX", 32, "WMMA",
               "fp32, fp16, bf16", "wave32; consumer"),
    "gfx1101": ("RDNA 3", "RX 7800/7700", 32, "WMMA", "fp32, fp16, bf16", "consumer"),
    "gfx1102": ("RDNA 3", "RX 7600", 32, "WMMA", "fp32, fp16", "consumer"),
    "gfx1103": ("RDNA 3", "integrated RDNA3", 32, "WMMA", "fp32, fp16", "iGPU"),
    "gfx1150": ("RDNA 3.5", "Strix Point", 32, "WMMA",
                "fp32, fp16, bf16", "mobile APU, unified memory"),
    "gfx1151": ("RDNA 3.5", "Strix Halo", 32, "WMMA",
                "fp32, fp16, bf16", "large unified memory APU"),
    "gfx1200": ("RDNA 4", "RX 9070", 32, "WMMA",
                "fp32, fp16, bf16, fp8", "newest consumer; needs recent ROCm"),
    "gfx1201": ("RDNA 4", "RX 9070 XT", 32, "WMMA",
                "fp32, fp16, bf16, fp8", "newest consumer; needs recent ROCm"),
}


def lookup(target: str) -> tuple[str, ...] | None:
    """Return the tuple for `target`, or None when unknown."""
    return ARCH.get(target.strip().lower())


def describe(target: str) -> str:
    row = lookup(target)
    if row is None:
        return (f"{target:<9} unknown target\n"
                f"{'':<9} not in the table -- check `rocminfo` for the real "
                f"gfx string, or run --all")
    family, products, wave, mfma, prec, note = row
    return "\n".join((
        f"{target:<9} {family}  ({products})",
        f"{'':<9} wavefront {wave}, matrix cores: {mfma}",
        f"{'':<9} supported precisions: {prec}",
        f"{'':<9} {note}",
    ))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Explain an AMD gfx target.",
        epilog="Targets come from rocminfo / rocm-smi; they are not guessable.",
    )
    p.add_argument("targets", nargs="*", help="gfx target(s), e.g. gfx942")
    p.add_argument("--all", action="store_true", help="print the whole table")
    args = p.parse_args(argv)

    if args.all:
        for t in sorted(ARCH):
            print(describe(t))
            print()
        return 0

    if not args.targets:
        p.error("give at least one target or --all")

    for i, t in enumerate(args.targets):
        if i:
            print()
        print(describe(t))
    return 0 if all(lookup(t) for t in args.targets) else 1


if __name__ == "__main__":
    sys.exit(main())
