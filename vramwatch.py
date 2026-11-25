#!/usr/bin/env python3
"""Watch VRAM on AMD GPUs while you load or serve a model.

Polling is deliberately boring: run the loop in one terminal, start the
server in another, and you can see the cache climb before it OOMs. That is
much cheaper than reading a stack trace ten seconds too late.

Reads `rocm-smi --showmeminfo vram --json`. Falls back to the plain-text
form when the installed rocm-smi predates --json.

    python vramwatch.py                      # every 2 s, until Ctrl-C
    python vramwatch.py --interval 0.5 --alert-pct 95
    python vramwatch.py --once                # single reading, for scripts

Exit codes: 0 = fine, 2 = a card crossed --alert-pct, 3 = no rocm-smi.
Stdlib only; needs a Linux host with ROCm installed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

GIB = 1024 ** 3

# rocm-smi key spellings seen across versions
TOTAL_KEYS = ("VRAM Total Memory (B)", "vram_total", "VRAM Total Memory")
USED_KEYS = ("VRAM Total Used Memory (B)", "vram_used", "VRAM Total Used Memory")


def _pick(d: dict, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return None


def read_vram() -> list[tuple[str, float, float]]:
    """Return [(card, used_bytes, total_bytes)] for every visible card."""
    exe = shutil.which("rocm-smi")
    if exe is None:
        raise FileNotFoundError("rocm-smi not found on PATH")

    proc = subprocess.run([exe, "--showmeminfo", "vram", "--json"],
                          capture_output=True, text=True, timeout=20)
    cards: list[tuple[str, float, float]] = []

    if proc.returncode == 0 and proc.stdout.strip().startswith("{"):
        data = json.loads(proc.stdout)
        for card, info in data.items():
            if not isinstance(info, dict):
                continue
            used, total = _pick(info, USED_KEYS), _pick(info, TOTAL_KEYS)
            if used is not None and total:
                cards.append((card, used, total))
        if cards:
            return cards

    # older rocm-smi: text output
    proc = subprocess.run([exe, "--showmeminfo", "vram"],
                          capture_output=True, text=True, timeout=20)
    total = used = None
    for line in proc.stdout.splitlines():
        m = re.search(r"GPU\[(\d+)\].*(Total|Used).*?(\d+)", line)
        if not m:
            continue
        card = f"card{m.group(1)}"
        if m.group(2) == "Total":
            total = float(m.group(3))
        else:
            used = float(m.group(3))
        if total is not None and used is not None:
            cards.append((card, used, total))
            total = used = None
    if not cards:
        raise RuntimeError("could not parse rocm-smi output")
    return cards


def render(cards: list[tuple[str, float, float]], alert_pct: float) -> int:
    """Print one frame; return the number of cards over the threshold."""
    over = 0
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"--- {stamp} ---")
    for card, used, total in cards:
        pct = 100.0 * used / total if total else 0.0
        bar = "#" * int(pct / 4)
        flag = ""
        if pct >= alert_pct:
            flag = "  <== over"
            over += 1
        print(f"{card:>7}  {used / GIB:7.1f} / {total / GIB:6.1f} GiB "
              f"{pct:5.1f}% |{bar:<25}|{flag}")
    print()
    return over


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Watch AMD GPU VRAM usage.")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between reads")
    p.add_argument("--alert-pct", type=float, default=95.0,
                   help="mark cards at or above this VRAM percentage")
    p.add_argument("--once", action="store_true", help="one frame, then exit")
    args = p.parse_args(argv)

    try:
        while True:
            over = render(read_vram(), args.alert_pct)
            if args.once:
                return 2 if over else 0
            time.sleep(max(args.interval, 0.2))
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
