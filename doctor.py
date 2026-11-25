#!/usr/bin/env python3
"""Probe a host for the things LLM serving on AMD actually needs.

Every check is read-only and timeboxed. Nothing is installed, nothing is
written outside stdout. The point is to get the boring environment facts
into one place before blaming the model.

Runs on Linux with ROCm. On other platforms it still runs -- the ROCm
checks simply report MISSING -- which makes it usable for "is this box
even close?" questions before provisioning.

    python doctor.py
    python doctor.py --json        # machine-readable, same data
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"

# (id, title, severity, detail) -- severity here is "how bad is the finding"
Findings = list[tuple[str, str, str, str]]


def run(cmd: list[str], timeout: float = 8.0) -> tuple[int, str]:
    """Run a command, return (returncode, combined output). Never raises."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 127, ""


def first_line(text: str, limit: int = 90) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:limit]


def check_os() -> Findings:
    out: Findings = []
    system = platform.system()
    if system == "Linux":
        out.append(("os", "operating system", OK, platform.platform()[:90]))
    else:
        out.append(("os", "operating system", WARN,
                    f"{system} -- ROCm checks below will mostly be MISSING"))
    return out


def check_rocm_install() -> Findings:
    out: Findings = []
    ver_file = "/opt/rocm/.info/version"
    if os.path.exists(ver_file):
        try:
            with open(ver_file, encoding="utf-8") as fh:
                out.append(("rocm", "ROCm install", OK, f"version {fh.read().strip()}"))
        except OSError:
            out.append(("rocm", "ROCm install", WARN, f"{ver_file} unreadable"))
    elif os.path.isdir("/opt/rocm"):
        out.append(("rocm", "ROCm install", WARN, "/opt/rocm exists but has no .info/version"))
    else:
        out.append(("rocm", "ROCm install", FAIL, "/opt/rocm not found"))

    for exe in ("rocm-smi", "hipcc", "rocminfo"):
        path = shutil.which(exe)
        if path:
            out.append((f"bin-{exe}", f"{exe} on PATH", OK, path))
        else:
            sev = FAIL if exe == "rocm-smi" else WARN
            out.append((f"bin-{exe}", f"{exe} on PATH", sev, "not found"))
    return out


def check_devices() -> Findings:
    out: Findings = []
    if os.path.exists("/dev/kfd"):
        out.append(("kfd", "compute device /dev/kfd", OK, "present"))
    elif platform.system() == "Linux":
        out.append(("kfd", "compute device /dev/kfd", FAIL,
                    "present? no -- userland is installed but the kernel side is not"))

    if os.path.isdir("/dev/dri"):
        nodes = sorted(n for n in os.listdir("/dev/dri") if n.startswith("render"))
        if nodes:
            out.append(("dri", "render nodes", OK, ", ".join(nodes)))
        else:
            out.append(("dri", "render nodes", WARN, "/dev/dri exists, no renderD* nodes"))
    return out


def check_arch() -> Findings:
    code, out_text = run(["rocminfo"])
    if code != 0:
        return [("arch", "gfx target", INFO, "rocminfo unavailable")]
    targets = sorted({ln.split()[-1] for ln in out_text.splitlines()
                      if "gfx" in ln and ":" in ln})
    if not targets:
        return [("arch", "gfx target", INFO, "no gfx line in rocminfo output")]
    return [("arch", "gfx target", OK, ", ".join(targets))]


def check_override() -> Findings:
    val = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if val:
        return [("override", "HSA_OVERRIDE_GFX_VERSION", WARN,
                 f"set to {val} -- an override means the real target is masquerading")]
    return [("override", "HSA_OVERRIDE_GFX_VERSION", OK, "not set (real target in use)")]


def check_python_stack() -> Findings:
    out: Findings = []
    code, version = run(["python3", "-c", "import torch;print(torch.__version__)"])
    if code != 0:
        out.append(("torch", "PyTorch", WARN, "not importable in python3"))
    else:
        _, hip = run(["python3", "-c", "import torch;print(torch.version.hip)"])
        if hip.strip() in ("None", ""):
            out.append(("torch", "PyTorch", FAIL,
                        f"{first_line(version)} is a CPU/CUDA build, no HIP"))
        else:
            out.append(("torch", "PyTorch", OK, f"{first_line(version)} (HIP {hip.strip()})"))

    _, vllm = run(["python3", "-c", "import vllm;print(vllm.__version__)"])
    if vllm and "Error" not in vllm:
        out.append(("vllm", "vLLM", OK, first_line(vllm)))
    else:
        out.append(("vllm", "vLLM", INFO, "not installed"))
    return out


def check_tools() -> Findings:
    out: Findings = []
    for exe in ("llama-cli", "llama-server", "docker"):
        path = shutil.which(exe)
        out.append((f"tool-{exe}", exe, INFO, path or "not found"))
    return out


def collect() -> tuple[Findings, str]:
    findings: Findings = []
    for fn in (check_os, check_rocm_install, check_devices, check_arch,
               check_override, check_python_stack, check_tools):
        findings.extend(fn())

    fails = sum(1 for f in findings if f[2] == FAIL)
    linux = platform.system() == "Linux"
    if fails == 0:
        verdict = "environment looks usable for ROCm work"
    elif linux and fails <= 2:
        verdict = "close, but fix the FAIL lines before serving anything"
    else:
        verdict = ("not a ROCm host yet -- install the driver and the ROCm stack"
                   if not linux else "several hard blockers, fix the FAIL lines first")
    return findings, verdict


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Read-only ROCm/LLM environment probe.")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = p.parse_args(argv)

    findings, verdict = collect()

    if args.json:
        print(json.dumps({
            "host": platform.node(),
            "verdict": verdict,
            "checks": [{"id": i, "title": t, "status": s, "detail": d}
                       for i, t, s, d in findings],
        }, indent=2))
    else:
        for _id, title, status, detail in findings:
            print(f"[{status:<4}] {title:<28} {detail}")
        print()
        print(f"verdict: {verdict}")

    return 1 if any(f[2] == FAIL for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
