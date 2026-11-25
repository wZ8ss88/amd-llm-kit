# amd-llm-kit

Five small, read-only helpers for running LLM inference on AMD GPUs. No
framework, no daemon, no wrapper around your wrapper — standard library
Python and one bash launcher.

They exist because the boring questions arrive before the interesting ones:
does this box even have a usable ROCm stack, what is this `gfx` string, how
much VRAM will the KV cache take, is the card already full, and does the
server actually keep its latency once it is up.

| tool | question it answers |
| --- | --- |
| `doctor.py` | is this host able to run anything on ROCm at all? |
| `gfxmap.py` | what is `gfx942` / `gfx1100` and what precisions does it have? |
| `kvcalc.py` | how much VRAM does the KV cache need for this context and batch? |
| `vramwatch.py` | is the card about to OOM while the model loads? |
| `bench.py` | what TTFT / ITL / throughput does a running endpoint really give? |

`serve_vllm.sh` wires the probes into a vLLM launch so the checks run before
the server does.

**Status, plainly:** the cache math and the target table are covered by tests
and run anywhere (this was developed on Windows). `bench.py` is also covered
by tests, but against a throwaway local SSE server — its numbers only mean
something against a real endpoint. The probe tools call `rocm-smi`,
`rocminfo` and `/opt/rocm`, so they only say something useful on a Linux host
with ROCm; everywhere else they report `MISSING` instead of pretending.
Nothing here has been benchmarked against vendor numbers; the formulas and
flags are the documented ones, the outputs are what they are.

## Requirements

- Python 3.9+, no third-party packages.
- For the probe tools: Linux + ROCm (`rocm-smi`, `rocminfo`) and
  `rocm-smi` on `PATH`.
- For `serve_vllm.sh`: `vllm` on `PATH`.
- For `bench.py`: a reachable OpenAI-compatible endpoint.

## doctor.py

Read-only probe of OS, ROCm install, `/dev/kfd`, render nodes, gfx target,
`HSA_OVERRIDE_GFX_VERSION`, the Python stack and the usual serving binaries.
Every check is timeboxed; nothing is installed.

Sample from a working ROCm host (the numbers below are illustrative — this
kit itself was written on Windows, see **Status**):

```
$ python3 doctor.py
[OK  ] operating system             Linux-6.8.0-...-generic-x86_64
[OK  ] ROCm install                 version 6.2.0
[OK  ] rocm-smi on PATH             /opt/rocm/bin/rocm-smi
[OK  ] compute device /dev/kfd      present
[OK  ] gfx target                   gfx942
[OK  ] PyTorch                      2.5.0 (HIP 6.2.41134)
verdict: environment looks usable for ROCm work
```

An `HSA_OVERRIDE_GFX_VERSION` is reported as `WARN` on purpose: an override
means the real target is masquerading as another one, which is exactly the
thing worth knowing when a kernel silently misbehaves. `--json` gives the
same data for scripts; the exit code is non-zero when anything is `FAIL`.

## gfxmap.py

Explains a gfx target: family, products, wavefront size, whether it has
matrix cores, which precisions it supports, and whether it is data-centre or
consumer silicon.

```
$ python3 gfxmap.py gfx942
gfx942    CDNA 3  (MI300X)
          wavefront 64, matrix cores: MFMA
          supported precisions: fp32, fp16, bf16, fp8, fp6
          the MI300X target; fp8 KV cache works
```

`--all` prints the table. The practical distinction it encodes: CDNA parts
have MFMA and are what serving stacks ship kernels for; RDNA parts are
wave32 and graphics-first. Useful before you promise someone fp8 on a
desktop card.

Product rows are pinned by tests, because a wrong row is worse than a
missing one: `gfx940`/`gfx941` are A0 steppings that production MI300
enumerates past (`gfx942`), MI325X shares `gfx942`, `gfx950` is MI350/MI355,
and Navi 48 (`gfx1201`, RX 9070/XT) is not Navi 44 (`gfx1200`, RX 9060/XT).
When `rocminfo` disagrees with this table, `rocminfo` is right — open an
issue with the line.

## kvcalc.py

Sizes the KV cache from the model shape:

```
bytes = 2 * layers * kv_heads * head_dim * bytes_per_element * seq * batch
```

`kv_heads` is the key/value head count, so grouped-query attention is
handled by construction — a 64-head model with 8 KV heads pays for 8, not 64
(there is a test for that).

```
$ python3 kvcalc.py --layers 80 --kv-heads 8 --head-dim 128 --seq 32768 --vram 192 --weights 140
per sequence    10.00 GiB
total cache     10.00 GiB
weights         140.00 GiB
weights + cache 150.00 GiB
share of 192 GiB   5.2%
```

Weights are excluded unless you pass `--weights`; this is the cache only.
`--fit 192` inverts the question into "how many 32k sequences fit", and
`--dtype fp8` shows what halving the cache buys.

## vramwatch.py

Polls `rocm-smi --showmeminfo vram` and prints used/total per card, flagging
cards above a threshold. Run it in one terminal, start the server in the
other.

```
$ python3 vramwatch.py --interval 1 --alert-pct 90
--- 12:04:11 ---
 card0      9.4 /  191.7 GiB   4.9% |#                        |
 card0     48.2 /  191.7 GiB  25.1% |######                   |
```

`--once` is for scripts: exit `0` when clear, `2` when a card is hot, `3`
when `rocm-smi` is missing.

## bench.py

Times a live endpoint: TTFT per request, inter-token latency, and aggregate
tokens per second under a fixed concurrency. Streaming only, because none of
the three means anything without it.

```
$ python3 bench.py --base-url http://127.0.0.1:8000 --model <served-name> --requests 16 --concurrency 4
  0  ttft   0.181s  itl    21.4ms    46.7 tok/s  n=128 (usage)
  ...
ok 16  failed 0  token count from usage
aggregate: 430.2 tok/s over 4.76s wall
ttft p50/p90/p99 ms: 190 / 244 / 261
itl  p50/p90/p99 ms: 21.0 / 23.9 / 25.1 ms
```

Two things it is deliberate about:

- **Token counts carry their own provenance.** It asks for
  `stream_options.include_usage`; a server that sends a `usage` block gives
  exact completion tokens and the sample is tagged `usage`. A server that
  rejects the field gets one retry without it, the count falls back to
  content deltas, and every line is tagged `chunks` — an approximation, said
  out loud instead of presented as exact.
- **Failures are samples, not exceptions.** A timeout or an HTTP error is
  reported per index and counted in `failed`; the exit code is `1` only when
  no request succeeded.

`--warmup` discards leading requests, `--json PATH` dumps the full per-request
set, `--timeout` bounds each request. The percentiles are linear-interpolated
over successful samples only.

## serve_vllm.sh

Preflight, then `exec vllm serve` — the shell is replaced, so signals reach
vLLM directly.

```
LAYERS=80 KV_HEADS=8 bash serve_vllm.sh meta-llama/Llama-3.1-8B-Instruct 8192 1
```

Defaults are `--gpu-memory-utilization 0.90` and `--dtype bfloat16`; `TP`,
`PORT`, `GPU_UTIL`, `KV_DTYPE=fp8` and `EXTRA_ARGS` are overridable from the
environment. It refuses to start when a card is already above 90% VRAM.
Set `LAYERS` and `KV_HEADS` and it prints the planned cache first.

## Tests

```
python3 -m unittest -v test_kit
```

Thirteen tests over the cache math, the input validation, the target table
and both CLIs, plus nine more over `bench.py` (percentiles, the `usage`/chunk
distinction, HTTP and empty-stream failures, the CLI end to end) driven
against a temporary in-process SSE server. The probe tools need hardware and
are not covered.

A bench run of the test suite on this machine, for what it is worth: 22 tests
in 2.4 s.

## Licence

MIT.
