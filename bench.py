#!/usr/bin/env python3
"""Measure TTFT, inter-token latency and throughput against an OpenAI-compatible endpoint.

Streaming is required for all three: TTFT is the time to the first content
delta, ITL is the gap between consecutive deltas, throughput is tokens over
wall time. Token counts come from the server's ``usage`` block when it sends
one (``stream_options.include_usage``); if the server rejects that field the
request is retried without it and the count falls back to content deltas,
which is reported honestly as ``counted=chunks`` rather than passed off as
exact.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


class BenchError(Exception):
    pass


def percentile(values, pct):
    """Linear-interpolated percentile, same convention as numpy's default."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _payload(delta):
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    return ""


def one_request(base_url, model, prompt, max_tokens, api_key, timeout, want_usage=True):
    """Run one streamed request and time it.

    Returns a dict with ttft/total/tokens/tps/itl and counted=usage|chunks.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    if want_usage:
        body["stream_options"] = {"include_usage": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    start = time.perf_counter()
    first = None
    gaps = []
    tokens = 0
    usage_tokens = None

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            chunk = raw.decode("utf-8", "replace").strip()
            if not chunk.startswith("data:"):
                continue
            payload = chunk[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            usage = obj.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                usage_tokens = usage["completion_tokens"]

            choices = obj.get("choices") or [{}]
            delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
            if not _payload(delta):
                continue

            now = time.perf_counter()
            if first is None:
                first = now
            else:
                gaps.append(now - last)
            last = now
            tokens += 1

    end = time.perf_counter()
    if first is None:
        raise BenchError("no content deltas received")

    counted = "usage" if usage_tokens else "chunks"
    if usage_tokens:
        tokens = usage_tokens
    return {
        "ok": True,
        "ttft": first - start,
        "total": end - start,
        "tokens": tokens,
        "counted": counted,
        "tps": tokens / (end - first) if end > first else 0.0,
        "itl": statistics.mean(gaps) * 1000 if gaps else float("nan"),
    }


def run_once(args, index, lock, results):
    try:
        res = one_request(args.base_url, args.model, args.prompt, args.max_tokens,
                          args.api_key, args.timeout, want_usage=not args.no_usage)
    except urllib.error.HTTPError as exc:
        if not args.no_usage and exc.code in (400, 422):
            try:
                res = one_request(args.base_url, args.model, args.prompt, args.max_tokens,
                                  args.api_key, args.timeout, want_usage=False)
                res["note"] = "server rejected stream_options, counted chunks"
            except Exception as inner:  # noqa: BLE001 - reported, not raised
                res = {"ok": False, "error": str(inner)}
        else:
            res = {"ok": False, "error": "HTTP " + str(exc.code)}
    except Exception as exc:  # noqa: BLE001 - every failure is one failed sample
        res = {"ok": False, "error": str(exc)}
    res["index"] = index
    with lock:
        results.append(res)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Explain in three sentences why p99 latency matters more than the mean.")
    ap.add_argument("--requests", type=positive, default=16)
    ap.add_argument("--concurrency", type=positive, default=4)
    ap.add_argument("--max-tokens", type=positive, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--no-usage", action="store_true",
                    help="never ask for a usage block; count content deltas only")
    ap.add_argument("--json", default="", metavar="PATH")
    args = ap.parse_args(argv)

    for _ in range(max(0, args.warmup)):
        try:
            one_request(args.base_url, args.model, args.prompt, 16, args.api_key,
                        args.timeout, want_usage=not args.no_usage)
        except Exception as exc:  # noqa: BLE001 - warmup is best-effort
            print("warmup failed:", exc, file=sys.stderr)

    results = []
    lock = threading.Lock()
    sem = threading.Semaphore(args.concurrency)

    def worker(index):
        with sem:
            run_once(args, index, lock, results)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.requests)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started

    ok = [r for r in results if r.get("ok")]
    failed = len(results) - len(ok)

    for res in sorted(results, key=lambda x: x["index"]):
        if res.get("ok"):
            print("{:>3}  ttft {:7.3f}s  itl {:7.1f}ms  {:6.1f} tok/s  n={} ({})".format(
                res["index"], res["ttft"], res["itl"], res["tps"], res["tokens"], res["counted"]))
        else:
            print("{:>3}  FAIL  {}".format(res["index"], res["error"]))

    ttfts = [r["ttft"] for r in ok]
    itls = [r["itl"] for r in ok]
    total_tokens = sum(r["tokens"] for r in ok)
    counted = {r["counted"] for r in ok}

    summary = {
        "requests": len(results),
        "failed": failed,
        "wall_seconds": wall,
        "counted": "usage" if counted == {"usage"} else "chunks" if counted == {"chunks"} else "mixed" if counted else "none",
        "aggregate_tokens_per_second": total_tokens / wall if wall else 0.0,
        "ttft_ms": {p: percentile(ttfts, p) * 1000 for p in (50, 90, 99)},
        "itl_ms": {p: percentile(itls, p) for p in (50, 90, 99)},
    }

    print()
    print("ok {}  failed {}  token count from {}".format(len(ok), failed, summary["counted"]))
    print("aggregate: {:.1f} tok/s over {:.2f}s wall".format(summary["aggregate_tokens_per_second"], wall))
    if ok:
        print("ttft p50/p90/p99 ms: {:.0f} / {:.0f} / {:.0f}".format(
            *[summary["ttft_ms"][p] for p in (50, 90, 99)]))
        print("itl  p50/p90/p99 ms: {:.1f} / {:.1f} / {:.1f}".format(
            *[summary["itl_ms"][p] for p in (50, 90, 99)]))
    else:
        print("no successful sample; p50/p90/p99 are nan")
    if summary["counted"] in ("chunks", "mixed"):
        print("note: token counts follow content deltas, so they are an approximation")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "results": results}, fh, indent=2)
        print("wrote", args.json)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
