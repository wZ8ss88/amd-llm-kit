#!/usr/bin/env python3
"""Tests for the pure parts of the kit: kvcalc math, gfxmap lookups, bench timing.

    python -m unittest -v test_kit

The probe tools need hardware; these do not. bench.py is exercised against a
throwaway local SSE server, so its timing path runs here too.
"""

from __future__ import annotations

import http.server
import json
import math
import os
import tempfile
import threading
import time
import unittest

import bench
import gfxmap
import kvcalc

GIB = 1024 ** 3


class TestKvCache(unittest.TestCase):
    def test_formula_is_two_tensors(self):
        # 2 (K,V) * 1 layer * 1 head * 128 dim * 2 B * 1024 tokens * 1 batch
        self.assertEqual(kvcalc.kv_cache_bytes(1, 1, 128, 1024, 1, "fp16"),
                         2 * 1 * 1 * 128 * 2 * 1024)

    def test_gqa_pays_only_for_kv_heads(self):
        mha = kvcalc.kv_cache_bytes(80, 64, 128, 4096, 1, "fp16")
        gqa = kvcalc.kv_cache_bytes(80, 8, 128, 4096, 1, "fp16")
        self.assertEqual(gqa * 8, mha)

    def test_batch_scales_linearly(self):
        one = kvcalc.kv_cache_bytes(32, 8, 128, 2048, 1, "bf16")
        four = kvcalc.kv_cache_bytes(32, 8, 128, 2048, 4, "bf16")
        self.assertEqual(four, one * 4)

    def test_fp8_halves_bf16(self):
        bf16 = kvcalc.kv_cache_bytes(32, 8, 128, 2048, 1, "bf16")
        fp8 = kvcalc.kv_cache_bytes(32, 8, 128, 2048, 1, "fp8")
        self.assertEqual(fp8 * 2, bf16)

    def test_max_sequences_is_budget_over_per_sequence(self):
        budget = 4 * GIB
        n = kvcalc.max_sequences(32, 8, 128, 4096, "fp16", budget)
        per = kvcalc.kv_cache_bytes(32, 8, 128, 4096, 1, "fp16")
        self.assertEqual(n, budget // per)
        self.assertGreaterEqual(n, 1)

    def test_rejects_bad_input(self):
        for kwargs in ({"layers": 0}, {"kv_heads": 0}, {"head_dim": 0},
                       {"seq_len": 0}, {"batch": 0}, {"dtype": "float128"}):
            base = {"layers": 8, "kv_heads": 8, "head_dim": 128,
                    "seq_len": 128, "batch": 1, "dtype": "fp16"}
            base.update(kwargs)
            with self.assertRaises(ValueError):
                kvcalc.kv_cache_bytes(base["layers"], base["kv_heads"],
                                      base["head_dim"], base["seq_len"],
                                      base["batch"], base["dtype"])

    def test_human_units(self):
        self.assertEqual(kvcalc.human(512), "512 B")
        self.assertIn("GiB", kvcalc.human(2 * GIB))

    def test_cli_runs(self):
        self.assertEqual(kvcalc.main(["--layers", "32", "--kv-heads", "8",
                                      "--seq", "2048", "--vram", "192"]), 0)


class TestBench(unittest.TestCase):
    """Timing path against a local fake server -- no GPU, no real model."""

    DELTAS = 4
    GAP = 0.01

    def _build_body(self, usage, deltas):
        events = []
        for i in range(deltas):
            events.append("data: " + json.dumps(
                {"choices": [{"delta": {"content": "t%d " % i}}]}))
        if usage:
            events.append("data: " + json.dumps(
                {"choices": [], "usage": {"completion_tokens": 42}}))
        events.append("data: [DONE]")
        return [event + "\n\n" for event in events]

    def _serve(self, with_usage=True, deltas=None, status=200):
        deltas = self.DELTAS if deltas is None else deltas
        build = self._build_body
        gap = self.GAP

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if status != 200:
                    self.send_response(status)
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b"{}")
                    return
                events = build(with_usage and "stream_options" in body, deltas)
                total = sum(len(e.encode()) for e in events)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(total))
                self.end_headers()
                for event in events:
                    self.wfile.write(event.encode())
                    self.wfile.flush()
                    time.sleep(gap)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return "http://127.0.0.1:%d" % server.server_address[1]

    def test_one_request_times_the_stream(self):
        base = self._serve(with_usage=False)
        res = bench.one_request(base, "m", "hi", 8, "", 10.0)
        self.assertTrue(res["ok"])
        self.assertEqual(res["tokens"], self.DELTAS)
        self.assertEqual(res["counted"], "chunks")
        self.assertGreater(res["ttft"], 0.0)
        self.assertLessEqual(res["ttft"], res["total"])
        self.assertGreater(res["itl"], 0.0)
        self.assertTrue(math.isfinite(res["tps"]))

    def test_usage_block_wins_over_delta_count(self):
        base = self._serve()
        res = bench.one_request(base, "m", "hi", 8, "", 10.0)
        self.assertEqual(res["counted"], "usage")
        self.assertEqual(res["tokens"], 42)

    def test_no_stream_options_means_chunk_counting(self):
        base = self._serve()
        res = bench.one_request(base, "m", "hi", 8, "", 10.0, want_usage=False)
        self.assertEqual(res["counted"], "chunks")

    def test_empty_stream_raises(self):
        base = self._serve(deltas=0)
        with self.assertRaises(bench.BenchError):
            bench.one_request(base, "m", "hi", 8, "", 10.0)

    def test_http_error_propagates(self):
        base = self._serve(status=500)
        with self.assertRaises(Exception):
            bench.one_request(base, "m", "hi", 8, "", 10.0)

    def test_cli_writes_json_and_reports(self):
        base = self._serve()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "r.json")
            code = bench.main(["--base-url", base, "--model", "m", "--requests", "2",
                               "--concurrency", "2", "--warmup", "0", "--json", out])
            self.assertEqual(code, 0)
            data = json.load(open(out, encoding="utf-8"))
        self.assertEqual(data["summary"]["requests"], 2)
        self.assertEqual(data["summary"]["failed"], 0)
        self.assertEqual(data["summary"]["counted"], "usage")

    def test_cli_fails_cleanly_when_endpoint_is_dead(self):
        code = bench.main(["--base-url", "http://127.0.0.1:1", "--model", "m",
                           "--requests", "1", "--concurrency", "1", "--warmup", "0",
                           "--timeout", "2"])
        self.assertEqual(code, 1)

    def test_percentile_interpolates(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(bench.percentile(values, 0), 1.0)
        self.assertEqual(bench.percentile(values, 100), 4.0)
        self.assertAlmostEqual(bench.percentile(values, 50), 2.5)
        self.assertTrue(math.isnan(bench.percentile([], 99)))

    def test_rejects_non_positive_counts(self):
        with self.assertRaises(SystemExit):
            bench.main(["--base-url", "http://x", "--model", "m", "--requests", "0"])


class TestGfxMap(unittest.TestCase):
    def test_mi300x_target(self):
        row = gfxmap.lookup("gfx942")
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "CDNA 3")
        self.assertEqual(row[2], 64)

    def test_rdna_uses_wave32(self):
        for target in ("gfx1100", "gfx1030", "gfx1201"):
            self.assertEqual(gfxmap.lookup(target)[2], 32)

    def test_cdna3_advertises_fp8(self):
        self.assertIn("fp8", gfxmap.lookup("gfx942")[4])

    def test_unknown_target(self):
        self.assertIsNone(gfxmap.lookup("gfx9999"))
        self.assertIn("unknown", gfxmap.describe("gfx9999"))
        self.assertEqual(gfxmap.main(["gfx9999"]), 1)

    def test_cli_known_target(self):
        self.assertEqual(gfxmap.main(["gfx90a"]), 0)

    # Product rows follow AMD's published target table; these are the ones
    # that got mixed up before, so pin them.
    def test_mi325_shares_the_mi300_target(self):
        self.assertIn("MI325X", gfxmap.lookup("gfx942")[1])
        self.assertNotIn("MI325X", gfxmap.lookup("gfx950")[1])

    def test_mi300a_is_gfx942_in_production(self):
        self.assertIn("MI300A", gfxmap.lookup("gfx942")[1])

    def test_gfx950_is_mi350_series(self):
        for product in ("MI350X", "MI355X"):
            self.assertIn(product, gfxmap.lookup("gfx950")[1])

    def test_navi_48_and_navi_44_are_distinct(self):
        self.assertEqual(gfxmap.lookup("gfx1201")[1], "RX 9070, RX 9070 XT")
        self.assertEqual(gfxmap.lookup("gfx1200")[1], "RX 9060, RX 9060 XT")

    def test_no_product_is_claimed_by_two_shipping_targets(self):
        # gfx940/gfx941 are A0 steppings and legitimately repeat the production
        # product names, so they are left out of the uniqueness check.
        seen = {}
        for target, row in gfxmap.ARCH.items():
            if target in ("gfx940", "gfx941"):
                continue
            for product in row[1].split(","):
                product = product.strip().split(" (")[0]
                self.assertNotIn(product, seen,
                                 f"{product} listed under {seen.get(product)} and {target}")
                seen[product] = target


if __name__ == "__main__":
    unittest.main()
