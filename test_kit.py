#!/usr/bin/env python3
"""Tests for the pure parts of the kit: kvcalc math and gfxmap lookups.

    python -m unittest -v test_kit

The probe tools need hardware; these do not.
"""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
