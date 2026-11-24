# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Layer 1, self-consistent exactness: every row is the exact canonical top-k of the dense scan's scores (the same
arithmetic, so equality is bit for bit), on every path: the scan, pruning, adaptive pruning, and both recoveries."""

import pytest
import torch
from conftest import DTYPES, TOPK, assert_same, canonical_topk, dense_scores, make_call, requires_gpu

import flashtopk
from flashtopk import topk as topk_mod
from flashtopk.kernels import launch

pytestmark = requires_gpu

SHAPES = [
    pytest.param(1, 5000, "causal", id="1x5k"),
    pytest.param(7, 3000, "causal", id="7x3k-short"),
    pytest.param(129, 20000, "random", id="129x20k-random"),
    pytest.param(300, 9000, "short", id="300x9k-windows<topk"),
    pytest.param(515, 140000, "causal", id="515x140k"),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,seq_len_kv,windows", SHAPES)
@pytest.mark.parametrize("pruning", ["off", "always", "adaptive"])
def test_exact(dtype, rows, seq_len_kv, windows, pruning):
    q, kv, s, w, ks, ke = make_call(rows, seq_len_kv, dtype, seed=rows, windows=windows)
    got = flashtopk.Selector(TOPK, dtype=dtype, pruning=pruning, return_scores=True).run(q, kv, w, ks, ke, s)
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_len_kv", [20000, 70000])
def test_per_row_recovery(dtype, seq_len_kv, monkeypatch):
    """Nine rows' gates closed after calibration come up short; per-row recovery (sliced past 32K keys) fixes them."""
    q, kv, s, w, ks, ke = make_call(515, seq_len_kv, dtype, seed=3)
    rows = torch.arange(514, 514 - 9 * 37, -37, device="cuda")
    seed_prep = launch.seed_prep

    def closed(sample, ks_, ke_, lo, hi, topk, slack, k_min, origin, inv_delta, th_bucket, cand_cnt):
        seed_prep(sample, ks_, ke_, lo, hi, topk, slack, k_min, origin, inv_delta, th_bucket, cand_cnt)
        th_bucket[rows] = 0
        inv_delta[rows] = 255.0 / 1e-6

    monkeypatch.setattr(launch, "seed_prep", closed)
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning="off", return_scores=True)
    got = sel.run(q, kv, w, ks, ke, s)
    assert sel.last_info.bad_rows == 9
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_whole_call_recovery(dtype, monkeypatch):
    """A candidate buffer too small for most rows: whole-call recovery rescores every row."""
    q, kv, s, w, ks, ke = make_call(515, 20000, dtype, seed=4)
    monkeypatch.setattr(topk_mod, "candidate_cap", lambda topk, num_q, seq_len_kv: 2100)
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning="off", return_scores=True)
    got = sel.run(q, kv, w, ks, ke, s)
    assert sel.last_info.bad_rows > 64
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))


@pytest.mark.parametrize("topk", [1, 64, 777, 4096])
def test_other_topk(topk):
    q, kv, s, w, ks, ke = make_call(130, 12000, torch.float8_e4m3fn, seed=5, windows="random")
    sel = flashtopk.Selector(topk, dtype=torch.float8_e4m3fn, pruning="off", return_scores=True)
    got = sel.run(q, kv, w, ks, ke, s)
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke), topk))


def test_ties():
    """Identical keys tie exactly: equal scores go to the smallest indices, in index order."""
    q, kv, s, w, ks, ke = make_call(64, 9000, torch.bfloat16, seed=6, windows="causal")
    kv[1::3] = kv[0]  # a third of the keys share one score per row
    got = flashtopk.Selector(TOPK, dtype=torch.bfloat16, pruning="off", return_scores=True).run(q, kv, w, ks, ke, s)
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_prune_24col_exact(dtype, monkeypatch):
    """SM100 24-column pruning (22 retained heads + 2 aggregate rows) is bit-exact against the dense scan."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("24-column pruning is SM100-only")
    from flashtopk import plan as planner

    plan_operands = planner.plan_operands
    monkeypatch.setattr(planner, "plan_operands", lambda *a, **kw: plan_operands(*a, cols=24, **kw))
    q, kv, s, w, ks, ke = make_call(135, 20000, dtype, seed=17, windows="random")
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning="always", return_scores=True)
    got = sel.run(q, kv, w, ks, ke, s)
    assert sel.last_info.pruned is True
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_partial_wave_tail_split_exact(dtype):
    """Partial-wave tail splitting in ScanSm100 matches the dense scan bit for bit."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("partial-wave tail split is SM100-only")
    sm = launch.sm_count(0)
    rows = (sm + 20) * 8 + 3
    seq_len_kv = 20000
    splits, full_tiles, tail_chunks = launch.pick_scan_schedule(rows, seq_len_kv, sm)
    assert splits == 1 and full_tiles == sm and tail_chunks > 1
    q, kv, s, w, ks, ke = make_call(rows, seq_len_kv, dtype, seed=19, windows="causal")
    got = flashtopk.Selector(TOPK, dtype=dtype, pruning="off", return_scores=True).run(q, kv, w, ks, ke, s)
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))


def test_three_way_adaptive_and_guards(monkeypatch):
    """Adaptive pruning skips small-row calls and short effective windows, and chooses 24-col when cheaper on SM100."""
    from flashtopk import plan as planner

    # 1. Small-row guard (< ADAPTIVE_MIN_ROWS) scans without entering the planner.
    q, kv, s, w, ks, ke = make_call(256, 140000, torch.float8_e4m3fn, seed=21)
    sel = flashtopk.Selector(TOPK, dtype=torch.float8_e4m3fn, pruning="adaptive", return_scores=True)
    got = sel.run(q, kv, w, ks, ke, s)
    assert sel.last_info.pruned is False and sel.last_info.prune_cols == 0
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))

    # 2. Short effective window (small ke inside 256K KV) stays on full scan via eff_L scaling.
    q, kv, s, w, ks, ke = make_call(520, 262144, torch.float8_e4m3fn, seed=22)
    ks_short = torch.zeros_like(ks)
    ke_short = torch.arange(1, 521, device="cuda", dtype=torch.int32) * 8
    sel = flashtopk.Selector(TOPK, dtype=torch.float8_e4m3fn, pruning="adaptive", return_scores=True)
    got = sel.run(q, kv, w, ks_short, ke_short, s)
    assert sel.last_info.pruned is False
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks_short, ke_short)))

    # 3. On SM100 at >=256K keys, 3-way adaptive picks 24-col when 24-col's cost beats scan and 16-col.
    if torch.cuda.get_device_capability()[0] == 10:
        kv_f = kv.float()
        small = (torch.rand(kv.shape[0], device="cuda") > 0.03)[:, None]
        kv_tight = torch.where(small, kv_f * 0.01, kv_f).to(torch.float8_e4m3fn).contiguous()
        monkeypatch.setenv("FLASHTOPK_PRUNE_COSTS", "0.08,0.95,2.0")
        monkeypatch.setenv("FLASHTOPK_PRUNE_COSTS_24", "0.05,0.20,1.0")
        planner.prune_costs.cache_clear()
        try:
            sel24 = flashtopk.Selector(TOPK, dtype=torch.float8_e4m3fn, pruning="adaptive", return_scores=True)
            got24 = sel24.run(q, kv_tight, w.abs(), ks, ke, s)
            assert sel24.last_info.pruned is True and sel24.last_info.prune_cols == 24
            assert_same(got24, canonical_topk(dense_scores(q, kv_tight, s, w.abs(), ks, ke)))
        finally:
            planner.prune_costs.cache_clear()


@pytest.mark.parametrize("dtype", DTYPES)
def test_short_kv_dense(dtype):
    """Short KV uses the direct dense path bit-exactly."""
    q, kv, s, w, ks, ke = make_call(128, 4096, dtype, seed=23, windows="random")
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning="off", return_scores=True)
    got = sel.run(q, kv, w, ks, ke, s)
    assert sel.last_info.bad_rows == 0
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))


def test_select_fast_and_edge_cases():
    """1-pass 12-bit SMEM Select matches canonical top-k on extreme scales, -0/+0, NaN/inf, crowded ties, and n>8192."""
    g = torch.Generator(device="cuda").manual_seed(29)
    for n, k in ((1200, 2048), (4096, 2048), (8192, 2048), (10000, 2048), (4096, 1), (4096, 777), (8192, 4096)):
        rows = 12
        scores = torch.randn(rows, n, generator=g, device="cuda", dtype=torch.float32)
        scores[1] *= 1e6  # far past FP16 max (65504)
        scores[2] *= 1e-6  # far below FP16 normal min
        scores[3].fill_(0.0)
        scores[3, ::2] = -0.0  # signed zero tie + zero span fallback
        scores[4, :600] = 5.0  # >BUCKET_LIMIT identical top scores (sample sort fallback)
        scores[5, :3000] = 1.0  # >TIE_CAP pivot ties (4-pass radix fallback)
        scores[6, ::7] = float("nan")
        scores[6, ::11] = float("-inf")
        scores[6, ::13] = float("inf")
        scores[7, :n // 2] = 1e38
        scores[7, n // 2:] = -1e38  # infinite span fallback
        ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
        ke = torch.full((rows,), n, dtype=torch.int32, device="cuda")
        row_map = torch.arange(rows, dtype=torch.int32, device="cuda")
        out_s = torch.empty(rows, k, dtype=torch.float32, device="cuda")
        out_i = torch.empty(rows, k, dtype=torch.int32, device="cuda")
        launch.select_dense(scores, ks, ke, row_map, out_s, out_i)
        # Compare against canonical_topk over finite entries (matching Select's finite-only contract).
        ref_in = torch.where(torch.isfinite(scores), scores, torch.tensor(float("-inf"), device="cuda"))
        assert_same((out_s, out_i), canonical_topk(ref_in, k))
