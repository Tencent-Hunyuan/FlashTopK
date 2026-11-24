# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Small per-row prologue kernels in CuTe DSL.

  * SampleGather: gathers the shared sample keys kv[p_j] and FP8 scales (4-float tail padded), computes each row's
    sample window [lo, hi), and zeros the call's health in one launch;
  * Decide: adaptive pruning's device-side choice (2-way scan vs 16-col, or 3-way scan vs 16-col vs 24-col) using
    the estimate groups' effective union window length, writing the row's window to the chosen path and empty windows
    to the others.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32

from .common import HEAD_DIM, fmax, load_global_v4_b32, select_i32, store_global_v4_b32
from .selection import HEALTH_WORDS
from .simt import THREADS, WARPS


@cute.jit
def samples_below(x, seq_len_kv, samples) -> Int32:
    """Shared-sample keys p_j = floor((2j + 1) L / (2M)) below x: p_j < x iff (2j + 1) L <= 2Mx - 1."""
    n = Int32(0)
    if x > 0:
        odd_max = (cutlass.Int64(2) * cutlass.Int64(samples) * cutlass.Int64(x) - 1) // cutlass.Int64(seq_len_kv)
        n = Int32(min((odd_max + 1) // 2, cutlass.Int64(samples)))
    return n


class SampleGather:
    """Fused shared-sample prep: gathers kv[p_j] and FP8 scales[p_j] (4-float zero padded), computes [lo, hi), and
    zeros health in one 128-bit coalesced launch."""

    def __init__(self, *, bf16: bool = False):
        self.bf16 = bf16
        self.row_bytes = HEAD_DIM * (2 if bf16 else 1)
        self.chunks_per_row = self.row_bytes // 16

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,
        kv_scales: cute.Tensor,
        ks: cute.Tensor,
        ke: cute.Tensor,
        kv_out: cute.Tensor,
        scales_out: cute.Tensor,
        lo: cute.Tensor,
        hi: cute.Tensor,
        health: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        samples: Int32,
        padded_samples: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = max((num_q + THREADS - 1) // THREADS, (padded_samples + 15) // 16)
        self.kernel(
            kv, kv_scales, ks, ke, kv_out, scales_out, lo, hi, health,
            num_q, seq_len_kv, samples, padded_samples,
        ).launch(grid=(grid_x, 1, 1), block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self, mKV, mScales, mKs, mKe, mKVOut, mScalesOut, mLo, mHi, mHealth,
        num_q: Int32, seq_len_kv: Int32, samples: Int32, padded_samples: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        if (bid == 0) & (tid < HEALTH_WORDS):
            mHealth[tid] = cutlass.Int64(0)
        r = bid * THREADS + tid
        if r < num_q:
            mLo[r] = samples_below(mKs[r], seq_len_kv, samples)
            mHi[r] = samples_below(mKe[r], seq_len_kv, samples)
        j = bid * 16 + (tid // 16)
        lane_s = tid % 16
        if j < samples:
            pos = ((cutlass.Int64(2) * cutlass.Int64(j) + 1) * cutlass.Int64(seq_len_kv)) // (
                cutlass.Int64(2) * cutlass.Int64(samples)
            )
            if lane_s < self.chunks_per_row:
                elem_bytes = 2 if self.bf16 else 1
                src_addr = mKV.iterator.toint() + pos * cutlass.Int64(mKV.layout.stride[0]) * cutlass.Int64(
                    elem_bytes
                ) + cutlass.Int64(lane_s * 16)
                dst_addr = mKVOut.iterator.toint() + cutlass.Int64(j) * cutlass.Int64(self.row_bytes) + cutlass.Int64(
                    lane_s * 16
                )
                v0, v1, v2, v3 = load_global_v4_b32(src_addr)
                store_global_v4_b32(dst_addr, v0, v1, v2, v3)
            if cutlass.const_expr(not self.bf16):
                if lane_s == 0:
                    mScalesOut[j] = mScales[Int32(pos)]
        elif j < padded_samples:
            if cutlass.const_expr(not self.bf16):
                if lane_s == 0:
                    mScalesOut[j] = Float32(0.0)


EST_GROUP = 8  # the refine's row group (refine_sm90 / refine_sm100 BLOCK_Q)
EST_WORDS = 16  # an estimate row's in-window mask over the 256 sample keys (8 words), then its past-the-gate mask


class Decide:
    """Device-side adaptive pruning choice: 2-way (0: scan, 1: 16-col) or 3-way (0: scan, 1: 16-col, 2: 24-col)."""

    def __init__(self, *, three_way: bool = False, use_eff_window: bool = False):
        self.three_way = three_way
        self.use_eff_window = use_eff_window

    @cute.jit
    def __call__(
        self,
        est16: cute.Tensor,
        est24: cute.Tensor,
        ks: cute.Tensor,
        ke: cute.Tensor,
        scan_ks: cute.Tensor,
        prune16_ks: cute.Tensor,
        prune24_ks: cute.Tensor,
        flag: cute.Tensor,
        estimate: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        p16: Float32,
        c16: Float32,
        b16: Float32,
        p24: Float32,
        c24: Float32,
        b24: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            est16, est24, ks, ke, scan_ks, prune16_ks, prune24_ks, flag, estimate,
            num_q, seq_len_kv, p16, c16, b16, p24, c24, b24,
        ).launch(grid=((num_q + THREADS - 1) // THREADS, 1, 1), block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self, mEst16, mEst24, mKs, mKe, mScanKs, mPrune16Ks, mPrune24Ks, mFlag, mEstimate,
        num_q: Int32, seq_len_kv: Int32,
        p16: Float32, c16: Float32, b16: Float32,
        p24: Float32, c24: Float32, b24: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = utils.SmemAllocator()
        sWin = smem.allocate_tensor(Float32, cute.make_layout((WARPS,)), 16)
        sPass16 = smem.allocate_tensor(Float32, cute.make_layout((WARPS,)), 16)
        sPass24 = smem.allocate_tensor(Float32, cute.make_layout((WARPS,)), 16)
        groups = mEst16.layout.shape[0] // EST_GROUP
        half = EST_WORDS // 2
        n_win = Float32(0.0)
        n_pass16 = Float32(0.0)
        n_pass24 = Float32(0.0)
        for g in cutlass.range(tid, groups, THREADS, unroll=1):
            for w in cutlass.range_constexpr(half):
                win = Int32(0)
                kept16 = Int32(0)
                kept24 = Int32(0)
                for r in cutlass.range_constexpr(EST_GROUP):
                    win = win | mEst16[(g * EST_GROUP + r, w)]
                    kept16 = kept16 | mEst16[(g * EST_GROUP + r, half + w)]
                    if cutlass.const_expr(self.three_way):
                        kept24 = kept24 | mEst24[(g * EST_GROUP + r, half + w)]
                n_win = n_win + Float32(cute.arch.popc(win))
                n_pass16 = n_pass16 + Float32(cute.arch.popc(kept16))
                if cutlass.const_expr(self.three_way):
                    n_pass24 = n_pass24 + Float32(cute.arch.popc(kept24))
        n_win = cute.arch.warp_reduction_sum(n_win)
        n_pass16 = cute.arch.warp_reduction_sum(n_pass16)
        if cutlass.const_expr(self.three_way):
            n_pass24 = cute.arch.warp_reduction_sum(n_pass24)
        if tid % 32 == 0:
            sWin[tid // 32] = n_win
            sPass16[tid // 32] = n_pass16
            if cutlass.const_expr(self.three_way):
                sPass24[tid // 32] = n_pass24
        cute.arch.sync_threads()
        n_win = Float32(0.0)
        n_pass16 = Float32(0.0)
        n_pass24 = Float32(0.0)
        for w in cutlass.range_constexpr(WARPS):
            n_win = n_win + sWin[w]
            n_pass16 = n_pass16 + sPass16[w]
            if cutlass.const_expr(self.three_way):
                n_pass24 = n_pass24 + sPass24[w]

        plan_l = Float32(seq_len_kv)
        if cutlass.const_expr(self.use_eff_window):
            if groups > 0:
                plan_l = fmax(Float32(seq_len_kv) * (n_win / Float32(groups * 256)), Float32(1.0))
        plan_scale = Float32(131072.0) / plan_l

        share16 = Float32(1.0)
        if n_win > Float32(0.0):
            share16 = n_pass16 / n_win
        cost16 = p16 * plan_scale + c16 + b16 * share16

        choice = Int32(0)
        best_cost = Float32(1.0)
        best_share = share16
        if (n_win > Float32(0.0)) & (cost16 <= best_cost):
            choice = Int32(1)
            best_cost = cost16
            best_share = share16
        if cutlass.const_expr(self.three_way):
            share24 = Float32(1.0)
            if n_win > Float32(0.0):
                share24 = n_pass24 / n_win
            cost24 = p24 * plan_scale + c24 + b24 * share24
            if (n_win > Float32(0.0)) & (cost24 < best_cost):
                choice = Int32(2)
                best_cost = cost24
                best_share = share24

        if (bid == 0) & (tid == 0):
            mFlag[0] = choice
            mEstimate[0] = best_share
        r = bid * THREADS + tid
        if r < num_q:
            s = mKs[r]
            e = mKe[r]
            mScanKs[r] = select_i32(choice == Int32(0), s, e)
            mPrune16Ks[r] = select_i32(choice == Int32(1), s, e)
            if cutlass.const_expr(self.three_way):
                mPrune24Ks[r] = select_i32(choice == Int32(2), s, e)
