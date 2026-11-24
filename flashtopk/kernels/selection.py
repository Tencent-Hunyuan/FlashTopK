# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
#
# The radix selection follows LiteTopK's compact top-k kernel (https://github.com/Heisenberg-Yin/LiteTopK, Apache 2.0).
"""Selection: each row's exact top-k over its candidate buffer, in canonical order, with the call's health, in CuTe DSL.

One CTA per row (256 threads). The candidates are (-score, kv index) pairs, the scan's or the refine's:

  * fast path (n <= FAST_MAX = 8192): one 16-byte coalesced load moves the row's scores into 32 KB of shared memory
    (computing min/max/finite count in flight), one 4096-bin (12-bit, packed 16-bit) monotonic log histogram finds
    the pivot bin and every higher bin's exact sorted offset in two barriers, one SMEM sweep scatters winning
    positions by bucket and collects the <=256 pivot-bin ties into exact 64-bit order keys, and only the K winning
    indices are gathered from global memory (zero index reads in DENSE mode). Because the 4096-bin histogram already
    groups winners monotonically by bucket (85%+ singletons), canonical sorting skips the extra log-bucketing pass;
  * exact fallback (n > 8192, zero/infinite span, or >256 elements in the pivot bin): four 8-bit value radix passes
    (plus four index passes on a crowded pivot) and log-bucket / sample canonical sort;
  * health: a row whose count passed the buffer (overflow) or that selected fewer than min(k, window) keys
    (underfill) is bad; the first RECOVERY_ROWS bad rows are recorded for recovery.

Health words (int64): 0 overflow rows, 1 underfill rows, 3 the largest candidate count (past the buffer, or always
with stats), 4 the candidate total (with stats), 7 bad rows.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32

from .common import add_rn, div_rn, fmax, fmin, load_global_v4_b32, mul_rn, select_i32, sub_rn
from .simt import (
    THREADS,
    WARPS,
    ascending_key,
    atomic_add_global_u64,
    atomic_add_shared_u32,
    atomic_max_global_u64,
    atomic_max_shared_u32,
    atomic_min_shared_u32,
    bit_length_u32,
    bits_f32,
    block_inclusive_scan_i32,
    block_max_f32,
    block_max_i32,
    block_min_f32,
    block_sum_i32,
    f32_bits,
    f32_to_i32_rz,
    finite,
    load_shared_v4_b32,
    red_add_shared_u32,
    store_shared_v4_b32,
)

RADIX = 256
LOG_BUCKETS = 2048
SORT_BUCKETS = LOG_BUCKETS  # 2048 u32s = 8 KB; also holds the 4096 x 16-bit fast-path histogram
FAST_MAX = 8192  # 32 KB resident sVal buffer in shared memory
HIST12_BINS = 4096
HIST12_WORDS = HIST12_BINS // 2  # 2048 u32s, 2 packed 16-bit counters per word
TIE_CAP = 256  # max pivot-bin elements ranked in registers/SMEM before falling back to 4-pass radix
# A row whose fullest bucket holds more keys falls back to the sample sort (each key's place costs a pass over
# its bucket).
BUCKET_LIMIT = 48
RECOVERY_ROWS = 64
HEALTH_WORDS = 8
H_OVERFLOW, H_UNDERFILL, H_CAND_MAX, H_CAND_SUM, H_BAD = 0, 1, 3, 4, 7
_NEG_INF_BITS = 0xFF800000
_UNROLL = 4
# What a CTA selects from: a row's candidate buffer (the scan's -score, kv index pairs), a slice of a row of dense
# scores (recovery; the index is the column), or the per-slice selections of a row (recovery's second stage).
CANDIDATES, DENSE, STAGED = 0, 1, 2


@cute.jit
def rank_key(v) -> cutlass.Uint32:
    """Ascending float bits with -0.0 tied to +0.0 (matching order_key and canonical_topk)."""
    e = ascending_key(v)
    if (f32_bits(v) & cutlass.Uint32(0x7FFFFFFF)) == cutlass.Uint32(0):
        e = cutlass.Uint32(0x80000000)
    return e


@cute.jit
def order_key(v, index) -> cutlass.Uint64:
    """Ascending order of this key is canonical order of (-score v, index >= 0): v ascending with -0.0 tying +0.0,
    then the index ascending. The lowest bit keeps a zero's sign, so the key alone gives back v's bits."""
    bits = f32_bits(v)
    e = ascending_key(v)
    negative_zero = cutlass.Uint32(0)
    if (bits & cutlass.Uint32(0x7FFFFFFF)) == cutlass.Uint32(0):
        e = cutlass.Uint32(0x80000000)
        negative_zero = bits >> cutlass.Uint32(31)
    low = (cutlass.Uint32(index) << cutlass.Uint32(1)) | negative_zero
    return (cutlass.Uint64(e) << cutlass.Uint64(32)) | cutlass.Uint64(low)


@cute.jit
def order_index(key) -> Int32:
    return Int32(cutlass.Uint32(key & cutlass.Uint64(0xFFFFFFFF)) >> cutlass.Uint32(1))


@cute.jit
def order_score_bits(key) -> cutlass.Uint32:
    """The score (-v) whose order_key this is."""
    e = cutlass.Uint32(key >> cutlass.Uint64(32))
    v = e ^ cutlass.Uint32(0xFFFFFFFF)
    if (e & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
        v = e ^ cutlass.Uint32(0x80000000)
    if (cutlass.Uint32(key & cutlass.Uint64(1))) != cutlass.Uint32(0):
        v = cutlass.Uint32(0x80000000)
    return v ^ cutlass.Uint32(0x80000000)


@cute.jit
def load_value(mVal, row, col, source: cutlass.Constexpr):
    """The entry as -score, the order the selection ranks in."""
    v = mVal[(row, col)]
    if cutlass.const_expr(source != CANDIDATES):
        v = -v
    return v


@cute.jit
def load_index(mIdx, row, col, source: cutlass.Constexpr) -> Int32:
    index = col
    if cutlass.const_expr(source != DENSE):
        index = mIdx[(row, col)]
    return index


@cute.jit
def order_high_score(high) -> Float32:
    """The score whose order_key has this high word (a zero comes back as -0.0)."""
    v = high ^ cutlass.Uint32(0xFFFFFFFF)
    if (high & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
        v = high ^ cutlass.Uint32(0x80000000)
    return bits_f32(v ^ cutlass.Uint32(0x80000000))


@cute.jit
def fast_bin12(score, worst, inv_span) -> Int32:
    """Monotone linear 12-bit bucket [0, 4095] of finite score >= worst (higher score -> higher bucket)."""
    b = f32_to_i32_rz(mul_rn(sub_rn(score, worst), inv_span))
    return min(max(b, Int32(0)), Int32(HIST12_BINS - 1))


@cute.jit
def bitonic_sort_asc(sA, tid):
    """Sort sA[0:THREADS] ascending, one element a thread; every thread must reach this."""
    for size_log in cutlass.range_constexpr(1, THREADS.bit_length()):
        size = 1 << size_log
        for j in cutlass.range_constexpr(size_log):
            stride = 1 << (size_log - 1 - j)
            partner = tid ^ stride
            if partner > tid:
                a = sA[tid]
                b = sA[partner]
                if (a > b) == ((tid & size) == 0):
                    sA[tid] = b
                    sA[partner] = a
            cute.arch.sync_threads()


@cute.jit
def radix_bin(sHist, sScan, sState, tid, kfind):
    """The bin holding the kfind-th smallest key (acc < kfind <= acc + hist[bin]): its owner thread writes (bin, acc,
    count) to sState[0..2]. Every thread must reach this; the caller syncs before reading sState."""
    h = sHist[tid]
    incl, _ = block_inclusive_scan_i32(h, sScan, tid)
    acc = incl - h
    if (acc < kfind) & (kfind <= incl):
        sState[0] = tid
        sState[1] = acc
        sState[2] = h


class Select:
    """source CANDIDATES: one CTA per row of the call, with the health. DENSE: `splits` CTAs per slot, each the top-k
    of its slice of the slot's window of dense scores, written to output row map[slot] when splits is 1 and to its
    own row of the stage buffers otherwise. STAGED: one CTA per slot, over the slot's stage rows. A slot whose map
    entry is negative is skipped."""

    def __init__(self, topk: int, source: int = CANDIDATES):
        self.topk = topk
        self.source = source

    @cute.jit
    def __call__(
        self,
        values: cute.Tensor,  # [rows, cap] f32: -score (CANDIDATES) or scores
        idx: cute.Tensor,  # [rows, cap] i32 (unused for DENSE)
        cand_cnt: cute.Tensor,  # [rows] i32: candidates appended, may exceed cap (CANDIDATES)
        ks: cute.Tensor,  # [rows] window starts
        ke: cute.Tensor,
        row_map: cute.Tensor,  # [slots] i32: output row, or -1 (DENSE, STAGED)
        scores: cute.Tensor,  # [out rows, topk] f32 out
        indices: cute.Tensor,  # [out rows, topk] int32 out
        health: cute.Tensor,  # [HEALTH_WORDS] i64, accumulated (CANDIDATES)
        bad_rows: cute.Tensor,  # [RECOVERY_ROWS] i32 out (CANDIDATES)
        num_ctas: Int32,
        splits: Int32,
        stats: Int32,
        stream: cuda.CUstream,
    ):
        min_blocks = 4 if self.topk <= 2048 else 2
        self.kernel(values, idx, cand_cnt, ks, ke, row_map, scores, indices, health, bad_rows, splits, stats).launch(
            grid=(num_ctas, 1, 1), block=[THREADS, 1, 1], min_blocks_per_mp=min_blocks, stream=stream
        )

    @cute.kernel
    def kernel(self, mVal, mIdx, mCnt, mKs, mKe, mMap, mScores, mIndices, mHealth, mBad, splits: Int32, stats: Int32):
        K = self.topk
        src = self.source
        items = (K + THREADS - 1) // THREADS
        split_u64 = (K * 3 + 3) // 4 * 2  # in Uint64 units (>= K + K/2, 16-byte aligned)
        arena_u64 = max(FAST_MAX // 2, split_u64 + THREADS)
        place_off = K * 2  # Int32 offset inside sArena (right after sKey[0..K-1])
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = tid % 32
        lt_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)

        smem = utils.SmemAllocator()
        # Main reusable SMEM arena (~32 KB at K=2048): sVal[8192 f32] in Phase 1, then sKey/sPlace/sSplit in Phase 2.
        sArena = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((arena_u64,)), 16)
        sVal = cute.recast_tensor(sArena, Float32)
        sKey = sArena
        sPlace = cute.recast_tensor(sArena, Int32)
        sSplit = cute.make_tensor(sArena.iterator + split_u64, cute.make_layout((THREADS,)))
        # 2049 u32s (~8 KB): 4096 x 16-bit histogram / bucket offsets in fast path; sCount & sHist in fallback.
        sCount = smem.allocate_tensor(Int32, cute.make_layout((SORT_BUCKETS + 1,)), 16)
        sIds = smem.allocate_tensor(Int32, cute.make_layout((K,)), 16)  # packed (bin << 13) | pos
        sTie = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((TIE_CAP,)), 16)
        sScan = smem.allocate_tensor(Int32, cute.make_layout((WARPS,)), 16)
        sScanF = smem.allocate_tensor(Float32, cute.make_layout((WARPS,)), 16)
        # 0..2 pivot / radix_bin, 3 write counter, 4 / 5 selected keys' smallest / largest high word
        sState = smem.allocate_tensor(Int32, cute.make_layout((8,)), 16)
        sVal_addr = sArena.iterator.toint()
        hist = sCount.iterator.toint()
        state = sState.iterator.toint()

        cap = mVal.layout.shape[1]
        row = bid
        out_row = bid
        lo = Int32(0)
        n = cap
        raw_n = Int32(0)
        target = Int32(0)
        active = True
        if cutlass.const_expr(src == CANDIDATES):
            raw_n = mCnt[row]
            n = min(max(raw_n, Int32(0)), cap)
        elif cutlass.const_expr(src == DENSE):
            row = bid // splits
            part = bid - row * splits
            target = mMap[row]
            active = target >= 0
            out_row = select_i32(splits > 1, bid, target)
            start = max(Int32(0), min(cap, mKs[row]))
            end = max(start, min(cap, mKe[row]))
            chunk = (end - start + splits - 1) // splits
            lo = min(end, start + part * chunk)
            n = min(end, lo + chunk) - lo
        else:
            target = mMap[row]
            active = target >= 0
            out_row = target

        if active:
            # ----------------------------------------------------------------------------------------------------
            # Fast path (n <= FAST_MAX = 8192): 1 coalesced 16B load into SMEM + 1-pass 12-bit (4096-bin) select.
            # ----------------------------------------------------------------------------------------------------
            need_fallback = n > Int32(FAST_MAX)
            done_sorted = cutlass.Boolean(False)
            valid = Int32(0)
            written = Int32(0)
            key_lo = cutlass.Uint32(0xFFFFFFFF)
            key_hi = cutlass.Uint32(0)

            if not need_fallback:
                for u in cutlass.range_constexpr(HIST12_WORDS // THREADS):
                    sCount[tid + u * THREADS] = Int32(0)
                row_addr = mVal.iterator.toint() + (
                    cutlass.Int64(row) * cutlass.Int64(mVal.layout.stride[0]) + cutlass.Int64(lo)
                ) * cutlass.Int64(4)
                aligned16 = (row_addr & cutlass.Int64(15)) == cutlass.Int64(0)
                fin = Int32(0)
                lo_s = Float32(float("inf"))
                hi_s = Float32(float("-inf"))
                for base in cutlass.range(0, n, THREADS * 4, unroll=1):
                    pos0 = base + tid * 4
                    if aligned16 & (pos0 + Int32(4) <= n):
                        w0, w1, w2, w3 = load_global_v4_b32(row_addr + cutlass.Int64(pos0) * cutlass.Int64(4))
                        v0 = bits_f32(cutlass.Uint32(w0))
                        v1 = bits_f32(cutlass.Uint32(w1))
                        v2 = bits_f32(cutlass.Uint32(w2))
                        v3 = bits_f32(cutlass.Uint32(w3))
                        if cutlass.const_expr(src != CANDIDATES):
                            v0, v1, v2, v3 = -v0, -v1, -v2, -v3
                        store_shared_v4_b32(
                            sVal_addr + pos0 * 4,
                            Int32(f32_bits(v0)),
                            Int32(f32_bits(v1)),
                            Int32(f32_bits(v2)),
                            Int32(f32_bits(v3)),
                        )
                        for v in (v0, v1, v2, v3):
                            if finite(v):
                                fin = fin + 1
                                s = -v
                                lo_s = fmin(lo_s, s)
                                hi_s = fmax(hi_s, s)
                    elif pos0 < n:
                        for u in cutlass.range_constexpr(4):
                            pos = pos0 + u
                            if pos < n:
                                v = load_value(mVal, row, lo + pos, src)
                                sVal[pos] = v
                                if finite(v):
                                    fin = fin + 1
                                    s = -v
                                    lo_s = fmin(lo_s, s)
                                    hi_s = fmax(hi_s, s)
                total = block_sum_i32(fin, sScan, tid)
                worst = block_min_f32(lo_s, sScanF, tid)
                best = block_max_f32(hi_s, sScanF, tid)

                if total <= K:
                    # Every finite element wins: collect positions from sVal, then gather indices once.
                    if tid == 0:
                        sState[3] = Int32(0)
                    cute.arch.sync_threads()
                    for base in cutlass.range(0, n, THREADS * 4, unroll=1):
                        pos0 = base + tid * 4
                        v0 = Float32(float("nan"))
                        v1 = Float32(float("nan"))
                        v2 = Float32(float("nan"))
                        v3 = Float32(float("nan"))
                        if pos0 < n:
                            w0, w1, w2, w3 = load_shared_v4_b32(sVal_addr + pos0 * 4)
                            v0 = bits_f32(cutlass.Uint32(w0))
                            v1 = bits_f32(cutlass.Uint32(w1))
                            v2 = bits_f32(cutlass.Uint32(w2))
                            v3 = bits_f32(cutlass.Uint32(w3))
                        vs = (v0, v1, v2, v3)
                        takes = []
                        ballots = []
                        reserve = Int32(0)
                        for u in cutlass.range_constexpr(4):
                            take = (pos0 + u < n) & finite(vs[u])
                            ballot = cutlass.Uint32(cute.arch.vote_ballot_sync(take))
                            takes.append(take)
                            ballots.append(ballot)
                            reserve = reserve + Int32(cute.arch.popc(ballot))
                        first = Int32(0)
                        if (lane == 0) & (reserve > 0):
                            first = atomic_add_shared_u32(state + 3 * 4, reserve)
                        first = cute.arch.shuffle_sync(first, 0)
                        for u in cutlass.range_constexpr(4):
                            slot = first + Int32(cute.arch.popc(ballots[u] & lt_mask))
                            if takes[u] & (slot < K):
                                sIds[slot] = pos0 + u
                            first = first + Int32(cute.arch.popc(ballots[u]))
                    cute.arch.sync_threads()
                    written = min(sState[3], Int32(K))
                    my_keys = []
                    for i in cutlass.range_constexpr(items):
                        slot = tid + i * THREADS
                        key_i = cutlass.Uint64(0)
                        if slot < written:
                            pos_i = sIds[slot]
                            v_i = sVal[pos_i]
                            idx_i = load_index(mIdx, row, lo + pos_i, src)
                            key_i = order_key(v_i, idx_i)
                            high = cutlass.Uint32(key_i >> cutlass.Uint64(32))
                            key_lo = min(key_lo, high)
                            key_hi = max(key_hi, high)
                            if idx_i >= 0:
                                valid = valid + 1
                        my_keys.append(key_i)
                    if tid == 0:
                        sState[4] = Int32(-1)
                        sState[5] = Int32(0)
                    cute.arch.sync_threads()
                    for i in cutlass.range_constexpr(items):
                        slot = tid + i * THREADS
                        if slot < written:
                            sKey[slot] = my_keys[i]
                    atomic_min_shared_u32(state + 4 * 4, key_lo)
                    atomic_max_shared_u32(state + 5 * 4, key_hi)
                    valid = block_sum_i32(valid, sScan, tid)
                else:
                    span = sub_rn(best, worst)
                    inv_span = div_rn(Float32(HIST12_BINS - 1), span)
                    can_hist = finite(span) & (span > Float32(0.0)) & finite(inv_span) & (inv_span > Float32(0.0))
                    if not can_hist:
                        need_fallback = cutlass.Boolean(True)
                    else:
                        # Pass 1: 4096-bin 16-bit packed histogram over SMEM sVal.
                        for base in cutlass.range(0, n, THREADS * 4, unroll=1):
                            pos0 = base + tid * 4
                            if pos0 < n:
                                w0, w1, w2, w3 = load_shared_v4_b32(sVal_addr + pos0 * 4)
                                vs = (
                                    bits_f32(cutlass.Uint32(w0)),
                                    bits_f32(cutlass.Uint32(w1)),
                                    bits_f32(cutlass.Uint32(w2)),
                                    bits_f32(cutlass.Uint32(w3)),
                                )
                                for u in cutlass.range_constexpr(4):
                                    if (pos0 + u < n) & finite(vs[u]):
                                        b = fast_bin12(-vs[u], worst, inv_span)
                                        red_add_shared_u32(
                                            hist + (b >> 1) * 4,
                                            select_i32((b & 1) != 0, Int32(0x10000), Int32(1)),
                                        )
                        cute.arch.sync_threads()

                        # Descending prefix scan over 4096 bins (16 bins = 8 u32 words per thread, 16B vector loads/stores).
                        w_base = Int32(HIST12_WORDS - 8) - tid * 8
                        lo0, lo1, lo2, lo3 = load_shared_v4_b32(hist + w_base * 4)
                        hi0, hi1, hi2, hi3 = load_shared_v4_b32(hist + (w_base + 4) * 4)
                        words_desc = (hi3, hi2, hi1, hi0, lo3, lo2, lo1, lo0)
                        counts_hi = []
                        counts_lo = []
                        mine = Int32(0)
                        for k in cutlass.range_constexpr(8):
                            word = words_desc[k]
                            chi = (word >> 16) & 0xFFFF
                            clo = word & 0xFFFF
                            counts_hi.append(chi)
                            counts_lo.append(clo)
                            mine = mine + chi + clo
                        incl, _ = block_inclusive_scan_i32(mine, sScan, tid)
                        running = incl - mine
                        my_max_above = Int32(0)
                        packed_desc = []
                        for k in cutlass.range_constexpr(8):
                            w_idx = w_base + (7 - k)
                            b_hi = w_idx * 2 + 1
                            b_lo = w_idx * 2
                            chi = counts_hi[k]
                            clo = counts_lo[k]
                            off_hi = running
                            if (running < K) & (K <= running + chi):
                                sState[0] = b_hi
                                sState[1] = running
                                sState[2] = chi
                            if running + chi < K:
                                my_max_above = max(my_max_above, chi)
                            running = running + chi
                            off_lo = running
                            if (running < K) & (K <= running + clo):
                                sState[0] = b_lo
                                sState[1] = running
                                sState[2] = clo
                            if running + clo < K:
                                my_max_above = max(my_max_above, clo)
                            running = running + clo
                            packed_desc.append(((off_hi & 0xFFFF) << 16) | (off_lo & 0xFFFF))
                        store_shared_v4_b32(hist + w_base * 4, packed_desc[7], packed_desc[6], packed_desc[5], packed_desc[4])
                        store_shared_v4_b32(hist + (w_base + 4) * 4, packed_desc[3], packed_desc[2], packed_desc[1], packed_desc[0])
                        max_above = block_max_i32(my_max_above, sScan, tid)
                        pivot_bin = sState[0]
                        count_above = sState[1]
                        pivot_cnt = sState[2]

                        if pivot_cnt > Int32(TIE_CAP):
                            need_fallback = cutlass.Boolean(True)
                        else:
                            # Pass 2: warp-ballot sequential emit into sIds (keeps pos nearly sorted for coalesced mIdx gather).
                            if tid == 0:
                                sState[3] = Int32(0)  # packed (tie_cnt << 16) | above_cnt
                            cute.arch.sync_threads()
                            for base in cutlass.range(0, n, THREADS * 4, unroll=1):
                                pos0 = base + tid * 4
                                v0 = Float32(float("nan"))
                                v1 = Float32(float("nan"))
                                v2 = Float32(float("nan"))
                                v3 = Float32(float("nan"))
                                if pos0 < n:
                                    w0, w1, w2, w3 = load_shared_v4_b32(sVal_addr + pos0 * 4)
                                    v0 = bits_f32(cutlass.Uint32(w0))
                                    v1 = bits_f32(cutlass.Uint32(w1))
                                    v2 = bits_f32(cutlass.Uint32(w2))
                                    v3 = bits_f32(cutlass.Uint32(w3))
                                vs = (v0, v1, v2, v3)
                                bs = []
                                gt_ballots = []
                                eq_ballots = []
                                gt_takes = []
                                eq_takes = []
                                gt_tot = Int32(0)
                                eq_tot = Int32(0)
                                for u in cutlass.range_constexpr(4):
                                    pos = pos0 + u
                                    ok = (pos < n) & finite(vs[u])
                                    bu = select_i32(ok, fast_bin12(-vs[u], worst, inv_span), Int32(-1))
                                    is_gt = ok & (bu > pivot_bin)
                                    is_eq = ok & (bu == pivot_bin)
                                    gb = cutlass.Uint32(cute.arch.vote_ballot_sync(is_gt))
                                    eb = cutlass.Uint32(cute.arch.vote_ballot_sync(is_eq))
                                    bs.append(bu)
                                    gt_takes.append(is_gt)
                                    eq_takes.append(is_eq)
                                    gt_ballots.append(gb)
                                    eq_ballots.append(eb)
                                    gt_tot = gt_tot + Int32(cute.arch.popc(gb))
                                    eq_tot = eq_tot + Int32(cute.arch.popc(eb))
                                packed_res = Int32(0)
                                if (lane == 0) & ((gt_tot > 0) | (eq_tot > 0)):
                                    packed_res = atomic_add_shared_u32(state + 3 * 4, (eq_tot << 16) | gt_tot)
                                packed_res = cute.arch.shuffle_sync(packed_res, 0)
                                gt_first = packed_res & 0xFFFF
                                eq_first = (packed_res >> 16) & 0xFFFF
                                for u in cutlass.range_constexpr(4):
                                    pos = pos0 + u
                                    if gt_takes[u]:
                                        sIds[gt_first + Int32(cute.arch.popc(gt_ballots[u] & lt_mask))] = (bs[u] << 13) | pos
                                    if eq_takes[u]:
                                        idx_u = load_index(mIdx, row, lo + pos, src)
                                        sTie[eq_first + Int32(cute.arch.popc(eq_ballots[u] & lt_mask))] = order_key(vs[u], idx_u)
                                    gt_first = gt_first + Int32(cute.arch.popc(gt_ballots[u]))
                                    eq_first = eq_first + Int32(cute.arch.popc(eq_ballots[u]))
                            cute.arch.sync_threads()

                            # Coalesced sparse index gather + direct bucket-slot reservation (when max_above <= BUCKET_LIMIT).
                            quota = Int32(K) - count_above
                            direct_sort = max_above <= Int32(BUCKET_LIMIT)
                            my_keys = []
                            my_db = []
                            for i in cutlass.range_constexpr(items):
                                slot = tid + i * THREADS
                                key_i = cutlass.Uint64(0)
                                db_i = Int32(K)
                                if slot < count_above:
                                    packed = sIds[slot]
                                    b_i = packed >> 13
                                    pos_i = packed & 0x1FFF
                                    v_i = sVal[pos_i]
                                    idx_i = load_index(mIdx, row, lo + pos_i, src)
                                    key_i = order_key(v_i, idx_i)
                                    if idx_i >= 0:
                                        valid = valid + 1
                                    dst_i = slot
                                    if direct_sort:
                                        old = atomic_add_shared_u32(
                                            hist + (b_i >> 1) * 4,
                                            select_i32((b_i & 1) != 0, Int32(0x10000), Int32(1)),
                                        )
                                        dst_i = select_i32((b_i & 1) != 0, (old >> 16) & 0xFFFF, old & 0xFFFF)
                                    db_i = (b_i << 13) | dst_i
                                my_keys.append(key_i)
                                my_db.append(db_i)

                            my_tie = cutlass.Uint64(0)
                            tie_rank = Int32(TIE_CAP)
                            if tid < pivot_cnt:
                                my_tie = sTie[tid]
                                tr = Int32(0)
                                for q in cutlass.range(pivot_cnt, unroll=1):
                                    if sTie[q] < my_tie:
                                        tr = tr + 1
                                tie_rank = tr
                                if tr < quota:
                                    if order_index(my_tie) >= 0:
                                        valid = valid + 1

                            # sVal is now dead: switch arena to sKey + sPlace and scatter bucket-grouped winners.
                            cute.arch.sync_threads()
                            for i in cutlass.range_constexpr(items):
                                dst_i = my_db[i] & 0x1FFF
                                if dst_i < count_above:
                                    sKey[dst_i] = my_keys[i]
                                    if direct_sort:
                                        sPlace[place_off + dst_i] = my_db[i] >> 13
                            if (tid < pivot_cnt) & (tie_rank < quota):
                                sKey[count_above + tie_rank] = my_tie
                            valid = block_sum_i32(valid, sScan, tid)
                            written = Int32(K)

                            if direct_sort:
                                # Canonical intra-bucket rank directly from sCount's post-scatter bucket bounds.
                                sorted_keys = []
                                sorted_places = []
                                for i in cutlass.range_constexpr(items):
                                    slot = tid + i * THREADS
                                    at = min(slot, Int32(K - 1))
                                    sk_i = sKey[at]
                                    place_i = Int32(K)
                                    if slot < count_above:
                                        b_i = sPlace[place_off + at]
                                        w_cur = sCount[b_i >> 1]
                                        last = select_i32((b_i & 1) != 0, (w_cur >> 16) & 0xFFFF, w_cur & 0xFFFF)
                                        first = Int32(0)
                                        if b_i < Int32(HIST12_BINS - 1):
                                            bp = b_i + 1
                                            w_prev = sCount[bp >> 1]
                                            first = select_i32((bp & 1) != 0, (w_prev >> 16) & 0xFFFF, w_prev & 0xFFFF)
                                        place_i = first
                                        if last - first > 1:
                                            for q in cutlass.range(first, last, unroll=1):
                                                if sKey[q] < sk_i:
                                                    place_i = place_i + 1
                                    sorted_keys.append(sk_i)
                                    sorted_places.append(place_i)
                                cute.arch.sync_threads()
                                for i in cutlass.range_constexpr(items):
                                    if sorted_places[i] < count_above:
                                        sKey[sorted_places[i]] = sorted_keys[i]
                                cute.arch.sync_threads()
                                done_sorted = cutlass.Boolean(True)
                            else:
                                # Pathological >48 keys in one high bucket: compute exact min/max high words for sort.
                                kl = cutlass.Uint32(0xFFFFFFFF)
                                kh = cutlass.Uint32(0)
                                for i in cutlass.range_constexpr(items):
                                    slot = tid + i * THREADS
                                    if slot < K:
                                        h_i = cutlass.Uint32(sKey[slot] >> cutlass.Uint64(32))
                                        kl = min(kl, h_i)
                                        kh = max(kh, h_i)
                                if tid == 0:
                                    sState[4] = Int32(-1)
                                    sState[5] = Int32(0)
                                cute.arch.sync_threads()
                                atomic_min_shared_u32(state + 4 * 4, kl)
                                atomic_max_shared_u32(state + 5 * 4, kh)
                                cute.arch.sync_threads()

            # ----------------------------------------------------------------------------------------------------
            # Exact 4-pass radix fallback (n > FAST_MAX, zero/infinite span, or >256 elements in pivot_bin).
            # ----------------------------------------------------------------------------------------------------
            if need_fallback:
                sCount[tid] = Int32(0)
                cute.arch.sync_threads()
                fin = Int32(0)
                for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                    vs = [load_value(mVal, row, lo + min(base + u * THREADS + tid, n - 1), src) for u in range(_UNROLL)]
                    for u in cutlass.range_constexpr(_UNROLL):
                        if (base + u * THREADS + tid < n) & finite(vs[u]):
                            fin = fin + 1
                            red_add_shared_u32(hist + Int32(rank_key(vs[u]) >> cutlass.Uint32(24)) * 4, Int32(1))
                total = block_sum_i32(fin, sScan, tid)
                target = min(total, Int32(K))

                pivot = cutlass.Uint32(0xFFFFFFFF)
                tie_cut = cutlass.Uint32(0xFFFFFFFF)
                if total > K:
                    desired = cutlass.Uint32(0)
                    mask = cutlass.Uint32(0)
                    kfind = target
                    below = Int32(0)
                    equal = Int32(0)
                    for p in cutlass.range_constexpr(4):
                        shift = 24 - 8 * p
                        if p > 0:
                            sCount[tid] = Int32(0)
                            cute.arch.sync_threads()
                            for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                                vs = [load_value(mVal, row, lo + min(base + u * THREADS + tid, n - 1), src)
                                      for u in range(_UNROLL)]
                                for u in cutlass.range_constexpr(_UNROLL):
                                    e = rank_key(vs[u])
                                    if (base + u * THREADS + tid < n) & finite(vs[u]) & ((e & mask) == desired):
                                        b = Int32((e >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF))
                                        red_add_shared_u32(hist + b * 4, Int32(1))
                            cute.arch.sync_threads()
                        radix_bin(sCount, sScan, sState, tid, kfind)
                        cute.arch.sync_threads()
                        desired = desired | (cutlass.Uint32(sState[0]) << cutlass.Uint32(shift))
                        kfind = kfind - sState[1]
                        below = below + sState[1]
                        equal = sState[2]
                        mask = mask | (cutlass.Uint32(0xFF) << cutlass.Uint32(shift))
                        cute.arch.sync_threads()
                    pivot = desired
                    eq_take = target - below
                    if equal > eq_take:
                        idesired = cutlass.Uint32(0)
                        imask = cutlass.Uint32(0)
                        ikfind = eq_take
                        for p in cutlass.range_constexpr(4):
                            shift = 24 - 8 * p
                            sCount[tid] = Int32(0)
                            cute.arch.sync_threads()
                            for j in cutlass.range(tid, n, THREADS, unroll=1):
                                v = load_value(mVal, row, lo + j, src)
                                if finite(v):
                                    if rank_key(v) == pivot:
                                        ik = cutlass.Uint32(load_index(mIdx, row, lo + j, src))
                                        if (ik & imask) == idesired:
                                            b = Int32((ik >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF))
                                            red_add_shared_u32(hist + b * 4, Int32(1))
                            cute.arch.sync_threads()
                            radix_bin(sCount, sScan, sState, tid, ikfind)
                            cute.arch.sync_threads()
                            idesired = idesired | (cutlass.Uint32(sState[0]) << cutlass.Uint32(shift))
                            ikfind = ikfind - sState[1]
                            imask = imask | (cutlass.Uint32(0xFF) << cutlass.Uint32(shift))
                            cute.arch.sync_threads()
                        tie_cut = idesired

                if tid == 0:
                    sState[3] = Int32(0)
                    sState[4] = Int32(-1)
                    sState[5] = Int32(0)
                cute.arch.sync_threads()
                key_lo = cutlass.Uint32(0xFFFFFFFF)
                key_hi = cutlass.Uint32(0)
                valid = Int32(0)
                for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                    vs = [load_value(mVal, row, lo + min(base + u * THREADS + tid, n - 1), src) for u in range(_UNROLL)]
                    xs = [load_index(mIdx, row, lo + min(base + u * THREADS + tid, n - 1), src) for u in range(_UNROLL)]
                    takes = []
                    ballots = []
                    reserve = Int32(0)
                    for u in cutlass.range_constexpr(_UNROLL):
                        e = rank_key(vs[u])
                        take = (base + u * THREADS + tid < n) & finite(vs[u]) & (
                            (e < pivot) | ((e == pivot) & (cutlass.Uint32(xs[u]) <= tie_cut)))
                        ballot = cutlass.Uint32(cute.arch.vote_ballot_sync(take))
                        takes.append(take)
                        ballots.append(ballot)
                        reserve = reserve + Int32(cute.arch.popc(ballot))
                    first = Int32(0)
                    if (lane == 0) & (reserve > 0):
                        first = atomic_add_shared_u32(state + 3 * 4, reserve)
                    first = cute.arch.shuffle_sync(first, 0)
                    for u in cutlass.range_constexpr(_UNROLL):
                        slot = first + Int32(cute.arch.popc(ballots[u] & lt_mask))
                        if takes[u] & (slot < K):
                            key = order_key(vs[u], xs[u])
                            sKey[slot] = key
                            high = cutlass.Uint32(key >> cutlass.Uint64(32))
                            key_lo = min(key_lo, high)
                            key_hi = max(key_hi, high)
                            if xs[u] >= 0:
                                valid = valid + 1
                        first = first + Int32(cute.arch.popc(ballots[u]))
                atomic_min_shared_u32(state + 4 * 4, key_lo)
                atomic_max_shared_u32(state + 5 * 4, key_hi)
                valid = block_sum_i32(valid, sScan, tid)
                written = min(sState[3], Int32(K))

            # ----------------------------------------------------------------------------------------------------
            # Canonical sort (when not already sorted by the 12-bit fast path) + coalesced global writeback.
            # ----------------------------------------------------------------------------------------------------
            if not done_sorted:
                for b in cutlass.range(tid, SORT_BUCKETS + 1, THREADS, unroll=1):
                    sCount[b] = Int32(0)
                cute.arch.sync_threads()
                best = order_high_score(cutlass.Uint32(sState[4]))
                worst = order_high_score(cutlass.Uint32(sState[5]))
                span = sub_rn(best, worst)
                eps = mul_rn(span, Float32(2.0**-13))
                base = f32_bits(eps)
                shift = max(bit_length_u32(f32_bits(add_rn(span, eps)) - base) - 11, Int32(0))
                spread = finite(span) & (span > Float32(0.0))
                for i in cutlass.range_constexpr(items):
                    slot = tid + i * THREADS
                    if slot < written:
                        s_key = order_high_score(cutlass.Uint32(sKey[slot] >> cutlass.Uint64(32)))
                        up = Int32((f32_bits(add_rn(sub_rn(s_key, worst), eps)) - base) >> cutlass.Uint32(shift))
                        d = Int32(LOG_BUCKETS - 1) - min(max(up, Int32(0)), Int32(LOG_BUCKETS - 1))
                        d = select_i32(spread, d, Int32(0))
                        rank = atomic_add_shared_u32(sCount.iterator.toint() + d * 4, Int32(1))
                        sPlace[place_off + slot] = (d << 16) | rank
                cute.arch.sync_threads()
                fullest = Int32(0)
                for u in cutlass.range_constexpr(LOG_BUCKETS // THREADS):
                    fullest = max(fullest, sCount[tid * (LOG_BUCKETS // THREADS) + u])
                fullest = block_max_i32(fullest, sScan, tid)
                if fullest > BUCKET_LIMIT:
                    for b in cutlass.range(tid, SORT_BUCKETS + 1, THREADS, unroll=1):
                        sCount[b] = Int32(0)
                    splitter = cutlass.Uint64(0)
                    if written > 0:
                        splitter = sKey[min((tid * written) // THREADS, Int32(K - 1))]
                    sSplit[tid] = splitter
                    cute.arch.sync_threads()
                    bitonic_sort_asc(sSplit, tid)
                    for i in cutlass.range_constexpr(items):
                        slot = tid + i * THREADS
                        if slot < written:
                            key = sKey[slot]
                            pos = Int32(0)
                            for j in cutlass.range_constexpr(THREADS.bit_length() - 1):
                                half = THREADS >> (j + 1)
                                if sSplit[pos + half - 1] < key:
                                    pos = pos + half
                            if sSplit[pos] < key:
                                pos = pos + 1
                            rank = atomic_add_shared_u32(sCount.iterator.toint() + pos * 4, Int32(1))
                            sPlace[place_off + slot] = (pos << 16) | rank
                    cute.arch.sync_threads()
                per = SORT_BUCKETS // THREADS
                counts = [sCount[tid * per + u] for u in range(per)]
                mine = Int32(0)
                for u in cutlass.range_constexpr(per):
                    mine = mine + counts[u]
                incl, _ = block_inclusive_scan_i32(mine, sScan, tid)
                run = incl - mine
                for u in cutlass.range_constexpr(per):
                    sCount[tid * per + u] = run
                    run = run + counts[u]
                if tid == THREADS - 1:
                    sCount[SORT_BUCKETS] = incl
                cute.arch.sync_threads()

                keys = []
                packed_places = []
                for i in cutlass.range_constexpr(items):
                    slot = tid + i * THREADS
                    at = min(slot, Int32(K - 1))
                    keys.append(sKey[at])
                    packed_places.append(select_i32(slot < written, sPlace[place_off + at], Int32(SORT_BUCKETS << 16)))
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(items):
                    d = packed_places[i] >> 16
                    if d < SORT_BUCKETS:
                        dst = sCount[d] + (packed_places[i] & 0xFFFF)
                        sKey[dst] = keys[i]
                        sPlace[place_off + dst] = d
                cute.arch.sync_threads()

                sorted_keys = []
                sorted_places = []
                for i in cutlass.range_constexpr(items):
                    slot = tid + i * THREADS
                    at = min(slot, Int32(K - 1))
                    key_fb = sKey[at]
                    place_i = Int32(K)
                    if slot < written:
                        d = sPlace[place_off + at]
                        first = sCount[d]
                        last = sCount[d + 1]
                        place_i = first
                        if last - first > 1:
                            for q in cutlass.range(first, last, unroll=1):
                                if sKey[q] < key_fb:
                                    place_i = place_i + 1
                    sorted_keys.append(key_fb)
                    sorted_places.append(place_i)
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(items):
                    if sorted_places[i] < K:
                        sKey[sorted_places[i]] = sorted_keys[i]
                cute.arch.sync_threads()

            for i in cutlass.range_constexpr(items):
                slot = tid + i * THREADS
                if slot < K:
                    if slot < written:
                        key_out = sKey[slot]
                        mScores[(out_row, slot)] = bits_f32(order_score_bits(key_out))
                        mIndices[(out_row, slot)] = order_index(key_out)
                    else:
                        mScores[(out_row, slot)] = bits_f32(cutlass.Uint32(_NEG_INF_BITS))
                        mIndices[(out_row, slot)] = Int32(-1)

            if cutlass.const_expr(src == CANDIDATES):
                if tid == 0:
                    expected = min(max(mKe[row] - mKs[row], Int32(0)), Int32(K))
                    health = mHealth.iterator.toint()
                    overflow = raw_n > cap
                    underfill = valid < expected
                    if overflow:
                        atomic_add_global_u64(health + H_OVERFLOW * 8, cutlass.Int64(1))
                    if underfill:
                        atomic_add_global_u64(health + H_UNDERFILL * 8, cutlass.Int64(1))
                    if overflow | underfill:
                        bad_slot = atomic_add_global_u64(health + H_BAD * 8, cutlass.Int64(1))
                        if bad_slot < RECOVERY_ROWS:
                            mBad[Int32(bad_slot)] = row
                    candidates = cutlass.Int64(max(raw_n, Int32(0)))
                    if (stats != 0) | overflow:
                        atomic_max_global_u64(health + H_CAND_MAX * 8, candidates)
                    if stats != 0:
                        atomic_add_global_u64(health + H_CAND_SUM * 8, candidates)
