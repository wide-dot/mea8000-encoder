"""Frame duration selection: replace runs of 8 ms frames by 16/32/64 ms frames.

The chip interpolates every parameter linearly from the previous frame's values to the new
frame's values over the frame duration (TP101 p. 6). A long frame is therefore a straight
ramp through the slots it covers; the dynamic programme below chooses, per utterance, the
segmentation into 1/2/4/8-slot frames that minimises the deviation of those ramps from the
8 ms code stream plus a price per frame (4 bytes each, whatever its duration).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import tables as T
from .codec import Frame, Utterance

LENGTHS = (1, 2, 4, 8)   # slots per frame for FD codes 0..3
AMPL_FLOOR_DB = 20 * np.log10(8 / 1000.0)


@dataclass
class MergeWeights:
    formant_db_per_octave: float = 6.0
    bandwidth_db_per_octave: float = 3.0
    ampl_db: float = 1.0          # per dB of amplitude deviation
    pitch_db_per_cent: float = 0.03


def _values(frames: list[Frame], pitch: np.ndarray) -> np.ndarray:
    """Per slot: log2 fm1..3, log2 bw1..4, ampl dB, log2 pitch — the quantities the ramps are judged on."""
    out = np.zeros((len(frames), 9))
    for k, f in enumerate(frames):
        out[k, 0:3] = np.log2(f.fm_hz[:3])
        out[k, 3:7] = np.log2(f.bw_hz)
        out[k, 7] = 20 * np.log10(max(f.ampl_permille, 8) / 1000.0) if f.ampl else AMPL_FLOOR_DB - 20
        out[k, 8] = np.log2(max(pitch[k], 1))
    return out


def _can_fill(noise: np.ndarray, k: int, min_slots: int, n: int) -> bool:
    """Whether a frame of at least `min_slots` can start at k: the excitation (noise or
    voiced) must not change before k + min_slots and the file must not end before."""
    return k + min_slots <= n and bool(np.all(noise[k: k + min_slots] == noise[k]))


def merge_utterance(frames: list[Frame], pitch: np.ndarray, start_pitch_hz: int, price_db: float,
                    weights: MergeWeights = MergeWeights(), min_slots: int = 1) -> list[Frame]:
    """DP over frame boundaries. `pitch[k]` is the pitch reached at the end of slot k (Hz).
    `min_slots` is the shortest frame allowed (1, 2, 4 or 8 slots of 8 ms); the last frame
    of the file may be shorter when the slots left do not fill one."""
    n = len(frames)
    if n == 0:
        return []
    v = _values(frames, pitch)
    w = np.array([weights.formant_db_per_octave] * 3 + [weights.bandwidth_db_per_octave] * 4
                 + [weights.ampl_db, weights.pitch_db_per_cent * 1200.0])
    noise = np.array([f.noise for f in frames])

    def ramp_cost(k: int, L: int) -> float:
        end = v[k + L - 1]
        if k == 0:
            # the first frame of an utterance is preset, only the amplitude ramps from zero
            start = end.copy()
            start[7] = AMPL_FLOOR_DB - 20
        else:
            start = v[k - 1]
        steps = np.arange(1, L + 1)[:, None] / L
        ramp = start[None, :] + (end - start)[None, :] * steps
        dev = np.abs(ramp - v[k: k + L]) * w[None, :]
        return float(dev.sum())

    INF = float("inf")
    best = np.full(n + 1, INF)
    back = np.zeros(n + 1, dtype=np.int64)
    best[0] = 0.0
    for k in range(n):
        if best[k] == INF:
            continue
        for L in LENGTHS:
            if k + L > n or not np.all(noise[k: k + L] == noise[k]):
                continue
            if L < min_slots and k + L != n and _can_fill(noise, k, min_slots, n):
                continue                          # too short, unless nothing longer fits here
            c = best[k] + ramp_cost(k, L) + price_db
            if c < best[k + L]:
                best[k + L] = c
                back[k + L] = L

    bounds = []
    k = n
    while k > 0:
        L = int(back[k])
        bounds.append((k - L, L))
        k -= L
    bounds.reverse()

    out = []
    current_pitch = float(start_pitch_hz)
    for k, L in bounds:
        f = frames[k + L - 1]
        fd = LENGTHS.index(L)
        if f.noise:
            pi = T.NOISE_CODE
        else:
            target = float(pitch[k + L - 1])
            step = int(round((target - current_pitch) / L))
            step = max(-15, min(15, step))
            pi = step if step >= 0 else 32 + step
            current_pitch += step * L
        out.append(Frame(bw=f.bw, fm=f.fm, ampl=f.ampl, fd=fd, pi=pi))
    return out


def merge(result, price_db: float, weights: MergeWeights = MergeWeights(), min_slots: int = 1) -> list[Utterance]:
    """Apply `merge_utterance` to every speech file of an `EncodeResult` (one 8 ms frame per slot)."""
    from .encoder import split_utterances

    frames, plan = result.frames, result.pitch
    starts = [k for k in range(len(frames)) if plan.restart[k] or k == 0]
    merged = []
    for a, b in zip(starts, starts[1:] + [len(frames)]):
        merged.append(merge_utterance(frames[a:b], plan.pitch_hz[a:b], 2 * int(plan.start_pitch[a]), price_db, weights, min_slots))
    return split_utterances(frames, plan, per_chunk=merged)


def size_bytes(utterances: list[Utterance]) -> int:
    return sum(4 + 4 * len(u.frames) for u in utterances)
