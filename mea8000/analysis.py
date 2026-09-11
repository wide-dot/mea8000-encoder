"""Target analysis for the encoder: pitch, voicing, energy and spectral envelope per 8 ms.

Everything works at 8 kHz, the rate of the chip's filter bank: nothing above 4 kHz can be
represented, so the input is band-limited first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

RATE = 8000
HOP = 64                 # 8 ms, the shortest frame of the chip
ENV_WIN = 256            # 32 ms analysis window for the envelope
PITCH_WIN = 224          # 28 ms window for the difference function
MAX_CANDIDATES = 6
F0_MIN, F0_MAX = 60.0, 510.0
N_BINS = 129             # envelope grid: 0..4000 Hz in 31.25 Hz steps
ENV_FREQS = np.linspace(0.0, RATE / 2, N_BINS)


@dataclass
class Analysis:
    rate: int
    hop: int
    f0: np.ndarray          # Hz per frame, nan when unvoiced
    voiced: np.ndarray      # bool per frame
    periodicity: np.ndarray  # 1 - min(cmndf), per frame
    energy_db: np.ndarray   # RMS of the frame, dB re full scale
    envelope_db: np.ndarray  # (frames, N_BINS) smoothed log magnitude
    lpc: np.ndarray         # (frames, order + 1) predictor coefficients
    hnr_db: np.ndarray | None = None  # harmonic-to-noise ratio at the tracked pitch

    @property
    def n_frames(self) -> int:
        return len(self.f0)

    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.hop / self.rate


def to_8k(x: np.ndarray, rate: float) -> np.ndarray:
    """Mono float signal at 8 kHz; `rate` may be fractional (clock compensation)."""
    x = np.asarray(x, dtype=np.float64)
    if x.dtype.kind in "iu" or np.abs(x).max() > 1.5:
        x = x / 32768.0
    rate = int(round(rate))
    if rate == RATE:
        return x
    from math import gcd

    g = gcd(rate, RATE)
    return resample_poly(x, RATE // g, rate // g)


def n_frames_for(n_samples: int, hop: int = HOP) -> int:
    return max(1, int(np.ceil(n_samples / hop)))


def _frame(x: np.ndarray, start: int, win: int) -> np.ndarray:
    """Window of `win` samples centred on the frame that starts at `start` (zero padded)."""
    c = start + HOP // 2
    a = c - win // 2
    seg = np.zeros(win)
    lo, hi = max(a, 0), min(a + win, len(x))
    if hi > lo:
        seg[lo - a: hi - a] = x[lo:hi]
    return seg


# ------------------------------------------------------------------ envelope (Burg LPC)

def burg(x: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Burg's method. Returns a[0..order] with a[0] = 1 and the prediction error power."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = float(np.dot(x, x)) / n
    f = x[1:].copy()   # forward prediction errors
    b = x[:-1].copy()  # backward prediction errors
    for m in range(1, order + 1):
        den = float(np.dot(f, f) + np.dot(b, b))
        k = -2.0 * float(np.dot(f, b)) / den if den > 0 else 0.0
        a_prev = a.copy()
        for i in range(1, m):
            a[i] = a_prev[i] + k * a_prev[m - i]
        a[m] = k
        f, b = f[1:] + k * b[1:], b[:-1] + k * f[:-1]
        err *= (1.0 - k * k)
    return a, err


def lpc_envelope_db(a: np.ndarray, gain: float, freqs: np.ndarray = ENV_FREQS) -> np.ndarray:
    w = 2 * np.pi * freqs / RATE
    k = np.arange(len(a))
    denom = np.abs(np.exp(-1j * np.outer(w, k)) @ a)
    return 20 * np.log10(gain / np.maximum(denom, 1e-9) + 1e-12)


# ------------------------------------------------------------------ pitch (YIN)

def cmndf(seg: np.ndarray, lag_max: int) -> np.ndarray:
    """Cumulative mean normalised difference function of YIN for lags 0..lag_max."""
    n = len(seg) - lag_max
    x = seg[:n]
    d = np.empty(lag_max + 1)
    e0 = float(np.dot(x, x))
    # d(tau) = sum (x[j] - x[j+tau])^2 over j < n
    csum = np.concatenate([[0.0], np.cumsum(seg * seg)])
    for tau in range(lag_max + 1):
        y = seg[tau: tau + n]
        d[tau] = e0 + (csum[tau + n] - csum[tau]) - 2.0 * float(np.dot(x, y))
    out = np.ones(lag_max + 1)
    run = np.cumsum(d[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.where(run > 0, d[1:] * np.arange(1, lag_max + 1) / run, 1.0)
    return out


def _parabolic(y: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    return float(i) if den == 0 else float(i + 0.5 * (a - c) / den)


# the subharmonic threshold and penalty live in tuning.current (subharmonic_threshold, _penalty)


def _candidates(c: np.ndarray, lag_min: int) -> tuple[np.ndarray, np.ndarray]:
    """Local minima of the CMNDF (refined lag, cost), best first, at most MAX_CANDIDATES.

    YIN's protection against pitch halving: on a periodic signal the CMNDF is as deep at
    two periods as at one, so the shortest lag whose dip is under SUBHARMONIC_THRESHOLD
    is taken as the period and every longer dip pays SUBHARMONIC_PENALTY per octave.
    Without it the tracker halves the pitch of sung voices and of the chip's own output."""
    idx = [i for i in range(max(lag_min, 1), len(c) - 1) if c[i] <= c[i - 1] and c[i] < c[i + 1]]
    if not idx:
        return np.zeros(0), np.zeros(0)
    # a dip counts as "good" under the absolute threshold, or within 0.1 of the deepest
    # one: on a fading note the dip at one period is shallower than 0.15 while the one at
    # two periods is not, and the pitch would still be halved
    from . import tuning
    K = tuning.current
    cmin = min(c[i] for i in idx)
    good = [i for i in idx if c[i] < max(K.subharmonic_threshold, min(0.4, cmin + 0.1))]
    first = min(good) if good else None
    cost = {i: c[i] + (K.subharmonic_penalty * np.log2(i / first) if first is not None and i > 1.5 * first else 0.0)
            for i in idx}
    idx.sort(key=lambda i: cost[i])
    idx = idx[:MAX_CANDIDATES]
    return np.array([_parabolic(c, i) for i in idx]), np.array([cost[i] for i in idx])


def _viterbi(lags: list[np.ndarray], costs: list[np.ndarray], unvoiced_cost: np.ndarray | float,
             octave_weight: float, switch_cost: float) -> np.ndarray:
    """Pick one candidate (or unvoiced, returned as nan) per frame by dynamic programming.
    `unvoiced_cost` may be one value per frame."""
    n = len(lags)
    uc = np.broadcast_to(np.asarray(unvoiced_cost, dtype=np.float64), (n,))
    best = [None] * n
    back = [None] * n
    prev_score = None
    prev_lags = None
    for k in range(n):
        lk = np.append(lags[k], np.nan)                 # last state = unvoiced
        local = np.append(costs[k], uc[k])
        if prev_score is None:
            score = local
            back[k] = np.full(len(lk), -1)
        else:
            m = len(prev_lags)
            trans = np.full((m, len(lk)), switch_cost)
            pv = ~np.isnan(prev_lags)
            cv = ~np.isnan(lk)
            both = np.outer(pv, cv)
            ratio = np.abs(np.log2(np.outer(prev_lags, 1.0 / lk)))
            trans[both] = octave_weight * ratio[both]
            trans[np.outer(~pv, ~cv)] = 0.0
            total = prev_score[:, None] + trans
            back[k] = np.argmin(total, axis=0)
            score = total[back[k], np.arange(len(lk))] + local
        best[k] = score
        prev_score, prev_lags = score, lk
    path = np.full(n, np.nan)
    j = int(np.argmin(best[-1]))
    for k in range(n - 1, -1, -1):
        path[k] = lags[k][j] if j < len(lags[k]) else np.nan
        j = int(back[k][j])
    return path


def analyze(x: np.ndarray, rate: int, order: int = 12, unvoiced_cost: float | None = None,
            switch_cost: float | None = None, min_run: int | None = None, silence_db: float = -60.0,
            hnr_unvoiced: float | None = None, hnr_voiced: float = float("inf"),
            loud_range_db: float = 20.0, loud_unvoiced_bonus: float | None = None) -> Analysis:
    from . import tuning
    K = tuning.current
    unvoiced_cost = K.unvoiced_cost if unvoiced_cost is None else unvoiced_cost
    switch_cost = K.switch_cost if switch_cost is None else switch_cost
    min_run = K.min_run if min_run is None else min_run
    hnr_unvoiced = K.hnr_unvoiced if hnr_unvoiced is None else hnr_unvoiced
    loud_unvoiced_bonus = K.loud_unvoiced_bonus if loud_unvoiced_bonus is None else loud_unvoiced_bonus
    # hnr_voiced (promotion of unvoiced slots) is disabled by default: on the chip's own
    # output, noise through narrow resonators shows a median HNR of 13 dB against 17 dB for
    # voiced slots, so promotion creates far more false voicing than it repairs
    x8 = to_8k(x, rate)
    n = n_frames_for(len(x8))
    lag_min = int(np.floor(RATE / F0_MAX))
    lag_max = int(np.ceil(RATE / F0_MIN))

    periodicity = np.zeros(n)
    energy = np.full(n, silence_db)
    env = np.zeros((n, N_BINS))
    lpcs = np.zeros((n, order + 1))
    hann_env = np.hanning(ENV_WIN)
    cand_lags: list[np.ndarray] = []
    cand_costs: list[np.ndarray] = []

    for k in range(n):
        start = k * HOP
        seg = x8[start: start + HOP]
        if len(seg):
            r = float(np.sqrt(np.mean(seg * seg)))
            energy[k] = 20 * np.log10(r) if r > 0 else silence_db

        # envelope
        w = _frame(x8, start, ENV_WIN) * hann_env
        if np.dot(w, w) > 1e-12:
            a, err = burg(w, order)
            lpcs[k] = a
            env[k] = lpc_envelope_db(a, np.sqrt(max(err, 1e-20)))
        else:
            env[k] = -120.0

        # pitch candidates
        p = _frame(x8, start, PITCH_WIN + lag_max)
        if np.dot(p, p) < 1e-10 or energy[k] <= silence_db + 10:
            cand_lags.append(np.zeros(0))
            cand_costs.append(np.zeros(0))
            continue
        c = cmndf(p, lag_max)
        lags, costs = _candidates(c, lag_min)
        periodicity[k] = 1.0 - float(costs[0]) if len(costs) else 0.0
        cand_lags.append(lags)
        cand_costs.append(costs)

    # loud frames are almost always vowels: the unvoiced state costs more near the peak level
    # (reverberation and background lower the measured periodicity of real recordings)
    peak = float(np.max(energy))
    loudness = np.clip((energy - (peak - loud_range_db)) / loud_range_db, 0.0, 1.0)
    uc = unvoiced_cost + loud_unvoiced_bonus * loudness
    lag_path = _viterbi(cand_lags, cand_costs, uc, octave_weight=K.octave_weight, switch_cost=switch_cost)
    f0 = RATE / lag_path
    voiced = ~np.isnan(f0)

    # harmonic-to-noise ratio at the tracked pitch (or the best candidate) decides the
    # doubtful slots: periodic energy through narrow resonators is not voicing, and weak
    # voiced slots at onsets still show harmonics
    hnr = np.full(n, np.nan)
    for k in range(n):
        if energy[k] <= silence_db + 10:
            continue
        if voiced[k]:
            fk = f0[k]
        elif len(cand_lags[k]):
            fk = RATE / cand_lags[k][0]
        else:
            continue
        hnr[k] = harmonic_to_noise_db(x8, k * HOP, fk)
    with np.errstate(invalid="ignore"):
        voiced = np.where(hnr < hnr_unvoiced, False, voiced)
        promote = (~voiced) & (hnr > hnr_voiced)
    for k in np.where(promote)[0]:
        f0[k] = RATE / cand_lags[k][0]
    voiced = voiced | promote
    voiced = _absorb_short_runs(voiced, min_run)
    # slots voiced by absorption take the pitch of their nearest voiced neighbour
    idx = np.where(~np.isnan(f0))[0]
    if len(idx):
        for k in np.where(voiced & np.isnan(f0))[0]:
            f0[k] = f0[idx[np.argmin(np.abs(idx - k))]]
    f0[~voiced] = np.nan
    return Analysis(RATE, HOP, f0, voiced, periodicity, energy, env, lpcs, hnr)


def harmonic_to_noise_db(x8: np.ndarray, start: int, f0_hz: float, fmax: float = 2500.0) -> float:
    """Mean level difference (dB) between the harmonic peaks of f0 and the valleys between
    them, over the harmonics below `fmax`, on a window of three periods."""
    if not np.isfinite(f0_hz) or f0_hz <= 0:
        return float("nan")
    win = max(256, int(round(3 * RATE / f0_hz)))
    win += win % 2
    seg = _frame(x8, start, win) * np.hanning(win)
    nfft = 4096
    spec = 20 * np.log10(np.abs(np.fft.rfft(seg, nfft)) + 1e-9)
    grid = np.fft.rfftfreq(nfft, 1.0 / RATE)
    diffs = []
    k = 1
    while (k + 0.5) * f0_hz < fmax:
        fc = k * f0_hz
        lo, hi = np.searchsorted(grid, fc - 0.25 * f0_hz), np.searchsorted(grid, fc + 0.25 * f0_hz)
        vlo, vhi = np.searchsorted(grid, fc + 0.3 * f0_hz), np.searchsorted(grid, fc + 0.7 * f0_hz)
        if hi > lo and vhi > vlo:
            diffs.append(spec[lo:hi].max() - spec[vlo:vhi].min())
        k += 1
    return float(np.mean(diffs)) if diffs else float("nan")


def _absorb_short_runs(flags: np.ndarray, min_run: int) -> np.ndarray:
    """Flip runs shorter than `min_run` that sit between two runs of the other value."""
    out = flags.copy()
    n = len(out)
    k = 0
    while k < n:
        j = k
        while j < n and out[j] == out[k]:
            j += 1
        if 0 < k and j < n and j - k < min_run:
            out[k:j] = not out[k]
        k = j
    return out
