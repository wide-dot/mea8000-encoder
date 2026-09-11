"""Audio to MEA8000 frames, version 1: one 8 ms frame per analysis slot.

Pipeline:
  1. `analysis.analyze`   — pitch/voicing (YIN + Viterbi), energy per 8 ms
  2. `fit_pitch`          — dynamic programming on the chip's pitch grid (PI codes,
                            optional restarts with a new starting pitch)
  3. `fit_filters`        — per slot, the (fm, bw) codes whose cascade response is closest
                            to the measured spectrum: harmonic amplitudes for voiced slots,
                            smoothed periodogram for unvoiced ones; exhaustive search on
                            pairs of generators, gain left free
  4. `fit_amplitude`      — the amplitude code whose steady-state synthesis matches the
                            slot energy
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

from . import tuning
from . import tables as T
from .analysis import HOP, RATE, Analysis, analyze, to_8k, _frame
from .codec import Frame, Utterance

QUANT = 512
FM_TABLES = (T.FM1_HZ, T.FM2_HZ, T.FM3_HZ, (T.FM4_HZ,))
F_LOW, F_HIGH = 100.0, 3900.0


# ------------------------------------------------------------------ spectral model

def resonator_db(fm_hz: np.ndarray, bw_hz: np.ndarray, freqs: np.ndarray, f0: int = RATE) -> np.ndarray:
    """Log magnitude of chip resonators (TP101 fig. 8): shape (len(fm), len(bw), len(freqs))."""
    r = np.exp(-np.pi * np.asarray(bw_hz, dtype=np.float64) / f0)[None, :, None]
    theta = (2 * np.pi * np.asarray(fm_hz, dtype=np.float64) / f0)[:, None, None]
    z1 = np.exp(-1j * 2 * np.pi * np.asarray(freqs, dtype=np.float64) / f0)[None, None, :]
    h = 1.0 / (1.0 - 2 * r * np.cos(theta) * z1 + r * r * z1 * z1)
    return 20 * np.log10(np.abs(h) + 1e-12)


def responses_at(freqs: np.ndarray) -> list[np.ndarray]:
    """Per generator, (n_fm, n_bw, n_freqs) resonator responses in dB at the given frequencies."""
    return [resonator_db(np.asarray(tab), np.asarray(T.BW_HZ), freqs) for tab in FM_TABLES]


def interpolator_db(freqs: np.ndarray) -> np.ndarray:
    """Roll-off of the chip's 8 kHz -> 64 kHz linear interpolator, (sin x / x)^2 (TP101 table 6)."""
    s = np.sinc(np.asarray(freqs, dtype=np.float64) / RATE)
    return 20 * np.log10(s * s + 1e-12)


def source_db(f0_hz: float | None, freqs: np.ndarray) -> np.ndarray:
    """Spectral envelope of everything but the resonators: sawtooth harmonics fall 6 dB/octave
    (noise is flat), and the output interpolator rolls off towards 4 kHz."""
    out = interpolator_db(freqs)
    if f0_hz is None or not np.isfinite(f0_hz) or f0_hz <= 0:
        return out
    f = np.maximum(freqs, f0_hz)
    return out - 20 * np.log10(f / f0_hz)


# ------------------------------------------------------------------ targets

class NoiseProfile:
    """Background spectrum of a recording by minimum statistics — a low percentile of each
    bin's power over time — and its removal from power spectra (spectral subtraction with a
    -10 dB floor). Recordings with true silence (chip output, clean synthesis) get a zero
    profile; a continuous room or tape floor is captured bin by bin."""

    def __init__(self, x8: np.ndarray, energy_db: np.ndarray, percentile: float = 10.0, win: int = 512) -> None:
        self.nfft = 1024
        self.grid = np.fft.rfftfreq(self.nfft, 1.0 / RATE)
        h = np.hanning(win)
        n = len(energy_db)
        spectra = np.empty((n, len(self.grid)))
        for k in range(n):
            seg = _frame(x8, k * HOP, win) * h
            spectra[k] = np.abs(np.fft.rfft(seg, self.nfft)) ** 2
        self.power = np.percentile(spectra, percentile, axis=0)   # per-bin, `win`-sample Hann window
        self.win = win
        self.floor_db = float(10 * np.log10(self.power.mean() + 1e-20))
        self.energy_db = float(np.percentile(energy_db, percentile))
        # a real floor is stationary: many slots sit at it (a room or tape floor between
        # words); in continuous speech without one the low percentile is just quiet speech
        self.at_floor = float(np.mean(np.abs(energy_db - self.energy_db) < 3.0))
        self.stationary = self.at_floor >= 0.15

    def subtract(self, power: np.ndarray, grid: np.ndarray, win: int) -> np.ndarray:
        # the periodogram of a stationary noise scales with the window energy
        scale = (np.hanning(win) ** 2).sum() / (np.hanning(self.win) ** 2).sum()
        noise = np.interp(grid, self.grid, self.power) * scale
        return np.maximum(power - noise, 0.1 * noise)


def _window_power(x8: np.ndarray, start: int, win: int, nfft: int, profile: NoiseProfile | None) -> tuple[np.ndarray, np.ndarray]:
    seg = _frame(x8, start, win) * np.hanning(win)
    power = np.abs(np.fft.rfft(seg, nfft)) ** 2
    grid = np.fft.rfftfreq(nfft, 1.0 / RATE)
    if profile is not None:
        power = profile.subtract(power, grid, win)
    return grid, power


def harmonic_target(x8: np.ndarray, start: int, f0_hz: float, profile: NoiseProfile | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, dB) of the harmonic peaks around the slot that starts at `start`."""
    win = max(256, int(round(3 * RATE / f0_hz)))
    win += win % 2
    nfft = 4096
    grid, power = _window_power(x8, start, win, nfft, profile)
    spec = np.sqrt(power) + 1e-9
    ks = np.arange(1, int(F_HIGH // f0_hz) + 1)
    freqs, amps = [], []
    for k in ks:
        fc = k * f0_hz
        if fc < F_LOW:
            continue
        lo = np.searchsorted(grid, fc - 0.3 * f0_hz)
        hi = np.searchsorted(grid, fc + 0.3 * f0_hz)
        if hi <= lo:
            continue
        i = lo + int(np.argmax(spec[lo:hi]))
        freqs.append(fc)
        amps.append(20 * np.log10(spec[i]))
    return np.asarray(freqs), np.asarray(amps)


def noise_target(x8: np.ndarray, start: int, profile: NoiseProfile | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, dB) of the smoothed periodogram around the slot, 100 Hz steps."""
    win = 512
    nfft = 1024
    grid, power = _window_power(x8, start, win, nfft, profile)
    power = power + 1e-12
    kernel = np.ones(25) / 25  # ±94 Hz
    smooth = np.convolve(power, kernel, mode="same")
    freqs = np.arange(F_LOW, F_HIGH, 100.0)
    amps = 10 * np.log10(np.interp(freqs, grid, smooth))
    return freqs, amps


# ------------------------------------------------------------------ 2. pitch

@dataclass
class PitchPlan:
    pitch_hz: np.ndarray      # integer pitch reached at the end of each slot
    pi_code: np.ndarray       # PI code per slot (16 = noise)
    restart: np.ndarray       # True where a new starting pitch is sent before the slot
    start_pitch: np.ndarray   # starting pitch byte where restart is True


def fit_pitch(f0: np.ndarray, voiced: np.ndarray, restart_cost: float = 400.0,
              onset_restart_cost: float = 60.0, max_cents: float = 600.0,
              glide: np.ndarray | None = None, force_pitch: np.ndarray | None = None,
              restart_mode: np.ndarray | None = None, restart_cost_slot: np.ndarray | None = None) -> PitchPlan:
    """DP over integer pitch states 0..510 with the chip's ±15 Hz per 8 ms transitions.

    Cost per voiced slot = |cents(pitch, target)| clipped at `max_cents`; unvoiced slots keep
    their pitch (noise frames do not change it). A restart (STOP + new even starting pitch)
    is allowed anywhere at `restart_cost`, and cheaply (`onset_restart_cost`) at a voiced
    onset after an unvoiced slot: there the alternative is a ±15 Hz per 8 ms glide towards
    the new pitch, an audible chirp, whereas a restart only fades the onset frame in.
    `glide` marks silent slots (AMPL = 0) where the pitch may move by ±15 Hz per slot at no
    cost - nobody hears a silent frame - so that a single sequence (restart costs infinite)
    can still reach the pitch of the next word through the pause before it.
    """
    n = len(f0)
    if glide is None:
        glide = np.zeros(n, dtype=bool)
    glide = glide & ~voiced
    # pins: a forced pitch keeps a single admissible state; a restart can be forced (free)
    # or forbidden (infinite) at a slot; the restart cost may vary by slot (regions)
    force_pitch = np.full(n, np.nan) if force_pitch is None else force_pitch
    restart_mode = np.zeros(n, dtype=np.int8) if restart_mode is None else restart_mode
    restart_cost_slot = np.full(n, np.nan) if restart_cost_slot is None else restart_cost_slot
    states = np.arange(0, 511)
    even = (states % 2 == 0)
    inc = np.array([T.PI_HZ[c] for c in range(32) if c != T.NOISE_CODE])
    inc_codes = np.array([c for c in range(32) if c != T.NOISE_CODE])
    INF = 1e18

    def local(k: int) -> np.ndarray:
        if not np.isnan(force_pitch[k]):
            out = np.full(len(states), INF)
            out[int(round(min(max(force_pitch[k], 1), 510)))] = 0.0
            return out
        if not voiced[k]:
            return np.zeros(len(states))
        with np.errstate(divide="ignore"):
            cents = np.abs(1200 * np.log2(np.maximum(states, 1) / f0[k]))
        cents[0] = max_cents
        return np.minimum(cents, max_cents)

    score = np.where(even, 0.0, INF) + local(0)  # slot 0 starts from a starting pitch
    back_state = np.zeros((n, len(states)), dtype=np.int64)
    back_code = np.zeros((n, len(states)), dtype=np.int64)
    back_restart = np.zeros((n, len(states)), dtype=bool)
    back_state[0] = -1
    for k in range(1, n):
        prev = score
        best = np.full(len(states), INF)
        bstate = np.zeros(len(states), dtype=np.int64)
        bcode = np.full(len(states), T.NOISE_CODE, dtype=np.int64)
        brestart = np.zeros(len(states), dtype=bool)
        if voiced[k] or glide[k]:
            for code, d in zip(inc_codes, inc):
                src = states - d
                ok = (src >= 0) & (src <= 510)
                cand = np.full(len(states), INF)
                cand[ok] = prev[src[ok]]
                better = cand < best
                best[better] = cand[better]
                bstate[better] = src[better]
                bcode[better] = code
        else:
            best = prev.copy()
            bstate = states.copy()
        base = restart_cost if np.isnan(restart_cost_slot[k]) else restart_cost_slot[k]
        rc = min(onset_restart_cost, base) if (voiced[k] and not voiced[k - 1]) else base
        if restart_mode[k] == 1:
            rc = 0.0
        elif restart_mode[k] == -1:
            rc = INF
        r = prev.min() + rc
        rs = int(prev.argmin())
        better = even & ((r <= best) if restart_mode[k] == 1 else (r < best))
        best[better] = r
        bstate[better] = rs
        bcode[better] = 0
        brestart[better] = True
        score = best + local(k)
        back_state[k], back_code[k], back_restart[k] = bstate, bcode, brestart

    pitch = np.zeros(n, dtype=np.int64)
    pi_code = np.full(n, T.NOISE_CODE, dtype=np.int64)
    restart = np.zeros(n, dtype=bool)
    s = int(score.argmin())
    for k in range(n - 1, -1, -1):
        pitch[k] = s
        if k > 0:
            pi_code[k] = back_code[k, s]
            restart[k] = back_restart[k, s]
            s = back_state[k, s]
    restart[0] = True
    pi_code[0] = T.NOISE_CODE if not voiced[0] else 0
    pi_code[~voiced & ~glide] = T.NOISE_CODE
    # a segment without a voiced slot (a breath, a silence) has nothing to decide its
    # pitch: the DP leaves an arbitrary state there, which the chip never uses (noise or
    # silent frames) but which the file shows as the starting pitch; give it the pitch of
    # the nearest voiced segment so that the listing reads sanely
    starts = [k for k in range(n) if restart[k]] + [n]
    segs = [(a, b) for a, b in zip(starts, starts[1:])]
    first_voiced = [int(np.argmax(voiced[a:b])) + a if voiced[a:b].any() else -1 for a, b in segs]
    for i, (a, b) in enumerate(segs):
        if first_voiced[i] >= 0:
            continue
        donor = next((first_voiced[j] for j in list(range(i + 1, len(segs))) + list(range(i - 1, -1, -1))
                      if first_voiced[j] >= 0), -1)
        pitch[a:b] = pitch[donor] if donor >= 0 else 120
    start_pitch = np.where(restart, pitch // 2, 0)
    return PitchPlan(pitch, pi_code, restart, start_pitch)


# ------------------------------------------------------------------ constraints

@dataclass
class Constraints:
    """Per-slot constraints resolved from a profile's pins (see `profile.py`).

    Everything is optional: `empty(n)` constrains nothing and gives the plain encoder."""

    NONE, VOICED, UNVOICED, SILENCE = 0, 1, 2, 3

    highpass_hz: np.ndarray    # 0 = no regional high-pass
    silence_db: np.ndarray     # nan = the global threshold
    restart_cost: np.ndarray   # nan = the global cost
    ampl_db: np.ndarray        # amplitude offset in dB
    force: np.ndarray          # NONE / VOICED / UNVOICED / SILENCE
    pitch_hz: np.ndarray       # nan = free
    restart: np.ndarray        # 0 free, 1 forced before the slot, -1 forbidden

    @classmethod
    def empty(cls, n: int) -> "Constraints":
        return cls(np.zeros(n), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n),
                   np.zeros(n, dtype=np.int8), np.full(n, np.nan), np.zeros(n, dtype=np.int8))

    @property
    def n(self) -> int:
        return len(self.force)


# ------------------------------------------------------------------ 3. filters

@dataclass
class FilterFit:
    fm: tuple[int, int, int]
    bw: tuple[int, int, int, int]
    gain_db: float
    residual_db: float


def fit_filters(freqs: np.ndarray, target_db: np.ndarray, f0_hz: float | None,
                top: int = 1) -> list[FilterFit]:
    """Exhaustive search over the 2 097 152 filter code combinations; cost = variance of the
    residual in dB (the gain is free, it becomes the amplitude). Returns the `top` best
    combinations, best first.

    With A = t - R12 (16 384 generator-1/2 pairs) and B = R34 (128 generator-3/4
    combinations), the residual variance of every pair expands into
    mean(A²) - 2·mean(A·B) + mean(B²) - (mean(A) - mean(B))², so all costs come out of one
    matrix product A·Bᵀ. A few milliseconds per slot.
    """
    t = (target_db - source_db(f0_hz, freqs)).astype(np.float32)
    resp = responses_at(freqs)
    nf = len(freqs)
    r12 = (resp[0].reshape(128, 1, nf) + resp[1].reshape(1, 128, nf)).reshape(-1, nf).astype(np.float32)
    r34 = (resp[2].reshape(32, 1, nf) + resp[3][0].reshape(1, 4, nf)).reshape(-1, nf).astype(np.float32)
    a = t[None, :] - r12
    a_mean = a.mean(axis=1)
    a_sq = (a * a).mean(axis=1)
    b_mean = r34.mean(axis=1)
    b_sq = (r34 * r34).mean(axis=1)
    cross = (a @ r34.T) / np.float32(nf)
    cost = a_sq[:, None] - 2.0 * cross + b_sq[None, :] - (a_mean[:, None] - b_mean[None, :]) ** 2
    # the `top` best entries lie in the `top` rows with the smallest row minimum (any other
    # row holds no entry below the top-th value), so only those rows are ranked in full
    n34 = len(r34)
    top = min(top, cost.size)
    if top < len(r12):
        rows = np.argpartition(cost.min(axis=1), top)[:top]
        sub = cost[rows].ravel()
        k = np.argpartition(sub, top - 1)[:top] if top < len(sub) else np.arange(len(sub))
        k = k[np.argsort(sub[k])]
        idx = rows[k // n34] * n34 + (k % n34)
    else:
        idx = np.argsort(cost.ravel())[:top]
    flat = cost.ravel()
    out = []
    for k in idx:
        i, j = divmod(int(k), n34)
        p, q = divmod(i, 128)
        fm = (p // 4, q // 4, j // 16)
        bw = (p % 4, q % 4, (j // 4) % 4, j % 4)
        out.append(FilterFit(fm, bw, float(a_mean[i] - b_mean[j]), float(np.sqrt(max(float(flat[k]), 0.0)))))
    return out


class SlotTarget:
    """A slot's target spectrum with the resonator responses precomputed on its frequencies."""

    def __init__(self, freqs: np.ndarray, target_db: np.ndarray, f0_hz: float | None) -> None:
        self.freqs = freqs
        self.f0_hz = f0_hz
        self.t = target_db - source_db(f0_hz, freqs)
        self.resp = responses_at(freqs)

    def score(self, fm: tuple[int, int, int], bw: tuple[int, int, int, int]) -> FilterFit:
        return self.score_many([(fm, bw)])[0]

    def score_many(self, settings: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]) -> list[FilterFit]:
        """Residuals of several settings at once."""
        if not settings:
            return []
        fm = np.array([s[0] for s in settings])
        bw = np.array([s[1] for s in settings])
        r = self.resp
        model = r[0][fm[:, 0], bw[:, 0]] + r[1][fm[:, 1], bw[:, 1]] + r[2][fm[:, 2], bw[:, 2]] + r[3][0, bw[:, 3]]
        d = self.t[None, :] - model
        gain = d.mean(axis=1)
        res = np.sqrt(d.var(axis=1))
        return [FilterFit(tuple(int(v) for v in fm[i]), tuple(int(v) for v in bw[i]), float(gain[i]), float(res[i]))
                for i in range(len(settings))]


def diversify(fits: list[FilterFit], per_triple: int = 3) -> list[FilterFit]:
    """Keep at most `per_triple` bandwidth variants of each formant triple (best first)."""
    count: dict[tuple[int, int, int], int] = {}
    out = []
    for f in fits:
        if count.get(f.fm, 0) < per_triple:
            out.append(f)
            count[f.fm] = count.get(f.fm, 0) + 1
    return out


def bridge_settings(prev: list[FilterFit], best: FilterFit, per_prev: int = 8,
                    steps: tuple[int, ...] = (1, 2)) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    """Settings on the way from the previous slot's best candidates to this slot's best:
    each generator moves 1 or 2 table steps towards the target. A vowel change spans
    several frames in speech; without these the search offers only the two ends and the
    smoothing must jump in one 8 ms frame (a chirp) or stay wrong."""
    out = []
    seen = set()
    for p in prev[:per_prev]:
        for n in steps:
            fm = tuple(p.fm[g] + int(np.sign(best.fm[g] - p.fm[g])) * min(n, abs(best.fm[g] - p.fm[g]))
                       for g in range(3))
            if fm != p.fm and fm != best.fm and (fm, best.bw) not in seen:
                seen.add((fm, best.bw))
                out.append((fm, best.bw))
    return out


def _fit_distance(a: FilterFit, b: FilterFit, octave_db: float = 6.0, bw_db: float = 0.7,
                  knee_octave: float = 0.3, beyond_knee_db: float = 60.0) -> float:
    """Cost of moving from one filter setting to the next: dB-equivalent of the jumps.

    Formant movement above `knee_octave` per 8 ms is not speech — it is a resonance being
    re-assigned to another generator (FM2 and FM3 share 1179-3400 Hz), which the chip
    renders as two crossing sweeps, a chirp — so it is priced far above any fit gain."""
    d = 0.0
    for gen in range(3):
        oct_ = abs(np.log2(FM_TABLES[gen][a.fm[gen]] / FM_TABLES[gen][b.fm[gen]]))
        d += octave_db * oct_ + beyond_knee_db * max(0.0, oct_ - knee_octave)
    d += bw_db * sum(abs(x - y) for x, y in zip(a.bw, b.bw))
    return d


def _features(fits: list[FilterFit]) -> tuple[np.ndarray, np.ndarray]:
    """(log2 formant frequencies (n, 3), bandwidth codes (n, 4)) of a candidate list."""
    lf = np.array([[np.log2(FM_TABLES[g][f.fm[g]]) for g in range(3)] for f in fits])
    bw = np.array([f.bw for f in fits], dtype=np.float64)
    return lf, bw


def _transition_matrix(prev: tuple[np.ndarray, np.ndarray], cur: tuple[np.ndarray, np.ndarray], octave_db: float = 6.0,
                       bw_db: float = 0.7, knee_octave: float = 0.3, beyond_knee_db: float = 60.0) -> np.ndarray:
    """Vectorised `_fit_distance` for every (previous, current) pair of feature sets."""
    lp, bp = prev
    lc, bc = cur
    oct_ = np.abs(lp[:, None, :] - lc[None, :, :])
    d = octave_db * oct_.sum(axis=2) + beyond_knee_db * np.maximum(0.0, oct_ - knee_octave).sum(axis=2)
    d += bw_db * np.abs(bp[:, None, :] - bc[None, :, :]).sum(axis=2)
    return d


def smooth_filters(candidates: list[list[FilterFit]], linked: list[bool]) -> list[FilterFit]:
    """Viterbi over per-slot candidate lists: residual + jump cost between consecutive slots.
    `linked[k]` says whether slot k follows slot k-1 without a gap."""
    n = len(candidates)
    if n == 0:
        return []
    K = tuning.current
    score = np.array([c.residual_db for c in candidates[0]])
    back: list[np.ndarray] = [np.full(len(candidates[0]), -1)]
    prev_feat = _features(candidates[0])
    for k in range(1, n):
        cur = candidates[k]
        cur_feat = _features(cur)
        if linked[k]:
            trans = _transition_matrix(prev_feat, cur_feat, octave_db=K.octave_db, bw_db=K.bw_db,
                                       knee_octave=K.knee_octave, beyond_knee_db=K.beyond_knee_db)
            total = score[:, None] + trans
            b = total.argmin(axis=0)
            score = total[b, np.arange(len(cur))] + np.array([c.residual_db for c in cur])
        else:
            b = np.full(len(cur), int(score.argmin()))
            score = score.min() + np.array([c.residual_db for c in cur])
        back.append(b)
        prev_feat = cur_feat
    out = [None] * n
    j = int(score.argmin())
    for k in range(n - 1, -1, -1):
        out[k] = candidates[k][j]
        j = int(back[k][j])
    return out


# ------------------------------------------------------------------ 4. amplitude

def steady_output(pitch_hz: float, noise: bool, fm: tuple[int, int, int], bw: tuple[int, int, int, int],
                  n: int = 1024, seed: int = 1) -> np.ndarray:
    """Steady-state chip output for amplitude code 15 — float model, unclipped, so that the
    clipped level of every amplitude code can be predicted by scaling."""
    if noise:
        rng = np.random.default_rng(seed)
        src = rng.integers(0, 2 * QUANT, size=n) - QUANT
    else:
        phi = (np.arange(n) * max(int(pitch_hz), 1)) % RATE
        src = (phi * QUANT * 2) // RATE - QUANT
    x = src.astype(np.float64) * (1000 / 32)
    for gen, (fm_code, bw_code) in enumerate(zip((*fm, 0), bw)):
        f = FM_TABLES[gen][fm_code]
        b_ = T.BW_HZ[bw_code]
        r = np.exp(-np.pi * b_ / RATE)
        x = lfilter([1.0], [1.0, -2 * r * np.cos(2 * np.pi * f / RATE), r * r], x)
    return x[n // 2:]


def fit_amplitude(target_rms: float, full: np.ndarray, min_code: int = 1, silence_margin_db: float = 6.0,
                  quiet_db: float = -45.0) -> int:
    """Amplitude code whose clipped steady-state RMS is closest to the target (dB).

    Code 1 is the quietest setting the chip has with the chosen resonators; a target that
    is both quiet in absolute terms (under `quiet_db` re full scale, near the 8-bit DAC's
    step) and more than `silence_margin_db` under code 1 is closer to silence than to
    anything the chip can say and gets code 0 — otherwise room noise at -60 dB comes out
    18 dB too loud. A quiet but audible onset (-36 dB) keeps code 1 even when too loud."""
    if target_rms <= 0:
        return 0
    y1 = np.clip(full * (T.AMPL_PERMILLE[max(min_code, 1)] / 1000.0), -32767, 32767)
    r1 = float(np.sqrt(np.mean(y1 * y1)))
    target_db = 20 * np.log10(target_rms / 32768.0)
    if r1 > 0 and target_db < quiet_db and 20 * np.log10(target_rms / r1) < -silence_margin_db:
        return 0
    best, best_err = min_code, np.inf
    for code in range(max(min_code, 1), 16):
        y = np.clip(full * (T.AMPL_PERMILLE[code] / 1000.0), -32767, 32767)
        r = float(np.sqrt(np.mean(y * y)))
        if r <= 0:
            continue
        err = abs(20 * np.log10(r / target_rms))
        if err < best_err:
            best, best_err = code, err
    return best


# ------------------------------------------------------------------ driver

@dataclass
class EncodeResult:
    utterances: list[Utterance]
    analysis: Analysis
    pitch: PitchPlan
    frames: list[Frame] = field(default_factory=list)   # one per 8 ms slot, before splitting
    fits: list[FilterFit] = field(default_factory=list)


def demote_onset_blips(voiced: np.ndarray, f0: np.ndarray, max_len: int = 6, lookahead: int = 10,
                       ratio: float = 1.7) -> np.ndarray:
    """A short voiced run pitched more than `ratio` above the run that follows it within
    `lookahead` slots is an octave-up error of the tracker on a breathy onset (half a
    period read as the period): the chip would chirp at 500 Hz for 40 ms before the
    word. Such runs become unvoiced."""
    v = voiced.copy()
    n = len(v)
    k = 0
    while k < n:
        if not v[k]:
            k += 1
            continue
        j = k
        while j < n and v[j]:
            j += 1
        # the head of a long run: the blip and the word are one voiced stretch
        for length in range(min(max_len, j - k - 4), 0, -1):    # the longest such head,
            head = np.nanmin(f0[k:k + length])                       # every slot of it high
            rest = np.nanmedian(f0[k + length:k + length + lookahead])
            if np.isfinite(head) and np.isfinite(rest) and head > ratio * rest:
                v[k:k + length] = False
                break
        if j - k <= max_len:
            m = j
            while m < n and not v[m] and m - j < lookahead:
                m += 1
            if m < n and v[m]:
                e = m
                while e < n and v[e]:
                    e += 1
                here, there = np.nanmedian(f0[k:j]), np.nanmedian(f0[m:e])
                if np.isfinite(here) and np.isfinite(there) and here > ratio * there:
                    v[k:j] = False
        k = j
    return v


def hysteresis(energy_db: np.ndarray, on_db: np.ndarray, off_db: np.ndarray) -> np.ndarray:
    """Active once the energy rises above `on`, until it falls below `off`: noise that
    grazes a single threshold would flicker in and out, 8 ms at a time."""
    out = np.zeros(len(energy_db), dtype=bool)
    state = False
    for k in range(len(energy_db)):
        state = energy_db[k] > (off_db[k] if state else on_db[k])
        out[k] = state
    return out


def clean_runs(active: np.ndarray, min_active: int = 2, max_gap: int = 2) -> np.ndarray:
    """Activity without flicker: gaps of up to `max_gap` slots inside speech are kept
    active (an 8 ms hole is a click), active blips shorter than `min_active` slots are
    dropped (noise grazing the threshold)."""
    a = active.copy()
    n = len(a)
    i = 0
    while i < n:                       # fill short gaps between active runs
        if not a[i]:
            j = i
            while j < n and not a[j]:
                j += 1
            if 0 < i and j < n and j - i <= max_gap:
                a[i:j] = True
            i = j
        else:
            i += 1
    i = 0
    while i < n:                       # drop short active blips
        if a[i]:
            j = i
            while j < n and a[j]:
                j += 1
            if j - i < min_active:
                a[i:j] = False
            i = j
        else:
            i += 1
    return a


def highpass(x8: np.ndarray, hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    return sosfiltfilt(butter(4, hz, btype="highpass", fs=RATE, output="sos"), x8)


def _smooth_mask(mask: np.ndarray, half: int) -> np.ndarray:
    k = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(mask.astype(float), k, mode="same")


def _resize(c: Constraints, n: int) -> Constraints:
    def fit(a, fill):
        out = np.full(n, fill, dtype=a.dtype)
        m = min(n, len(a))
        out[:m] = a[:m]
        return out
    return Constraints(fit(c.highpass_hz, 0), fit(c.silence_db, np.nan), fit(c.restart_cost, np.nan),
                       fit(c.ampl_db, 0), fit(c.force, 0), fit(c.pitch_hz, np.nan), fit(c.restart, 0))


def encode(x: np.ndarray, rate: int, restart_cost: float | None = None, silence_db: float | None = None,
           output_rms_fullscale: float = 32768.0, candidates: int | None = None, carry_db: float | None = None,
           max_candidates: int | None = None, denoise: bool = True, noise_margin_db: float | None = None,
           highpass_hz: float | None = None, single: bool = False,
           constraints: Constraints | None = None, cache: dict | None = None,
           rate_scale: float = 1.0, progress=None) -> EncodeResult:
    """Encode `x` (any rate) into MEA8000 frames.

    `constraints` carries the pins of a profile (per slot); `cache`, when given, memoises
    the candidate search by target content across calls, so that a re-encode after a pin
    only searches the slots whose target changed. `rate_scale` compensates the chip's
    clock (`Profile.rate_scale`): the recording is analysed as if its rate were
    `rate * rate_scale`, so that a chip clocked faster than the 3.84 MHz reference says it
    at the recorded speed and pitch."""
    K = tuning.current
    restart_cost = K.restart_cost if restart_cost is None else restart_cost
    silence_db = K.silence_db if silence_db is None else silence_db
    candidates = K.candidates if candidates is None else candidates
    carry_db = K.carry_db if carry_db is None else carry_db
    max_candidates = K.max_candidates if max_candidates is None else max_candidates
    noise_margin_db = K.noise_margin_db if noise_margin_db is None else noise_margin_db
    x8 = to_8k(x, rate * rate_scale)
    if highpass_hz:
        # removes what the chip cannot say anyway (kick drums, rumble) before analysis; the
        # formant fit starts at 100 Hz and the pitch survives on the harmonics
        x8 = highpass(x8, highpass_hz)
    n_expected = int(np.ceil(len(x8) / HOP))
    if constraints is None:
        constraints = Constraints.empty(n_expected)
    C = constraints
    if np.any(C.highpass_hz > 0):
        # regional high-pass: the filtered signal replaces the source inside the regions,
        # with a short crossfade, before anything is analysed
        x8 = x8.copy()
        for hz in np.unique(C.highpass_hz[C.highpass_hz > 0]):
            y = highpass(x8, float(hz))
            mask = np.repeat(C.highpass_hz == hz, HOP)[: len(x8)]
            mask = np.pad(mask, (0, len(x8) - len(mask)))
            w = _smooth_mask(mask, HOP // 2)
            x8 = x8 * (1 - w) + y * w
    an = analyze(x8, RATE)
    n = an.n_frames
    if C.n != n:
        C = _resize(C, n)
    # background: estimated on the quietest slots; slots that do not rise above it are
    # silence, and its spectrum is subtracted from every target and from the slot energies
    profile = NoiseProfile(x8, an.energy_db) if denoise else None
    if profile is not None and not profile.stationary:
        profile = None
    if profile is not None and profile.energy_db > -100:
        silence_db = max(silence_db, profile.energy_db + noise_margin_db)
        energy_lin = 10 ** (an.energy_db / 10) - 10 ** (profile.energy_db / 10)
        energy_db = 10 * np.log10(np.maximum(energy_lin, 10 ** ((profile.energy_db - 10) / 10)))
    else:
        energy_db = an.energy_db
    silence_slot = np.where(np.isnan(C.silence_db), silence_db, C.silence_db)
    active = clean_runs(hysteresis(an.energy_db, silence_slot + K.hysteresis_db, silence_slot - K.hysteresis_db),
                        min_active=K.min_active, max_gap=K.max_gap)
    voiced = an.voiced & active
    voiced = demote_onset_blips(voiced, an.f0)
    # value pins on voicing
    active &= C.force != Constraints.SILENCE
    voiced = (voiced | (C.force == Constraints.VOICED)) & active & (C.force != Constraints.UNVOICED)
    f0_in = np.where(voiced, an.f0, np.nan)
    f0_in = np.where(np.isnan(f0_in) & voiced, np.nanmedian(an.f0) if np.any(~np.isnan(an.f0)) else 120.0, f0_in)
    if single:
        # one speech file like the Cedic-Nathan data: a single STOP + starting pitch, the
        # pitch glides through the silences instead of restarting at each word
        plan = fit_pitch(f0_in, voiced, restart_cost=np.inf, onset_restart_cost=np.inf, glide=~active,
                         max_cents=K.max_cents,
                         force_pitch=np.where(voiced, C.pitch_hz, np.nan), restart_mode=C.restart)
    else:
        plan = fit_pitch(f0_in, voiced, restart_cost=restart_cost, onset_restart_cost=K.onset_restart_cost,
                         max_cents=K.max_cents,
                         force_pitch=np.where(voiced, C.pitch_hz, np.nan), restart_mode=C.restart,
                         restart_cost_slot=C.restart_cost)

    # per-slot candidate filter settings, then the smoothest path through them
    silent = FilterFit((0, 0, 0), (0, 0, 0, 0), 0.0, 0.0)
    slots = [k for k in range(n) if active[k]]
    cands: list[list[FilterFit]] = []
    for i, k in enumerate(slots):
        if progress is not None and i % 32 == 0:
            progress(i / max(1, len(slots)))
        f0 = float(f0_in[k]) if voiced[k] else None
        if voiced[k]:
            freqs, target = harmonic_target(x8, k * HOP, f0, profile)
        else:
            freqs, target = noise_target(x8, k * HOP, profile)
        if len(freqs) < 4:
            cands.append([silent])
            continue
        if cache is not None:
            key = hashlib.blake2b(freqs.tobytes() + target.tobytes() + repr((f0, candidates)).encode(),
                                  digest_size=16).digest()
            c = cache.get(key)
            if c is None:
                c = diversify(fit_filters(freqs, target, f0, top=candidates), per_triple=K.per_triple)
                cache[key] = c
            c = list(c)
        else:
            c = diversify(fit_filters(freqs, target, f0, top=candidates), per_triple=K.per_triple)
        # carry the previous slot's candidates over (re-scored on this target) while they
        # still fit within `carry_db` of the best: "stay" must remain an option along a whole
        # run, otherwise the smoothing is forced to jump wherever the carried set runs out
        if i > 0 and slots[i - 1] == k - 1:
            st = SlotTarget(freqs, target, f0)
            limit = c[0].residual_db + carry_db
            seen = {(f.fm, f.bw) for f in c}
            carried = [(p.fm, p.bw) for p in cands[-1] if (p.fm, p.bw) not in seen]
            for s in st.score_many(carried):
                if s.residual_db <= limit:
                    c.append(s)
            # and the steps in between, so that a vowel change can be walked over a few
            # frames instead of jumped in one (they are kept regardless of their fit: the
            # smoothing weighs them against the jump)
            seen = {(f.fm, f.bw) for f in c}
            bridges = [b for b in bridge_settings(cands[-1], c[0], per_prev=K.per_prev) if b not in seen]
            c.extend(st.score_many(bridges))
            if len(c) > max_candidates:
                c = c[:candidates] + sorted(c[candidates:], key=lambda f: f.residual_db)[: max_candidates - candidates]
        cands.append(c)
    # consecutive slots are linked unless a restart presets the chip (no interpolation there)
    linked = [i > 0 and slots[i] == slots[i - 1] + 1 and not plan.restart[slots[i]] for i in range(len(slots))]
    chosen = smooth_filters(cands, linked)
    fit_of = dict(zip(slots, chosen))

    frames: list[Frame] = []
    fits: list[FilterFit] = []
    last_fit = silent
    for k in range(n):
        if not active[k]:
            # true silence (AMPL = 0 is a silent frame inside a Philips chunk); the filters
            # stay so the chip does not sweep towards a dummy setting
            fit, ampl = last_fit, 0
        else:
            fit = fit_of[k]
            last_fit = fit
            full = steady_output(plan.pitch_hz[k], not voiced[k], fit.fm, fit.bw)
            target_rms = 10 ** ((energy_db[k] + C.ampl_db[k]) / 20) * output_rms_fullscale
            ampl = fit_amplitude(target_rms, full, silence_margin_db=K.silence_margin_db, quiet_db=K.quiet_db)
        frames.append(Frame(bw=fit.bw, fm=fit.fm, ampl=ampl, fd=0, pi=int(plan.pi_code[k])))
        fits.append(fit)

    return EncodeResult(split_utterances(frames, plan), an, plan, frames, fits)


def split_utterances(frames: list[Frame], plan: PitchPlan, per_chunk: list[list[Frame]] | None = None) -> list[Utterance]:
    """Group per-slot frames into utterances at the plan's restarts, each starting with its
    pitch byte and ending with the dummy AMPL = 0 frame of TP101 fig. 19 (the player sends
    STOP once it has started). `per_chunk` substitutes the frame list of each chunk."""
    starts = [k for k in range(len(frames)) if plan.restart[k] or k == 0]
    chunks = [frames[a:b] for a, b in zip(starts, starts[1:] + [len(frames)])]
    if per_chunk is not None:
        chunks = per_chunk
    utts = []
    for k, body in zip(starts, chunks):
        u = Utterance(pitch=int(plan.start_pitch[k]), frames=list(body), extra=0)
        last = u.frames[-1] if u.frames else Frame((0, 0, 0, 0), (0, 0, 0), 0, 0, T.NOISE_CODE)
        u.frames.append(Frame(bw=last.bw, fm=last.fm, ampl=0, fd=0, pi=T.NOISE_CODE if last.noise else 0))
        utts.append(u)
    return utts
