"""Fast MEA8000 simulator: the integer chip of `sim.py` compiled by numba.

Same state machine, same arithmetic, same feeding policies; the CPU side is flattened
into a list of events (wait for REQ, data byte, command, drain) that the compiled loop
consumes. Bit-exact with `sim.render` for the integer models (`tests/test_fastsim.py`);
float mode falls back to the reference chip. Faster, not dramatically: a 30 s stream
renders in a fraction of a second where the reference chip takes about a second and a
half (both are vectorised where it counts).
"""

from __future__ import annotations

import numpy as np

from . import tables as T
from .codec import Utterance
from .sim import DEFAULT, NOISE_LEN, NOMINAL_F0, QUANT, SUPERSAMPLING, TABLE_LEN, Model, noise_table, render

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # the reference chip in pure Python does the same work, slowly
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

EV_WAIT, EV_DATA, EV_CMD, EV_DRAIN = 0, 1, 2, 3
ST_STOPPED, ST_WAIT_FIRST, ST_STARTED, ST_SLOWING = 0, 1, 2, 3


def _tables(model: Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = np.arange(TABLE_LEN, dtype=np.float64)
    f = i / NOMINAL_F0
    cos_t = 2.0 * np.cos(2.0 * np.pi * f) * QUANT
    exp_t = np.exp(-np.pi * f) * QUANT
    exp2_t = np.exp(-2.0 * np.pi * f) * QUANT
    # C double -> int truncates toward zero
    return (np.trunc(cos_t).astype(np.int64), np.trunc(exp_t).astype(np.int64),
            np.trunc(exp2_t).astype(np.int64))


def events_for(utterances: list[Utterance], policy: str) -> np.ndarray:
    """The CPU side of `sim.render`, as (kind, value) rows."""
    if policy not in ("chunk", "philips"):
        raise ValueError(f"unknown policy {policy!r}")
    ev: list[tuple[int, int]] = []
    for utt in utterances:
        ev.append((EV_WAIT, 0))
        ev.append((EV_CMD, T.CMD_POWER_ON))
        ev.append((EV_DATA, utt.pitch))
        for frame in utt.frames:
            for byte in frame.to_bytes():
                ev.append((EV_WAIT, 0))
                ev.append((EV_DATA, byte))
            ev.append((EV_WAIT, 0))
        if policy == "chunk":
            ev.append((EV_CMD, T.CMD_POWER_ON))
        else:
            ev.append((EV_DRAIN, 0))
    return np.asarray(ev, dtype=np.int64).reshape(-1, 2)


@njit(cache=True)
def _cdiv(a, b):
    q = abs(a) // abs(b)
    if (a >= 0) == (b > 0):
        return q
    return -q


@njit(cache=True)
def _i32(x):
    x &= 0xFFFFFFFF
    if x & 0x80000000:
        return x - 0x100000000
    return x


@njit(cache=True)
def _run(events, cos_t, exp_t, exp2_t, noise_t, f0, ampl_fix, fade_fix, half, pitch_scale,
         truncate_bits, pi_hz, bw_hz, fm1_hz, fm2_hz, fm3_hz, fm4_hz, ampl_permille, out):
    # chip state
    state = ST_STOPPED
    buf = np.zeros(4, np.int64)
    bufpos = 0
    cont = 0
    framelength = 0
    framepos = 0
    framelog = 0
    lastsample = 0
    sample = 0
    phi = 0
    fm = np.zeros(4, np.int64); last_fm = np.zeros(4, np.int64)
    bw = np.zeros(4, np.int64); last_bw = np.zeros(4, np.int64)
    fout = np.zeros(4, np.int64); last_fout = np.zeros(4, np.int64)
    last_ampl = 0
    ampl = 0
    last_pitch = 0
    pitch = 0
    noise = False
    output = 0
    n = 0
    for e in range(events.shape[0]):
        kind = events[e, 0]
        val = events[e, 1]
        if kind == EV_CMD:
            if val & 8:
                cont = (val >> 2) & 1
            if val & 0x10:
                state = ST_STOPPED
                output = 0
            continue
        if kind == EV_DATA:
            if state == ST_STOPPED:
                pitch = 2 * val
                state = ST_WAIT_FIRST
                bufpos = 0
            elif bufpos == 4:
                pass  # data overflow, dropped like the reference
            else:
                buf[bufpos] = val
                bufpos += 1
                if bufpos == 4 and state == ST_WAIT_FIRST:
                    old_pitch = pitch
                    last_pitch = old_pitch
                    # decode_frame
                    fd = (buf[3] >> 5) & 3
                    pi = pi_hz[buf[3] & 0x1F] << fd
                    noise = (buf[3] & 0x1F) == 16
                    pitch = (last_pitch + pi) & 0xFFFF
                    bw[0] = bw_hz[buf[0] >> 6]; bw[1] = bw_hz[(buf[0] >> 4) & 3]
                    bw[2] = bw_hz[(buf[0] >> 2) & 3]; bw[3] = bw_hz[buf[0] & 3]
                    fm[3] = fm4_hz; fm[2] = fm3_hz[buf[1] >> 5]
                    fm[1] = fm2_hz[buf[1] & 0x1F]; fm[0] = fm1_hz[buf[2] >> 3]
                    ampl = ampl_permille[((buf[2] & 7) << 1) | (buf[3] >> 7)]
                    framelog = fd + 6 + 3
                    framelength = 1 << framelog
                    bufpos = 0
                    # shift_frame
                    last_pitch = pitch
                    for i in range(4):
                        last_bw[i] = bw[i]; last_fm[i] = fm[i]
                    last_ampl = ampl
                    last_pitch = old_pitch
                    if fade_fix:
                        last_ampl = 0
                    else:
                        ampl = 0
                    framepos = 0
                    state = ST_STARTED
            continue
        # EV_WAIT: tick until the chip accepts a byte; EV_DRAIN: tick until inactive
        while True:
            active = state == ST_STARTED or state == ST_SLOWING
            if kind == EV_WAIT:
                accept = state == ST_STOPPED or state == ST_WAIT_FIRST or (state == ST_STARTED and bufpos < 4)
                if accept:
                    break
            else:
                if not active:
                    break
            # tick
            if not active:
                out[n] = output; n += 1
                continue
            pos = framepos % SUPERSAMPLING
            if pos == 0:
                lastsample = sample
                # compute_sample_int
                a = last_ampl + (((ampl - last_ampl) * framepos) >> framelog)
                if noise:
                    phi = (phi + 1) % NOISE_LEN
                    o = noise_t[phi]
                else:
                    p = last_pitch + (((pitch - last_pitch) * framepos) >> framelog)
                    if pitch_scale != 1.0:
                        p = int(p * pitch_scale)
                    phi = (phi + p) % f0
                    o = _cdiv((phi % f0) * QUANT * 2, f0) - QUANT
                if ampl_fix:
                    o = _cdiv(o * a, 32)
                else:
                    o *= _cdiv(a, 32)
                for i in range(4):
                    fmi = last_fm[i] + (((fm[i] - last_fm[i]) * framepos) >> framelog)
                    bwi = last_bw[i] + (((bw[i] - last_bw[i]) * framepos) >> framelog)
                    b = _cdiv(cos_t[fmi] * exp_t[bwi], QUANT)
                    c = exp2_t[bwi]
                    nxt = _i32(o + _cdiv(b * fout[i] - c * last_fout[i], QUANT))
                    last_fout[i] = fout[i]
                    fout[i] = nxt
                    o = nxt
                if half:
                    o = _cdiv(o, 2)
                if o > 32767:
                    o = 32767
                elif o < -32767:
                    o = -32767
                if truncate_bits:
                    o = (o >> (16 - truncate_bits)) << (16 - truncate_bits)
                sample = o
                output = lastsample
            else:
                output = lastsample + _cdiv(pos * (sample - lastsample), SUPERSAMPLING)
            framepos += 1
            if framepos >= framelength:
                # shift_frame
                last_pitch = pitch
                for i in range(4):
                    last_bw[i] = bw[i]; last_fm[i] = fm[i]
                last_ampl = ampl
                if bufpos == 4:
                    fd = (buf[3] >> 5) & 3
                    pi = pi_hz[buf[3] & 0x1F] << fd
                    noise = (buf[3] & 0x1F) == 16
                    pitch = (last_pitch + pi) & 0xFFFF
                    bw[0] = bw_hz[buf[0] >> 6]; bw[1] = bw_hz[(buf[0] >> 4) & 3]
                    bw[2] = bw_hz[(buf[0] >> 2) & 3]; bw[3] = bw_hz[buf[0] & 3]
                    fm[3] = fm4_hz; fm[2] = fm3_hz[buf[1] >> 5]
                    fm[1] = fm2_hz[buf[1] & 0x1F]; fm[0] = fm1_hz[buf[2] >> 3]
                    ampl = ampl_permille[((buf[2] & 7) << 1) | (buf[3] >> 7)]
                    framelog = fd + 6 + 3
                    framelength = 1 << framelog
                    bufpos = 0
                    framepos = 0
                elif cont:
                    framepos = 0
                elif state == ST_STARTED:
                    ampl = 0
                    framepos = 0
                    state = ST_SLOWING
                elif state == ST_SLOWING:
                    state = ST_STOPPED
                    output = 0
            out[n] = output; n += 1
    return n


def render_fast(utterances: list[Utterance], model: Model = DEFAULT, policy: str = "chunk",
                noise: np.ndarray | None = None) -> np.ndarray:
    """Same result as `sim.render`, compiled. Float-mode models, and installations without
    numba (`pip install mea8000-encoder[fast]`), use the reference chip."""
    if model.float_mode or not model.int_tables or not HAVE_NUMBA:
        return render(utterances, model, policy, noise)
    ev = events_for(utterances, policy)
    cos_t, exp_t, exp2_t = _tables(model)
    noise_t = np.asarray(noise if noise is not None else noise_table(model.noise_seed), dtype=np.int64)
    n_frames = sum(len(u.frames) for u in utterances)
    out = np.zeros((n_frames + 3 * len(utterances) + 2) * 4096 + 8192, dtype=np.int64)
    n = _run(ev, cos_t, exp_t, exp2_t, noise_t, NOMINAL_F0, model.ampl_division_fix,
             model.first_frame_fade_fix, model.java_half_output, float(model.pitch_scale),
             model.truncate_bits,
             np.asarray(T.PI_HZ, dtype=np.int64), np.asarray(T.BW_HZ, dtype=np.int64),
             np.asarray(T.FM1_HZ, dtype=np.int64), np.asarray(T.FM2_HZ, dtype=np.int64),
             np.asarray(T.FM3_HZ, dtype=np.int64), int(T.FM4_HZ),
             np.asarray(T.AMPL_PERMILLE, dtype=np.int64), out)
    return out[:n].astype(np.int16)
