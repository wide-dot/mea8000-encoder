"""MEA8000 chip simulator.

The reference model is Antoine Miné's MAME device (`doc/sources/mame/mea8000.cpp`),
reproduced operation for operation in integer arithmetic, with every known deviation
exposed as a `Model` switch so that each oracle (MAME integer, MAME float, legacy Java,
Philips documents) can be matched exactly and the differences measured.

Sample rates: filters run at F0 = clock / 480, output at 8 x F0 by linear interpolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from . import tables as T
from .codec import Utterance

QUANT = 512
TABLE_LEN = 3600
NOISE_LEN = 8192
SUPERSAMPLING = 8
NOMINAL_F0 = 8000     # the chip's DSP works in samples and does not know its clock: its
                      # tables are those of the 3.84 MHz reference (F0 = 8 kHz); another
                      # clock only changes the output rate, so pitch, formants and
                      # durations all scale with it (MAME instead recomputes the tables
                      # from the clock, which keeps the frequencies nominal)


@dataclass(frozen=True)
class Model:
    """Behavioural switches of the simulator."""

    clock_hz: int = 3_840_000
    float_mode: bool = False
    # MAME stores cos/exp tables as int; the legacy Java port keeps them as double
    int_tables: bool = True
    # `out *= ampl / 32` (MAME) vs `out = out * ampl / 32` (Java, dcmoto 2024)
    ampl_division_fix: bool = True
    # first frame fades in from zero (TP101, Java, dcmoto 2024) vs MAME's inverted fade
    first_frame_fade_fix: bool = True
    # the legacy Java port halves the output (`out = out / 2`)
    java_half_output: bool = False
    # TP101: exact pitch values are the nominal ones x 1.024
    pitch_scale: float = 1.0
    # TP101: filter output truncated to 11 bits before the interpolator and the 8-bit DAC
    truncate_bits: int = 0
    noise_seed: int = 1

    @property
    def f0(self) -> int:
        return self.clock_hz // 480

    @property
    def sample_rate(self) -> int:
        return self.f0 * SUPERSAMPLING


MAME_INT = Model(ampl_division_fix=False, first_frame_fade_fix=False)
MAME_FLOAT = Model(float_mode=True, first_frame_fade_fix=False)
LEGACY_JAVA = Model(int_tables=False, java_half_output=True)
DEFAULT = Model()


class State(Enum):
    STOPPED = 0
    WAIT_FIRST = 1
    STARTED = 2
    SLOWING = 3


def _div(a: int, b: int) -> int:
    """C integer division (truncation toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def _u16(x: int) -> int:
    return x & 0xFFFF


def noise_table(seed: int) -> np.ndarray:
    """Deterministic replacement for MAME's `machine().rand()` noise table."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2 * QUANT, size=NOISE_LEN, dtype=np.int64) - QUANT


class _Filter:
    __slots__ = ("fm", "last_fm", "bw", "last_bw", "output", "last_output")

    def __init__(self) -> None:
        self.fm = self.last_fm = 0
        self.bw = self.last_bw = 0
        self.output = self.last_output = 0


class Chip:
    """Register-level model: `write(0, data)`, `write(1, cmd)`, `read()`, `tick()`."""

    def __init__(self, model: Model = DEFAULT, noise: np.ndarray | None = None) -> None:
        self.model = model
        self.f0 = NOMINAL_F0
        self._init_tables(noise if noise is not None else noise_table(model.noise_seed))
        self.state = State.STOPPED
        self.buf = [0, 0, 0, 0]
        self.bufpos = 0
        self.cont = 0
        self.roe = 0
        self.framelength = 0
        self.framepos = 0
        self.framelog = 0
        self.lastsample = 0
        self.sample = 0
        self.phi = 0
        self.f = [_Filter() for _ in range(4)]
        self.last_ampl = 0
        self.ampl = 0
        self.last_pitch = 0
        self.pitch = 0
        self.noise = False
        self.output = 0
        self.frames_decoded = 0
        self.log: list[str] = []

    # ------------------------------------------------------------------ tables
    def _init_tables(self, noise: np.ndarray) -> None:
        i = np.arange(TABLE_LEN, dtype=np.float64)
        f = i / self.f0
        cos_t = 2.0 * np.cos(2.0 * math.pi * f) * QUANT
        exp_t = np.exp(-math.pi * f) * QUANT
        exp2_t = np.exp(-2.0 * math.pi * f) * QUANT
        if self.model.int_tables:
            self.cos_table = [int(x) for x in cos_t]  # C double -> int truncates toward zero
            self.exp_table = [int(x) for x in exp_t]
            self.exp2_table = [int(x) for x in exp2_t]
        else:
            self.cos_table = list(cos_t)
            self.exp_table = list(exp_t)
            self.exp2_table = list(exp2_t)
        self.noise_table = [int(x) for x in noise]

    # ------------------------------------------------------------------ REQ
    def accept_byte(self) -> bool:
        return (self.state in (State.STOPPED, State.WAIT_FIRST)
                or (self.state == State.STARTED and self.bufpos < 4))

    def read(self) -> int:
        return 0x80 if self.accept_byte() else 0

    # ------------------------------------------------------------------ DSP, integer mode
    def _interp(self, org: int, dst: int) -> int:
        return org + (((dst - org) * self.framepos) >> self.framelog)

    def _filter_step(self, i: int, inp: int) -> int:
        flt = self.f[i]
        fm = self._interp(flt.last_fm, flt.fm)
        bw = self._interp(flt.last_bw, flt.bw)
        if self.model.int_tables:
            b = _div(self.cos_table[fm] * self.exp_table[bw], QUANT)
            c = self.exp2_table[bw]
        else:
            b = int(self.cos_table[fm] * self.exp_table[bw] / QUANT)
            c = int(self.exp2_table[bw])
        nxt = _i32(inp + _div(b * flt.output - c * flt.last_output, QUANT))
        flt.last_output = flt.output
        flt.output = nxt
        return nxt

    def _noise_gen(self) -> int:
        self.phi = (self.phi + 1) % NOISE_LEN
        return self.noise_table[self.phi]

    def _freq_gen(self) -> int:
        pitch = self._interp(self.last_pitch, self.pitch)
        if self.model.pitch_scale != 1.0:
            pitch = int(pitch * self.model.pitch_scale)
        self.phi = (self.phi + pitch) % self.f0
        return _div((self.phi % self.f0) * QUANT * 2, self.f0) - QUANT

    def _compute_sample_int(self) -> int:
        ampl = self._interp(self.last_ampl, self.ampl)
        out = self._noise_gen() if self.noise else self._freq_gen()
        if self.model.ampl_division_fix:
            out = _div(out * ampl, 32)
        else:
            out *= _div(ampl, 32)
        for i in range(4):
            out = self._filter_step(i, out)
        if self.model.java_half_output:
            out = _div(out, 2)
        out = max(-32767, min(32767, out))
        if self.model.truncate_bits:
            out = (out >> (16 - self.model.truncate_bits)) << (16 - self.model.truncate_bits)
        return out

    # ------------------------------------------------------------------ DSP, float mode
    def _interp_f(self, org: float, dst: float) -> float:
        return org + ((dst - org) * self.framepos) / self.framelength

    def _filter_step_f(self, i: int, inp: float) -> float:
        flt = self.f[i]
        fm = self._interp_f(flt.last_fm, flt.fm)
        bw = self._interp_f(flt.last_bw, flt.bw)
        b = 2.0 * math.cos(2.0 * math.pi * fm / self.f0)
        c = -math.exp(-math.pi * bw / self.f0)
        nxt = inp - c * (b * flt.output + c * flt.last_output)
        flt.last_output = flt.output
        flt.output = nxt
        return nxt

    def _noise_gen_f(self) -> float:
        self.phi += 1
        return self.noise_table[self.phi % NOISE_LEN] / QUANT

    def _freq_gen_f(self) -> float:
        pitch = int(self._interp_f(self.last_pitch, self.pitch))
        if self.model.pitch_scale != 1.0:
            pitch = int(pitch * self.model.pitch_scale)
        self.phi = (self.phi + pitch) & 0xFFFFFFFF
        return (self.phi % self.f0) / (self.f0 / 2.0) - 1.0

    def _compute_sample_float(self) -> int:
        ampl = self._interp_f(8.0 * self.last_ampl, 8.0 * self.ampl)
        out = self._noise_gen_f() if self.noise else self._freq_gen_f()
        out *= ampl
        for i in range(4):
            out = self._filter_step_f(i, out)
        if self.model.java_half_output:
            out /= 2.0
        out = int(max(-32767.0, min(32767.0, out)))
        if self.model.truncate_bits:
            out = (out >> (16 - self.model.truncate_bits)) << (16 - self.model.truncate_bits)
        return out

    def _compute_sample(self) -> int:
        return self._compute_sample_float() if self.model.float_mode else self._compute_sample_int()

    # ------------------------------------------------------------------ frames
    def _shift_frame(self) -> None:
        self.last_pitch = self.pitch
        for flt in self.f:
            flt.last_bw = flt.bw
            flt.last_fm = flt.fm
        self.last_ampl = self.ampl

    def _decode_frame(self) -> None:
        b = self.buf
        fd = (b[3] >> 5) & 3
        pi = T.PI_HZ[b[3] & 0x1F] << fd
        self.noise = (b[3] & 0x1F) == 16
        self.pitch = _u16(self.last_pitch + pi)
        self.f[0].bw = T.BW_HZ[b[0] >> 6]
        self.f[1].bw = T.BW_HZ[(b[0] >> 4) & 3]
        self.f[2].bw = T.BW_HZ[(b[0] >> 2) & 3]
        self.f[3].bw = T.BW_HZ[b[0] & 3]
        self.f[3].fm = T.FM4_HZ
        self.f[2].fm = T.FM3_HZ[b[1] >> 5]
        self.f[1].fm = T.FM2_HZ[b[1] & 0x1F]
        self.f[0].fm = T.FM1_HZ[b[2] >> 3]
        self.ampl = T.AMPL_PERMILLE[((b[2] & 7) << 1) | (b[3] >> 7)]
        self.framelog = fd + 6 + 3
        self.framelength = 1 << self.framelog
        self.bufpos = 0
        self.frames_decoded += 1

    def _start_frame(self) -> None:
        self.framepos = 0

    def _stop_frame(self) -> None:
        self.state = State.STOPPED
        self.output = 0

    @property
    def active(self) -> bool:
        return self.state in (State.STARTED, State.SLOWING)

    def tick(self) -> int:
        """Advance one output sample (1 / (8 x F0) s). Returns the DAC value."""
        if not self.active:
            return self.output
        pos = self.framepos % SUPERSAMPLING
        if pos == 0:
            self.lastsample = self.sample
            self.sample = self._compute_sample()
            self.output = self.lastsample
        else:
            self.output = self.lastsample + _div(pos * (self.sample - self.lastsample), SUPERSAMPLING)
        self.framepos += 1
        if self.framepos >= self.framelength:
            self._shift_frame()
            if self.bufpos == 4:
                self._decode_frame()
                self._start_frame()
            elif self.cont:
                self._start_frame()
            elif self.state == State.STARTED:
                self.ampl = 0
                self._start_frame()
                self.state = State.SLOWING
            elif self.state == State.SLOWING:
                self._stop_frame()
        return self.output

    # ------------------------------------------------------------------ CPU interface
    def write(self, offset: int, data: int) -> None:
        data &= 0xFF
        if offset == 0:
            if self.state == State.STOPPED:
                self.pitch = 2 * data
                self.state = State.WAIT_FIRST
                self.bufpos = 0
            elif self.bufpos == 4:
                self.log.append(f"data overflow {data:02x}")
            else:
                self.buf[self.bufpos] = data
                self.bufpos += 1
                if self.bufpos == 4 and self.state == State.WAIT_FIRST:
                    old_pitch = self.pitch
                    self.last_pitch = old_pitch
                    self._decode_frame()
                    self._shift_frame()
                    self.last_pitch = old_pitch
                    if self.model.first_frame_fade_fix:
                        self.last_ampl = 0
                    else:
                        self.ampl = 0
                    self._start_frame()
                    self.state = State.STARTED
        elif offset == 1:
            if data & 8:
                self.cont = (data >> 2) & 1
            if data & 2:
                self.roe = data & 1
            if data & 0x10:
                self._stop_frame()
        else:
            raise ValueError(f"invalid offset {offset}")


# ---------------------------------------------------------------------- feeding policies

def render(utterances: list[Utterance], model: Model = DEFAULT, policy: str = "chunk",
           noise: np.ndarray | None = None, chip: Chip | None = None, progress=None) -> np.ndarray:
    """Feed a stream to a chip and return int16 samples at model.sample_rate.

    Policies:
      ``chunk``   — the chunk protocol of the encoder output: STOP, pitch, every frame of
                    the utterance (AMPL = 0 frames are silence), STOP once the last frame
                    has started (it is the dummy frame of TP101 fig. 19).
      ``philips`` — TP101 fig. 28: STOP, pitch, frames, then let the chip slow-stop.
    For the legacy Java feeding see `render_java_like`.
    """
    if policy not in ("chunk", "philips"):
        raise ValueError(f"unknown policy {policy!r}")
    chip = chip or Chip(model, noise)
    out: list[int] = []

    def wait_ready() -> None:
        while not chip.accept_byte():
            out.append(chip.tick())

    total = max(1, sum(len(u.frames) for u in utterances))
    done = 0
    for utt in utterances:
        wait_ready()
        chip.write(1, T.CMD_POWER_ON)
        chip.write(0, utt.pitch)
        for frame in utt.frames:
            done += 1
            if progress is not None and done % 16 == 0:
                progress(done / total)
            for byte in frame.to_bytes():
                wait_ready()
                chip.write(0, byte)
            wait_ready()
        if policy == "chunk":
            chip.write(1, T.CMD_POWER_ON)
        else:
            while chip.active:
                out.append(chip.tick())
    return np.asarray(out, dtype=np.int16)


def render_java_like(utterances: list[Utterance], noise: np.ndarray | None = None) -> np.ndarray:
    """Exact replay of legacy `Mea8000Device.compute()`: a fresh chip per chunk."""
    chunks = []
    for utt in utterances:
        chip = Chip(LEGACY_JAVA, noise)
        chunks.append(_render_java_chunk(chip, utt))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


def _render_java_chunk(chip: Chip, utt: Utterance) -> np.ndarray:
    out: list[int] = []
    chip.write(1, T.CMD_POWER_ON)
    chip.write(0, utt.pitch)
    for frame in utt.frames:
        for byte in frame.to_bytes():
            chip.write(0, byte)
        if chip.framepos > 0:
            chip._shift_frame()
            chip._decode_frame()
            chip._start_frame()
        # the Java loop renders framelength samples without the end-of-frame bookkeeping
        while chip.framepos < chip.framelength:
            pos = chip.framepos % SUPERSAMPLING
            if pos == 0:
                chip.lastsample = chip.sample
                chip.sample = chip._compute_sample()
                chip.output = chip.lastsample
            else:
                chip.output = chip.lastsample + _div(pos * (chip.sample - chip.lastsample), SUPERSAMPLING)
            out.append(chip.output)
            chip.framepos += 1
    return np.asarray(out, dtype=np.int16)


def with_model(model: Model, **changes) -> Model:
    return replace(model, **changes)
