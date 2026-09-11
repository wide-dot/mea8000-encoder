"""The filters of a profile, applied to the recording before it is encoded: the channel
choice, the high-pass, the normalization, the trim. They produce the recording the
encoder actually sees — `encode --report` writes it beside the output (`-source.wav`)
and the report compares the chip against it, on the same time axis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import tuning
from .wav import AudioError, select_channel

FULL_SCALE = 32767.0
LEAD_SLOTS = 3        # 24 ms kept before the first sound
TAIL_SLOTS = 6        # 48 ms kept after the last one


@dataclass(frozen=True)
class Prepared:
    """The recording after the filters, and what they did."""

    x: np.ndarray             # mono, int16 scale, the recording's rate
    rate: int
    channel: str              # "mono", "left channel", ...
    gain_db: float | None     # the normalization applied (None: off)
    trim: tuple[float, float] | None   # what was kept, in seconds of the recording (None: off)

    @property
    def seconds(self) -> float:
        return len(self.x) / self.rate

    def describe(self) -> str:
        parts = [self.channel]
        if self.gain_db is not None:
            parts.append(f"normalized {self.gain_db:+.1f} dB")
        if self.trim is not None:
            parts.append(f"trimmed to {self.seconds:.1f} s ({self.trim[0]:.2f}-{self.trim[1]:.2f} s)")
        return ", ".join(parts)


def highpass(x: np.ndarray, rate: int, hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    return sosfiltfilt(butter(4, hz, btype="highpass", fs=rate, output="sos"), x)


def slot_energy_db(x: np.ndarray, rate: int) -> np.ndarray:
    """RMS per 8 ms slot, dB re full scale."""
    hop = max(1, int(round(rate * 8 / 1000)))
    n = int(np.ceil(len(x) / hop))
    padded = np.zeros(n * hop)
    padded[: len(x)] = x
    rms = np.sqrt(np.mean(padded.reshape(n, hop) ** 2, axis=1))
    return 20 * np.log10(rms / 32768.0 + 1e-12)


def speech_bounds(x: np.ndarray, rate: int) -> tuple[int, int] | None:
    """First and last sample of what rises above the silence threshold (the encoder's:
    `tuning.silence_db`, raised above the recording's noise floor), two slots in a row so
    that a click does not count. None when nothing does."""
    K = tuning.current
    e = slot_energy_db(x, rate)
    floor = float(np.percentile(e, 10))
    threshold = max(K.silence_db, floor + K.noise_margin_db) if floor > -100 else K.silence_db
    on = e > threshold
    pairs = np.flatnonzero(on[:-1] & on[1:]) if len(on) > 1 else np.flatnonzero(on)
    if len(pairs) == 0:
        return None
    hop = max(1, int(round(rate * 8 / 1000)))
    first = max(0, int(pairs[0]) - LEAD_SLOTS) * hop
    last = min(len(e), int(pairs[-1]) + 2 + TAIL_SLOTS) * hop
    return first, min(last, len(x))


def prepare(prof, x: np.ndarray, rate: int) -> Prepared:
    """The recording as the encoder will see it: one channel, the profile's high-pass,
    normalized to full scale, the silence before and after the speech dropped."""
    mono, which = select_channel(x, prof.channel)
    y = np.asarray(mono, dtype=np.float64)
    if prof.highpass_hz:
        y = highpass(y, rate, prof.highpass_hz)
    gain_db = None
    if prof.normalize:
        peak = float(np.abs(y).max())
        if peak <= 0:
            raise AudioError("the recording is silent")
        gain_db = 20 * np.log10(FULL_SCALE / peak)
        y = y * (FULL_SCALE / peak)
    trim = None
    if prof.trim:
        with tuning.override(**prof.tuning):
            bounds = speech_bounds(y, rate)
        if bounds is None:
            raise AudioError(f"no speech found: nothing rises above the silence threshold ({tuning.current.silence_db:g} dB FS)")
        a, b = bounds
        trim = (a / rate, b / rate)
        y = y[a:b]
    return Prepared(np.clip(y, -32768, 32767), int(rate), which, gain_db, trim)


def convert(prof, x: np.ndarray, rate: int, progress=None):
    """Filters, then the encoder: (Prepared, speech files, EncodeResult). The pins of the
    profile must fall inside the filtered recording."""
    from .profile import encode_profile

    prepared = prepare(prof, x, rate)
    for p in prof.pins:
        if p.start >= prepared.seconds:
            raise ValueError(f"pin from {p.start:g} to {p.end:g} s is past the end of the recording "
                             f"({prepared.seconds:.2f} s after the filters)")
    utts, result = encode_profile(prof, prepared.x, prepared.rate, progress=progress, filtered=True)
    return prepared, utts, result
