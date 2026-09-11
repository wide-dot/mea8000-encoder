"""WAV in and out: any PCM or float WAV in, 16-bit PCM out."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class AudioError(ValueError):
    """A recording that cannot be read, with a message meant for the user."""


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Samples (int16 scale, one row per frame, one column per channel — 1-D when mono)
    and rate. Accepts 8/16/24/32-bit PCM and 32/64-bit float WAV, any channel count."""
    from scipy.io import wavfile

    path = Path(path)
    if not path.exists():
        raise AudioError(f"{path}: no such file")
    try:
        with open(path, "rb") as f:
            head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise AudioError(f"{path.name}: not a WAV file (MP3, M4A and FLAC are not read; convert to WAV first)")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # non-audio chunks (metadata) are skipped, silently
            rate, data = wavfile.read(str(path))
    except AudioError:
        raise
    except Exception as e:  # a compressed or exotic WAV
        raise AudioError(f"{path.name}: unsupported WAV encoding ({e}); export as PCM") from e
    if data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) * 256.0
    elif data.dtype == np.int16:
        x = data.astype(np.float64)
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 65536.0          # 24-bit WAVs arrive left-aligned in int32
    elif data.dtype.kind == "f":
        x = data.astype(np.float64) * 32768.0
    else:
        raise AudioError(f"{path.name}: unsupported sample type {data.dtype}")
    if len(x) == 0:
        raise AudioError(f"{path.name}: empty recording")
    return x, int(rate)


def describe(x: np.ndarray, rate: int, path: str | Path, dtype_note: str = "") -> str:
    ch = 1 if x.ndim == 1 else x.shape[1]
    kind = {1: "mono", 2: "stereo"}.get(ch, f"{ch} channels")
    return f"{Path(path).name}: {rate} Hz, {kind}, {len(x) / rate:.1f} s"


def select_channel(x: np.ndarray, channel: str) -> tuple[np.ndarray, str]:
    """One channel of a recording: left (channel 1), right (channel 2) or mix (the mean).
    Mono recordings pass through. Returns the signal and a word for the output line."""
    if x.ndim == 1:
        return x, "mono"
    if channel == "mix":
        return x.mean(axis=1), "mix of the channels"
    if channel == "right":
        return x[:, 1] if x.shape[1] > 1 else x[:, 0], "right channel"
    return x[:, 0], "left channel"


def write_wav(path: str | Path, samples: np.ndarray, rate: int) -> None:
    import wave

    samples = np.asarray(samples)
    if samples.dtype != np.int16:
        samples = np.clip(np.round(samples), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(samples.tobytes())


def read_raw_i16le(path: str | Path) -> np.ndarray:
    return np.fromfile(str(path), dtype="<i2").astype(np.int16)
