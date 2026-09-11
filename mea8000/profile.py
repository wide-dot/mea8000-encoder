"""Profiles: how a recording is converted, in one TOML file.

A profile says for which machine (its clock), in which format, with which pitch policy,
frame length, quality, channel and filters — and, when it has been tuned for one
recording, what was decided on given passages (`[[pin]]` tables) and, for whoever
experiments, the encoder's internal constants (`[tuning]`). Every profile carries every
top-level value: there is no inheritance. Three are shipped (`mea8000/profiles/`):
`thomson` (the default), `philips`, `compact`.

```toml
clock_hz = 4000000     # the chip's clock on the target machine
format = "speech"      # "speech": speech files one after the other; "vocabulary": an image
pitch = "local"        # "local": a starting pitch per word group; "global": one for all
frame_ms = 8           # the shortest frame allowed: 8, 16, 32 or 64
quality = "best"       # "best", "balanced", "compact": tolerance for longer frames
channel = "left"       # "left", "right", "mix": which channel of a stereo recording
highpass_hz = 0        # a high-pass on the recording before analysis, 0 = none
normalize = true       # scale the recording so that its peak reaches full scale
trim = true            # drop the silence before the first sound and after the last

[[pin]]                # what was decided on a passage of this recording (seconds)
from = 56.6
to = 56.9
voicing = "unvoiced"   # "silence", "unvoiced" or "voiced"

[[pin]]
from = 12.0
to = 14.5
highpass_hz = 200      # a high-pass on this passage only
gain_db = -6           # an amplitude offset on this passage
pitch_hz = 180         # a forced pitch
silence_db = -50       # the silence threshold on this passage
new_file = true        # a new speech file starts at `from` (false: none inside)

[tuning]               # the encoder's internal constants (see mea8000/tuning.py)
knee_octave = 0.25
```

`docs/profile.md` of the release repository documents the file for users.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from .encoder import HOP, RATE, Constraints

PHILIPS_CLOCK = 3_840_000
SLOT_MS = 1000.0 * HOP / RATE   # 8 ms
FORMATS = ("speech", "vocabulary")
PITCHES = ("local", "global")
FRAMES_MS = (8, 16, 32, 64)
QUALITY_PRICE_DB = {"best": 0.0, "balanced": 2.0, "compact": 8.0}
CHANNELS = ("left", "right", "mix")
VOICINGS = ("silence", "unvoiced", "voiced")
PIN_KEYS = ("from", "to", "voicing", "pitch_hz", "new_file", "highpass_hz", "gain_db", "silence_db")

SHIPPED = Path(__file__).parent / "profiles"
HIGHPASS_MAX_HZ = 3999   # the analysis runs at 8 kHz


def _check_highpass(hz, what: str) -> None:
    if not (0 <= float(hz) <= HIGHPASS_MAX_HZ):
        raise ValueError(f"{what} must be between 0 and {HIGHPASS_MAX_HZ} Hz, not {hz:g}")


@dataclass(frozen=True)
class Pin:
    """What was decided on a passage: from `start` to `end` seconds of the recording."""

    start: float
    end: float
    voicing: str | None = None       # "silence" | "unvoiced" | "voiced"
    pitch_hz: float | None = None    # a forced pitch
    new_file: bool | None = None     # True: a new speech file starts at `start`; False: none inside
    highpass_hz: float | None = None
    gain_db: float | None = None
    silence_db: float | None = None

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"pin from {self.start} to {self.end}: 'to' must be after 'from'")
        if self.voicing is not None and self.voicing not in VOICINGS:
            raise ValueError(f"pin voicing must be one of {VOICINGS}, not {self.voicing!r}")
        if self.highpass_hz is not None:
            _check_highpass(self.highpass_hz, f"pin from {self.start} to {self.end}: highpass_hz")
        if all(getattr(self, k) is None for k in ("voicing", "pitch_hz", "new_file", "highpass_hz", "gain_db", "silence_db")):
            raise ValueError(f"pin from {self.start} to {self.end} decides nothing")

    def slots(self, n: int, rate_scale: float = 1.0) -> slice:
        """The 8 ms slots covered (with a clock compensation the slot grid runs on the
        recording's time scaled by `rate_scale`)."""
        a = int(np.floor(1000 * self.start / rate_scale / SLOT_MS))
        b = int(np.ceil(1000 * self.end / rate_scale / SLOT_MS))
        return slice(max(0, a), min(n, max(b, a + 1)))

    def as_dict(self) -> dict:
        d = {"from": self.start, "to": self.end}
        for k in ("voicing", "pitch_hz", "new_file", "highpass_hz", "gain_db", "silence_db"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Pin":
        unknown = set(d) - set(PIN_KEYS)
        if unknown:
            raise ValueError(f"unknown pin keys: {sorted(unknown)}")
        return cls(start=float(d["from"]), end=float(d["to"]), voicing=d.get("voicing"),
                   pitch_hz=d.get("pitch_hz"), new_file=d.get("new_file"), highpass_hz=d.get("highpass_hz"),
                   gain_db=d.get("gain_db"), silence_db=d.get("silence_db"))


@dataclass(frozen=True)
class Profile:
    name: str = "thomson"
    clock_hz: int = 4_000_000
    format: str = "speech"
    pitch: str = "local"
    frame_ms: int = 8
    quality: str = "best"
    channel: str = "left"
    highpass_hz: float = 0.0
    normalize: bool = True
    trim: bool = True
    pins: tuple[Pin, ...] = ()
    tuning: dict = field(default_factory=dict)

    VALUE_KEYS = ("clock_hz", "format", "pitch", "frame_ms", "quality", "channel", "highpass_hz", "normalize", "trim")

    def __post_init__(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, not {self.format!r}")
        if self.pitch not in PITCHES:
            raise ValueError(f"pitch must be one of {PITCHES}, not {self.pitch!r}")
        if self.frame_ms not in FRAMES_MS:
            raise ValueError(f"frame_ms must be one of {FRAMES_MS}, not {self.frame_ms!r}")
        if self.quality not in QUALITY_PRICE_DB:
            raise ValueError(f"quality must be one of {tuple(QUALITY_PRICE_DB)}, not {self.quality!r}")
        if self.channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, not {self.channel!r}")
        if self.clock_hz <= 0:
            raise ValueError("clock_hz must be positive")
        _check_highpass(self.highpass_hz, "highpass_hz")
        if not isinstance(self.normalize, bool) or not isinstance(self.trim, bool):
            raise ValueError("normalize and trim must be true or false")
        if self.tuning:
            from . import tuning as tuning_module

            known = {f.name for f in fields(tuning_module.Tuning)}
            unknown = set(self.tuning) - known
            if unknown:
                raise ValueError(f"unknown tuning keys: {sorted(unknown)}")

    # ------------------------------------------------------------ derived
    @property
    def price_db(self) -> float:
        return QUALITY_PRICE_DB[self.quality]

    @property
    def min_slots(self) -> int:
        return self.frame_ms // 8

    @property
    def single(self) -> bool:
        return self.pitch == "global"

    @property
    def rate_scale(self) -> float:
        """Source rate scale that compensates the chip clock: a chip clocked faster than
        Philips' reference plays every frame shorter and every frequency higher, so the
        recording is analysed as if it were that much slower and lower."""
        return PHILIPS_CLOCK / self.clock_hz

    @property
    def extension(self) -> str:
        return ".mea" if self.format == "speech" else ".voc.mea"

    def with_overrides(self, **values) -> "Profile":
        values = {k: v for k, v in values.items() if v is not None}
        return replace(self, **values) if values else self

    def with_pins(self, pins) -> "Profile":
        return replace(self, pins=tuple(pins))

    # ------------------------------------------------------------ constraints
    def constraints(self, n_slots: int) -> Constraints:
        c = Constraints.empty(n_slots)
        for p in self.pins:
            s = p.slots(n_slots, self.rate_scale)
            if p.highpass_hz is not None:
                c.highpass_hz[s] = p.highpass_hz
            if p.silence_db is not None:
                c.silence_db[s] = p.silence_db
            if p.gain_db is not None:
                c.ampl_db[s] += p.gain_db
            if p.voicing == "silence":
                c.force[s] = Constraints.SILENCE
            elif p.voicing == "unvoiced":
                c.force[s] = Constraints.UNVOICED
            elif p.voicing == "voiced":
                c.force[s] = Constraints.VOICED
            if p.pitch_hz is not None:
                c.pitch_hz[s] = float(p.pitch_hz)
            if p.new_file is True:
                c.restart[s.start] = 1
            elif p.new_file is False:
                c.restart[s] = -1
        return c

    # ------------------------------------------------------------ files
    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.VALUE_KEYS}
        if self.pins:
            d["pin"] = [p.as_dict() for p in self.pins]
        if self.tuning:
            d["tuning"] = dict(self.tuning)
        return d

    def dump(self) -> str:
        """The profile as TOML (a small emitter: the file has only scalars, [[pin]] tables
        and a [tuning] table)."""
        def scalar(v):
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            if isinstance(v, (float, np.floating)):
                return repr(float(v)) if float(v) != int(v) else f"{int(v)}.0"
            return '"' + str(v).replace('\\', '\\\\').replace('"', '\\"') + '"'

        lines = []
        for k in self.VALUE_KEYS:
            lines.append(f"{k} = {scalar(getattr(self, k))}")
        for p in self.pins:
            lines.append("")
            lines.append("[[pin]]")
            for k, v in p.as_dict().items():
                lines.append(f"{k} = {scalar(v)}")
        if self.tuning:
            lines.append("")
            lines.append("[tuning]")
            for k, v in self.tuning.items():
                lines.append(f"{k} = {scalar(v)}")
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.dump())

    def key(self) -> str:
        import json

        return json.dumps(self.as_dict(), sort_keys=True)


def parse(text: str, name: str = "profile") -> Profile:
    d = tomllib.loads(text)
    pins = tuple(Pin.from_dict(p) for p in d.pop("pin", []))
    tuning = dict(d.pop("tuning", {}))
    d.pop("name", None)
    unknown = set(d) - set(Profile.VALUE_KEYS)
    if unknown:
        raise ValueError(f"unknown profile keys: {sorted(unknown)} (known: {', '.join(Profile.VALUE_KEYS)})")
    missing = [k for k in Profile.VALUE_KEYS if k not in d]
    if missing:
        raise ValueError(f"profile {name} lacks {', '.join(missing)} (every profile carries every value; "
                         f"`mea8000 profiles thomson` prints one to start from)")
    return Profile(name=name, pins=pins, tuning=tuning, **d)


def path_of(name_or_path: str | Path = "thomson") -> Path:
    """The file of a shipped profile by name, or any TOML file by path."""
    p = Path(name_or_path)
    if not p.suffix and not p.exists():
        p = SHIPPED / f"{name_or_path}.toml"
    if not p.exists():
        raise FileNotFoundError(f"no profile {name_or_path!r} (shipped: {', '.join(names())})")
    return p


def load(name_or_path: str | Path = "thomson") -> Profile:
    """A shipped profile by name, or any TOML file by path."""
    p = path_of(name_or_path)
    return parse(p.read_text(), name=p.stem)


def names() -> list[str]:
    return sorted(p.stem for p in SHIPPED.glob("*.toml"))


# ------------------------------------------------------------------ encoding

def encode_profile(profile: Profile, x: np.ndarray, rate: int, cache: dict | None = None,
                   progress=None, filtered: bool = False):
    """Run the encoder on `x` under the profile; returns (speech files, EncodeResult).
    `filtered`: `x` already went through the profile's filters (`filters.prepare`), so the
    high-pass is not applied again; the product path (`filters.convert`) does that, the
    laboratory's bench and editor feed the raw recording."""
    from . import tuning as tuning_module
    from .encoder import encode
    from .merge import merge

    n = int(np.ceil(len(x) * RATE / (rate * profile.rate_scale) / HOP))
    with tuning_module.override(**profile.tuning):
        result = encode(x, rate, highpass_hz=None if filtered else (profile.highpass_hz or None), single=profile.single,
                        constraints=profile.constraints(n), cache=cache, rate_scale=profile.rate_scale,
                        progress=progress)
        if profile.price_db > 0 or profile.min_slots > 1:
            utts = merge(result, profile.price_db, min_slots=profile.min_slots)
        else:
            utts = result.utterances
    return utts, result
