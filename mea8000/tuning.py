"""The encoder's tuning: every internal constant of the analysis and the encoder.

These are choices calibrated by the regression bench of the laboratory, not options: a
user of the product never needs them. The functions read `current` at call time, so a
bench can set a value, encode, and reset:

    from mea8000 import tuning
    with tuning.override(knee_octave=0.2):
        ...

A TOML file with a subset of the fields can be loaded with `load(path)` (the `--tuning`
option of the command line) for experiments.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path


@dataclass
class Tuning:
    # --- analysis: voicing and pitch tracking (analysis.py)
    unvoiced_cost: float = 0.55        # cost of the unvoiced state per slot
    switch_cost: float = 0.3           # voiced <-> unvoiced transition
    loud_unvoiced_bonus: float = 0.3   # extra unvoiced cost near the peak level
    octave_weight: float = 1.5         # pitch continuity, per octave of jump
    subharmonic_threshold: float = 0.15
    subharmonic_penalty: float = 0.1   # per octave of lag beyond the first good dip
    hnr_unvoiced: float = 3.0          # demote voiced slots under this HNR (dB)
    min_run: int = 3                   # voicing runs shorter than this are absorbed
    # --- activity (encoder.py)
    silence_db: float = -55.0
    noise_margin_db: float = 3.0       # above the noise floor (sweep 2026-09-11: 6 -> 3, LSD -0.6)
    hysteresis_db: float = 3.0
    min_active: int = 3
    max_gap: int = 4
    # --- pitch plan
    restart_cost: float = 400.0
    onset_restart_cost: float = 60.0
    max_cents: float = 600.0
    # --- filter search
    candidates: int = 64
    carry_db: float = 10.0
    max_candidates: int = 192
    per_triple: int = 3
    per_prev: int = 8
    # --- temporal smoothing
    octave_db: float = 3.0             # sweep 2026-09-11: 6 -> 3, STOI +0.010
    bw_db: float = 0.3                 # sweep 2026-09-11: 0.7 -> 0.3
    knee_octave: float = 0.3
    beyond_knee_db: float = 60.0
    # --- amplitude
    silence_margin_db: float = 6.0
    quiet_db: float = -45.0

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


DEFAULT = Tuning()
current = Tuning()


def set_values(**values) -> None:
    global current
    current = replace(current, **values)


def reset() -> None:
    global current
    current = Tuning()


@contextmanager
def override(**values):
    global current
    saved = current
    current = replace(current, **values)
    try:
        yield current
    finally:
        current = saved


def load(path: str | Path) -> Tuning:
    """A Tuning from a TOML file holding a subset of the fields (unknown keys are errors)."""
    import tomllib

    values = tomllib.loads(Path(path).read_text())
    known = {f.name for f in fields(Tuning)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown tuning keys: {sorted(unknown)}")
    return replace(DEFAULT, **values)
