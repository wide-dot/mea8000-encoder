"""End to end on the samples: the command line, the formats, the report, and the
compiled simulator against the reference one."""

from pathlib import Path

import numpy as np
import pytest

from mea8000 import codec, sim, wav
from mea8000.cli import main
from mea8000.fastsim import HAVE_NUMBA, render_fast
from mea8000.filters import convert
from mea8000.profile import encode_profile, load

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted(p for p in ROOT.glob("samples/*.wav") if not p.name.endswith(("-chip.wav", "-source.wav")))


@pytest.mark.parametrize("path", SAMPLES, ids=[p.stem for p in SAMPLES])
def test_encode_matches_the_shipped_data_and_renders(path):
    x, rate = wav.read_wav(path)
    prepared, utts, _ = convert(load("thomson"), x, rate)
    data = codec.build_stream(utts, "speech")
    assert data == path.with_suffix(".mea").read_bytes()
    assert codec.detect_layout(data) == "speech"
    # the filters: normalized to full scale, the silence around the sentence dropped
    assert abs(np.abs(prepared.x).max() - 32767) < 1 and prepared.seconds < len(x) / rate
    source, _ = wav.read_wav(path.with_name(path.stem + "-source.wav"))
    assert len(source) == len(prepared.x)
    y = render_fast(utts, sim.with_model(sim.DEFAULT, clock_hz=4_000_000), "chunk")
    assert abs(len(y) / 66664 - prepared.seconds) < 0.5
    assert np.abs(y).max() > 1000


def test_command_line(tmp_path):
    src = SAMPLES[0]
    out = tmp_path / "voice"
    assert main(["encode", str(src), str(out), "--format", "vocabulary", "--frame", "16", "--report"]) == 0
    image = out.with_name("voice.voc.mea")
    assert image.exists() and codec.detect_layout(image.read_bytes()) == "vocabulary"
    assert (tmp_path / "voice.html").exists() and (tmp_path / "voice-chip.wav").exists()
    assert (tmp_path / "voice-source.wav").exists()
    assert main(["render", str(image), str(tmp_path / "voice.wav"), "--rate", "16000"]) == 0
    assert main(["inspect", str(image), "--summary"]) == 0
    assert main(["profiles"]) == 0 and main(["profiles", "thomson"]) == 0
    # refused, with a message and exit code 2, never a traceback
    assert main(["encode", str(tmp_path / "nothing.wav"), str(tmp_path / "x.mea")]) == 2
    assert main(["encode", str(src), str(src)]) == 2                                   # over the input
    assert main(["encode", str(src), str(tmp_path / "x.wav")]) == 2                    # under a .wav name
    assert main(["encode", str(src), str(tmp_path / "no/such/dir/x")]) == 2
    assert main(["encode", str(src), str(tmp_path / "x.voc.mea")]) == 2               # name vs format
    assert main(["encode", str(src), str(tmp_path / "x"), "--highpass", "5000"]) == 2
    assert main(["render", str(image), str(image)]) == 2
    assert main(["render", str(image), str(tmp_path / "x.wav"), "--rate", "0"]) == 2
    assert main(["render", str(src), str(tmp_path / "x.wav")]) == 2                    # a WAV is not speech data
    silent = tmp_path / "silent.wav"
    wav.write_wav(silent, np.zeros(8000), 8000)
    assert main(["encode", str(silent), str(tmp_path / "s")]) == 2
    assert main(["encode", str(silent), str(tmp_path / "s"), "--no-normalize", "--no-trim"]) == 0


def test_profile_file(tmp_path):
    p = load("compact").with_overrides(quality="balanced")
    path = tmp_path / "mine.toml"
    p.save(path)
    assert load(path).quality == "balanced" and load(path).pitch == "global"
    (tmp_path / "partial.toml").write_text('clock_hz = 4000000\nformat = "speech"\n')
    with pytest.raises(ValueError, match="lacks"):
        load(tmp_path / "partial.toml")


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_fast_simulator_is_bit_exact():
    x, rate = wav.read_wav(SAMPLES[0])
    utts, _ = encode_profile(load("philips"), x[: rate * 2], rate)
    assert np.array_equal(sim.render(utts, sim.DEFAULT, "chunk"), render_fast(utts, sim.DEFAULT, "chunk"))
