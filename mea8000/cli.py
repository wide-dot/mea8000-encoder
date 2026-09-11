"""Command line.

    mea8000 encode voice.wav voice.mea                 # profile thomson
    mea8000 encode voice.wav voice.mea --profile compact
    mea8000 encode voice.wav voice.mea --profile voice.toml --report
    mea8000 render voice.mea voice.wav                 # what the chip says
    mea8000 inspect voice.mea                          # what the file holds
    mea8000 profiles                                   # the shipped profiles
    mea8000 profiles thomson > mine.toml               # one of them, to start from

A profile (`--profile`) says how to convert: thomson (default), philips, compact, or a
TOML file of your own (docs/profile.md). Options override its values. `python -m mea8000`
is the same command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import profile as profiles
from . import sim
from .codec import build_stream, describe_stream, detect_layout, parse_stream
from .filters import convert
from .wav import AudioError, describe, read_wav, write_wav

MODELS = {"default": sim.DEFAULT, "mame-int": sim.MAME_INT, "mame-float": sim.MAME_FLOAT, "java": sim.LEGACY_JAVA}


class Progress:
    """A percentage on one line of stderr, only when stderr is a terminal."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.on = sys.stderr.isatty()
        self.last = -1

    def __call__(self, fraction: float) -> None:
        pct = int(100 * min(1.0, max(0.0, fraction)))
        if self.on and pct != self.last:
            self.last = pct
            sys.stderr.write(f"\r{self.label} {pct:3d}%")
            sys.stderr.flush()

    def done(self) -> None:
        if self.on and self.last >= 0:
            sys.stderr.write("\r" + " " * (len(self.label) + 6) + "\r")
            sys.stderr.flush()


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _profile(args) -> profiles.Profile:
    prof = profiles.load(args.profile)
    return prof.with_overrides(format=getattr(args, "format", None), pitch=getattr(args, "pitch", None),
                               frame_ms=getattr(args, "frame", None), quality=getattr(args, "quality", None),
                               highpass_hz=getattr(args, "highpass", None), channel=getattr(args, "channel", None),
                               normalize=getattr(args, "normalize", None), trim=getattr(args, "trim", None))


def _load_stream(path: str, fmt: str | None):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{path}: no such file")
    data = p.read_bytes()
    layout = fmt or detect_layout(data)
    try:
        return parse_stream(data, layout), layout
    except ValueError as e:
        raise ValueError(f"{path}: not a {'vocabulary image' if layout == 'vocabulary' else 'speech file'} ({e})") from e


def _check_output(output: Path, source: Path, wrong_suffix: str, wrong_what: str) -> str | None:
    """Why the output cannot be written there, or None: never over the input, never under
    the other kind's extension, only in a directory that exists."""
    if output.resolve() == source.resolve():
        return f"{output}: the output is the input"
    if output.suffix.lower() == wrong_suffix:
        return f"{output}: {wrong_what} cannot be written under a {wrong_suffix} name"
    if not output.parent.is_dir():
        return f"{output.parent}: no such directory"
    return None


# ----------------------------------------------------------------- commands

def cmd_encode(args) -> int:
    try:
        prof = _profile(args)
    except (FileNotFoundError, ValueError) as e:
        return fail(str(e))
    output = Path(args.output)
    if not output.suffix:
        output = output.with_name(output.name + prof.extension)
    name = output.name.lower()
    if name.endswith(".voc.mea") and prof.format == "speech":
        return fail(f"{output}: the name says vocabulary image, the format is speech (pass --format vocabulary, or name it .mea)")
    if name.endswith(".mea") and not name.endswith(".voc.mea") and prof.format == "vocabulary":
        return fail(f"{output}: the name says speech files, the format is vocabulary (pass --format speech, or name it .voc.mea)")
    why = _check_output(output, Path(args.source), ".wav", "speech data")
    if why:
        return fail(why)
    try:
        x, rate = read_wav(args.source)
        progress = Progress("encoding")
        prepared, utts, result = convert(prof, x, rate, progress=progress)
        progress.done()
    except (AudioError, ValueError) as e:
        return fail(str(e))
    print(f"{describe(x, rate, args.source)} -> {prepared.describe()}")
    data = build_stream(utts, prof.format)
    output.write_bytes(data)
    what = "vocabulary image" if prof.format == "vocabulary" else "speech file" + ("s" if len(utts) != 1 else "")
    frames = sum(len(u.frames) for u in utts)
    print(f"{output}: {what}, {len(utts)} word group{'s' if len(utts) != 1 else ''}, {frames} frames, {len(data)} bytes "
          f"(profile {prof.name}, {prof.clock_hz / 1e6:.2f} MHz, pitch {prof.pitch}, frame {prof.frame_ms} ms, {prof.quality})")
    if args.report:
        from .report import write_report

        page = output.with_suffix("").with_suffix(".html") if output.name.endswith(".voc.mea") else output.with_suffix(".html")
        source_wav = page.with_name(page.stem + "-source.wav")
        chip_wav = page.with_name(page.stem + "-chip.wav")
        write_report(page, source_wav, chip_wav, Path(args.source).name, prepared, utts, result, prof, len(data))
        print(f"{page}: report ({source_wav.name}, the recording after the filters, and {chip_wav.name} beside it)")
    return 0


def cmd_render(args) -> int:
    from .fastsim import HAVE_NUMBA, render_fast

    try:
        prof = profiles.load(args.profile)
        utts, layout = _load_stream(args.file, args.format)
    except (FileNotFoundError, ValueError) as e:
        return fail(str(e))
    if args.rate is not None and args.rate < 1:
        return fail(f"--rate {args.rate}: a rate is a positive number of Hz")
    why = _check_output(Path(args.output), Path(args.file), ".mea", "a rendering")
    if why:
        return fail(why)
    model = sim.with_model(MODELS[args.model], clock_hz=prof.clock_hz)
    if args.pitch_scale:
        model = sim.with_model(model, pitch_scale=args.pitch_scale)
    if args.truncate_bits:
        model = sim.with_model(model, truncate_bits=args.truncate_bits)
    noise = np.fromfile(args.noise_table, dtype="<i4").astype(np.int64) if args.noise_table else None
    seconds = sum(u.duration_ms() for u in utts) / 1000
    if args.policy == "java-exact":
        samples = sim.render_java_like(utts, noise)
    elif HAVE_NUMBA:
        if sys.stderr.isatty():
            sys.stderr.write("rendering ...\r")
        samples = render_fast(utts, model, args.policy, noise)
    else:
        progress = Progress("rendering")
        samples = sim.render(utts, model, args.policy, noise, progress=progress)
        progress.done()
    rate = model.sample_rate
    if args.rate and args.rate != rate:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(int(args.rate), rate)
        samples = np.clip(resample_poly(samples.astype(np.float64), int(args.rate) // g, rate // g), -32767, 32767)
        rate = int(args.rate)
    write_wav(args.output, samples, rate)
    print(f"{args.output}: {len(samples) / rate:.2f} s at {rate} Hz (chip clock {prof.clock_hz / 1e6:.2f} MHz, "
          f"profile {prof.name}, {layout} of {len(utts)} word group{'s' if len(utts) != 1 else ''})")
    return 0


def cmd_inspect(args) -> int:
    try:
        utts, layout = _load_stream(args.file, args.format)
    except (FileNotFoundError, ValueError) as e:
        return fail(str(e))
    what = "vocabulary image" if layout == "vocabulary" else "speech file" + ("s" if len(utts) != 1 else "")
    print(f"{args.file}: {what}, {len(utts)} word group{'s' if len(utts) != 1 else ''}, "
          f"{sum(len(u.frames) for u in utts)} frames, {sum(u.duration_ms() for u in utts) / 1000:.2f} s")
    if not args.summary:
        print(describe_stream(utts))
    return 0


def cmd_profiles(args) -> int:
    if args.name:
        try:
            print(profiles.path_of(args.name).read_text(), end="")
        except FileNotFoundError as e:
            return fail(str(e))
        return 0
    for name in profiles.names():
        p = profiles.load(name)
        print(f"{name:10s} clock {p.clock_hz / 1e6:.2f} MHz  format {p.format:10s}  pitch {p.pitch:6s}  "
              f"frame {p.frame_ms:2d} ms  quality {p.quality:8s}  channel {p.channel:5s}  highpass {p.highpass_hz:g} Hz  "
              f"normalize {'on' if p.normalize else 'off':3s}  trim {'on' if p.trim else 'off'}")
    return 0


# ----------------------------------------------------------------- parser

class Formatter(argparse.RawDescriptionHelpFormatter):
    pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mea8000", description="A recording in, MEA8000 speech data out.",
                                 formatter_class=Formatter, epilog=__doc__.split("\n", 2)[2])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="encode a recording into MEA8000 speech data",
                       usage="mea8000 encode SOURCE.wav OUTPUT.mea [--profile NAME|FILE] [options]",
                       description="Encode a recording (WAV, any rate, any bit depth, mono or stereo).",
                       formatter_class=Formatter)
    e.add_argument("source", metavar="SOURCE.wav", help="the recording")
    e.add_argument("output", metavar="OUTPUT.mea", help="the output (.mea for speech files, .voc.mea for a vocabulary image;"
                                                        " added when the name has no extension)")
    e.add_argument("--profile", default="thomson", metavar="NAME|FILE",
                   help="how to convert: thomson (default), philips, compact, or a TOML file of your own")
    e.add_argument("--format", choices=profiles.FORMATS,
                   help="speech: speech files one after the other, one per word group (default); "
                        "vocabulary: an image with an offset table, each file reachable by its number")
    e.add_argument("--pitch", choices=profiles.PITCHES,
                   help="local: a starting pitch per word group, each group its own speech file (default); "
                        "global: one starting pitch for the whole recording, gliding through the pauses, in a single file")
    e.add_argument("--frame", type=int, choices=profiles.FRAMES_MS, metavar="{8,16,32,64}",
                   help="the shortest frame allowed, in ms (default 8, the chip's finest grain; 64 makes "
                        "every frame 64 ms whatever the quality; only where noise meets voice inside a frame, "
                        "or at the end of a file, can a frame be shorter)")
    e.add_argument("--quality", choices=tuple(profiles.QUALITY_PRICE_DB),
                   help="how much degradation is accepted to lengthen frames beyond --frame: "
                        "best: none (default); balanced: a little, about 40 %% fewer bytes at --frame; "
                        "compact: what stays intelligible, about 60 %% fewer bytes at --frame")
    e.add_argument("--highpass", type=float, metavar="HZ",
                   help="a high-pass on the recording before analysis (80-150 Hz removes rumble and kick drums)")
    e.add_argument("--channel", choices=profiles.CHANNELS,
                   help="which channel of a stereo recording to encode (default left; a mono file has only one)")
    e.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None,
                   help="scale the recording so that its loudest sample reaches full scale (default on; the chip's "
                        "amplitude codes are absolute, a quiet recording would come out as silence)")
    e.add_argument("--trim", action=argparse.BooleanOptionalAction, default=None,
                   help="drop the silence before the first sound and after the last (default on)")
    e.add_argument("--report", action="store_true",
                   help="also write, next to the output, the recording after the filters (-source.wav), the chip's "
                        "rendering (-chip.wav) and an HTML page (.html): spectrograms of both, pitch, voicing and "
                        "level lanes, the two sounds to play")
    e.set_defaults(func=cmd_encode)

    r = sub.add_parser("render", help="render speech data to a WAV with the chip simulator",
                       description="Hear what the chip says: a WAV at the chip's own rate.", formatter_class=Formatter)
    r.add_argument("file", metavar="INPUT.mea")
    r.add_argument("output", metavar="OUTPUT.wav")
    r.add_argument("--profile", default="thomson", metavar="NAME|FILE",
                   help="the clock of the target machine, taken from the profile: thomson 4 MHz (default), philips 3.84 MHz")
    r.add_argument("--rate", type=int, metavar="HZ",
                   help="resample the WAV to this rate (default: the chip's own, 64000 Hz at 3.84 MHz, 66664 Hz at 4 MHz)")
    r.add_argument("--format", choices=profiles.FORMATS, help="speech or vocabulary; detected from the content when omitted")
    r.add_argument("--model", choices=list(MODELS), default="default", help=argparse.SUPPRESS)
    r.add_argument("--policy", choices=["chunk", "philips", "java-exact"], default="chunk", help=argparse.SUPPRESS)
    r.add_argument("--pitch-scale", type=float, help=argparse.SUPPRESS)
    r.add_argument("--truncate-bits", type=int, help=argparse.SUPPRESS)
    r.add_argument("--noise-table", help=argparse.SUPPRESS)
    r.set_defaults(func=cmd_render)

    i = sub.add_parser("inspect", help="print what a .mea file holds", formatter_class=Formatter)
    i.add_argument("file", metavar="INPUT.mea")
    i.add_argument("--format", choices=profiles.FORMATS, help="speech or vocabulary; detected from the content when omitted")
    i.add_argument("--summary", action="store_true", help="the first line only")
    i.set_defaults(func=cmd_inspect)

    pr = sub.add_parser("profiles", help="list the shipped profiles, or print one",
                        description="Without a name, one line per shipped profile; with one, its file, to start yours from.")
    pr.add_argument("name", nargs="?", metavar="NAME", help="thomson, philips or compact: print that profile's file")
    pr.set_defaults(func=cmd_profiles)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
