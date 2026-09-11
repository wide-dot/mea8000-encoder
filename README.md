# mea8000-encoder

A recording in, MEA8000 speech data out — in the period file formats — with a preview
of what the chip will say. The MEA8000 is the formant speech synthesizer of the 1980s
(Philips / Signetics) found in the Cedic-Nathan speech box of Thomson computers and in
other machines of the time.

## Try it in two minutes

Python 3.12 or later, on macOS, Linux or Windows:

```
pip install mea8000-encoder
mea8000 encode samples/fr-female.wav fr-female.mea --report
```

`fr-female.mea` is the speech data. `fr-female.html` opens in a browser: the recording
and the chip's rendering side by side, with a button to play each (TAB switches while
playing), the chip's resonators drawn over its spectrogram, the pitch, level and voicing
lanes, and the frames. Beside it,
`fr-female-source.wav` is the recording as the encoder saw it (one channel, normalized,
the silence around the speech dropped — see the profile) and `fr-female-chip.wav` the
rendering alone.

`samples/` holds three sentences from Mozilla Common Voice, their speech data and their
reports.

## The commands

| command | what |
|---|---|
| `mea8000 encode SOURCE.wav OUTPUT.mea [--profile P] [options]` | encode a recording (any WAV: any rate, bit depth, mono or stereo); `--report` for the page and the two WAVs |
| `mea8000 render INPUT.mea OUTPUT.wav [--profile P]` | hear what the chip says (a WAV at the chip's own rate; `--rate HZ` to resample) |
| `mea8000 inspect INPUT.mea [--summary]` | what a file holds: format, word groups, frames |
| `mea8000 profiles [NAME]` | the shipped profiles; with a name, that profile's file, to start yours from |

Both formats are recognised from the file's content (`--format` forces one). Encoding a
30 s recording takes about 20 s; a percentage shows the progress. `python -m mea8000` is
the same command.

## The profile

A profile says how to convert: for which machine, in which format, with which pitch
policy, frame length and quality, and through which filters (channel, high-pass,
normalization, trim). It is one small TOML file.
Three are shipped — `thomson` (the default), `philips`, `compact` — and you can write
your own: `--profile mine.toml`. Every value can also be overridden on the command line.

```toml
clock_hz = 4000000       # the box clocks the chip at 4 MHz: everything would play 4.2 %
                         # faster and higher than Philips' 3.84 MHz reference; the encoder
                         # compensates so that the machine says what the recording says
format = "speech"        # "speech": speech files one after the other; "vocabulary": an
                         # image with an offset table, each file reachable by its number
pitch = "local"          # "local": a starting pitch per word group, each its own speech
                         # file; "global": one for the whole recording, gliding through
                         # the pauses, in a single speech file
frame_ms = 8             # the shortest frame allowed: 8 (the chip's finest), 16, 32, 64
quality = "best"         # how much degradation is accepted to lengthen frames beyond
                         # frame_ms: "best" none, "balanced" a little (about 40 % fewer
                         # bytes), "compact" what stays intelligible (about 60 % fewer)
channel = "left"         # which channel of a stereo recording: "left", "right", "mix"
highpass_hz = 0          # a high-pass on the recording before analysis (80-150 removes
                         # rumble and kick drums); 0: none
normalize = true         # scale the recording so that its loudest sample reaches full
                         # scale (the chip's amplitude codes are absolute: a quiet
                         # recording would come out as silence)
trim = true              # drop the silence before the first sound and after the last
                         # (they would cost silent frames and a wait before the word)
```

The last four are filters: they produce the recording the encoder actually sees, the one
`--report` writes as `-source.wav` and compares the chip against.

The same file can carry decisions on passages of one recording (`[[pin]]`: force a
passage to silence or to a pitch, filter it, start a new speech file there…) and, for
whoever experiments, the encoder's internal constants (`[tuning]`). Everything a profile
can say is in [docs/profile.md](docs/profile.md), written to be read and written by a
person or a program.

## The formats

`encode` writes the original, official formats of the chip's era: **speech files** (`.mea`,
Philips Technical Publication 101, the format of the Cedic-Nathan data) one after the
other, or a **vocabulary image** (`.voc.mea`, the layout of the Cedic-Nathan cartridges: an
offset table, then the speech files). Both are documented byte by byte in
[docs/formats/](docs/formats/), with the chip's protocol to play them.

## How it works

The chip plays frames of 4 bytes lasting 8 to 64 ms: a pitch step, an amplitude, and
four resonators (three formants and a fixed one) with a bandwidth each, excited by a
sawtooth (voiced sound) or by noise. The encoder analyses the recording every 8 ms
(level, voicing, pitch); for each 8 ms it searches all 2 097 152 resonator settings for
the one whose synthesized spectrum is closest to the recording, smooths the choices over
time so that formants move the way speech does, and picks the amplitude code that
reproduces the level. Silences become silent frames; a pause starts a new word group.

It works on clean speech, spoken or sung, in any language. It does not turn music into
speech. The chip's range is 100-4000 Hz: what is above is lost, what is below (a kick
drum) should be filtered out first (`--highpass`).

## The simulator

`mea8000/sim.py` is a register-level model of the chip derived from Antoine Miné's MAME
device (BSD-3, see [THIRD_PARTY.md](THIRD_PARTY.md)), with two corrections of that device
(the amplitude division that silenced the quietest codes, and the inverted fade-in of
the first frame — both also fixed in dcmoto 2024). `fastsim.py` is the same model compiled
by numba, bit-exact; the plain model already renders 30 s in about 2 s, so the
`[fast]` extra (`pip install mea8000-encoder[fast]`) is a comfort, not a need.

## License

MIT. The samples come from Mozilla Common Voice, CC0 (see [samples/README.md](samples/README.md)).
