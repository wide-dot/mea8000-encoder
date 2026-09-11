# The profile file

A profile is a TOML file that says how a recording is converted. `mea8000 encode` takes
one with `--profile`; without it, the shipped `thomson` profile applies. Three profiles
are shipped (`mea8000 profiles`): `thomson`, `philips`, `compact`. A profile you write is
the same kind of file; there is no inheritance between profiles: **every profile carries
every value of the top level**.

The file has three parts: the values (always), `[[pin]]` tables (decisions on passages of
one recording), a `[tuning]` table (the encoder's internal constants). A profile with
pins belongs to one recording, since the pins name its seconds; name it after the
recording (`voice.toml` beside `voice.wav`). `mea8000 profiles thomson` prints the
shipped file, to start yours from.

## Values

| key | values | meaning |
|---|---|---|
| `clock_hz` | integer | the chip's clock on the target machine. `4000000` for the Thomson Cedic-Nathan box, `3840000` for Philips' reference. A chip clocked faster than 3.84 MHz plays every frame shorter and every frequency higher; the encoder compensates so that the machine says what the recording says. |
| `format` | `"speech"`, `"vocabulary"` | `speech`: speech files one after the other, one per word group. `vocabulary`: a vocabulary image — an offset table, then the speech files, each reachable by its number. See `docs/formats/`. |
| `pitch` | `"local"`, `"global"` | `local`: each word group gets its own starting pitch and is its own speech file. `global`: one starting pitch for the whole recording; through the pauses the pitch glides towards the next word; a single speech file. |
| `frame_ms` | `8`, `16`, `32`, `64` | the shortest frame allowed. `8` is the chip's finest grain. With `64` every frame lasts 64 ms whatever the quality. Only where noise meets voice inside a frame, or at the end of a file, can a frame be shorter. |
| `quality` | `"best"`, `"balanced"`, `"compact"` | how much degradation is accepted to lengthen frames beyond `frame_ms`: `best` none; `balanced` a little (about 40 % fewer bytes); `compact` what stays intelligible (about 60 % fewer). |
| `channel` | `"left"`, `"right"`, `"mix"` | which channel of a stereo recording to encode; `mix` averages them. A mono recording ignores it. |
| `highpass_hz` | number, 0 to 3999 | a high-pass filter on the recording before analysis; `0` for none. 80-150 Hz removes rumble and kick drums; useless on clean speech. |
| `normalize` | `true`, `false` | scale the recording so that its loudest sample reaches full scale. The chip's amplitude codes are absolute: without it, a recording peaking at -40 dB comes out as silence. |
| `trim` | `true`, `false` | drop the silence before the first sound and after the last (what stays under the silence threshold: -55 dB re full scale, raised above the recording's noise floor). Kept, that silence costs silent frames and a wait before the word. |

The last four are the filters, applied in that order; they produce the recording the
encoder actually sees, which `encode --report` writes beside the output as
`-source.wav`. A silent recording, or one where nothing rises above the threshold, is
refused.

The command line overrides any of them: `--format`, `--pitch`, `--frame`, `--quality`,
`--channel`, `--highpass`, `--normalize` / `--no-normalize`, `--trim` / `--no-trim`.

## Pins: decisions on passages

A `[[pin]]` table names a passage of the recording, `from` and `to` in seconds, and one or
more decisions. The encoder applies them and works out the rest around them. The seconds
are those of the recording after the filters — the `-source.wav` of the report, whose
time axis the report shows; with `trim` on, they are not the seconds of the original
file. A pin that starts past the end of the recording is an error.

```toml
[[pin]]
from = 56.6
to = 56.9
voicing = "unvoiced"    # "silence": silent frames; "unvoiced": noise; "voiced": a pitch

[[pin]]
from = 12.0
to = 14.5
highpass_hz = 200       # a high-pass on this passage only
gain_db = -6            # an amplitude offset on this passage, in dB
silence_db = -50        # the level (dB re full scale) under which this passage is silence

[[pin]]
from = 3.2
to = 3.5
pitch_hz = 180          # a forced pitch on this passage
new_file = true         # a new speech file starts at `from` (false: none starts inside)
```

| key | values | meaning |
|---|---|---|
| `from`, `to` | seconds | the passage, on the filtered recording's time; `to` must be after `from` |
| `voicing` | `"silence"`, `"unvoiced"`, `"voiced"` | what the frames of the passage are: silent, noise-excited, or pitched |
| `pitch_hz` | number | the pitch of the passage's voiced frames |
| `new_file` | `true`, `false` | `true`: a new speech file (with its own starting pitch) begins at `from`; `false`: no new file begins inside the passage |
| `highpass_hz` | number, 0 to 3999 | a high-pass on the passage before analysis |
| `gain_db` | number | an amplitude offset on the passage |
| `silence_db` | number | the silence threshold on the passage (the default is -55 dB re full scale, raised above the recording's noise floor) |

A pin must decide at least one thing. Pins may overlap; the later one wins where they do.

## Tuning: the encoder's constants

The `[tuning]` table sets internal constants of the analysis and the encoder — costs,
weights, thresholds — that were calibrated on a corpus and sit on a plateau. They are not
options; the table exists for experiments. The keys are the fields of
`mea8000/tuning.py`, with their meaning beside each; an unknown key is an error.

```toml
[tuning]
knee_octave = 0.25      # formant movement per 8 ms above which the smoothing resists hard
noise_margin_db = 3.0   # silence threshold above the recording's noise floor
```

## Writing a profile by program

The file is plain TOML: scalars at the top level, an array of tables `pin`, a table
`tuning`. Any TOML writer produces it; `mea8000` reads it with Python's `tomllib`. A
minimal valid profile is the nine values, all of them: there are no defaults to fall back
on. A profile is invalid if a value is missing or outside its set, if a pin decides
nothing or has `to` before `from`, or if a key is unknown.
