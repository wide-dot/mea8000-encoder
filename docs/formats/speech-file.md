# The speech file

The format of Philips Technical Publication 101 (p. 14, fig. 19 and 27), used as is by the
Cedic-Nathan speech products for Thomson: the demonstration cartridge, the phoneme sets,
the vocabulary listings of *Parole et Micros*. One speech file says one word group — a
run of speech between two pauses — with its own starting pitch.

| offset | size | content |
|---|---|---|
| 0 | 2 | length of the file in bytes, this word included, big endian |
| 2 | 1 | a copy of the starting pitch (Cedic-Nathan convention; a free byte, not interpreted by the chip, in Philips' format) |
| 3 | 1 | the starting pitch, in Hz / 2 |
| 4 | 4 × n | the frames; the last one has amplitude 0 and lets the previous one play to its end (the "dummy frame" of TP101 fig. 19) |

`mea8000 encode --format speech` writes the word groups' files one after the other; a
player walks them by their length words. With `--pitch global` the whole recording is
one file.

## The frame

A frame is 4 bytes, played for 8, 16, 32 or 64 ms; the chip interpolates every parameter
linearly from the previous frame's values to the new ones over the frame (TP101 p. 6).

| byte | bits (msb first) | content |
|---|---|---|
| 1 | `BW1 BW2 BW3 BW4` (2 bits each) | bandwidth of each of the four resonators: 0 = 726 Hz, 1 = 309 Hz, 2 = 125 Hz, 3 = 50 Hz |
| 2 | `FM3 (3 bits) FM2 (5 bits)` | third formant: 1179, 1337, 1528, 1761, 2047, 2400, 2842, 3400 Hz; second formant: 32 values from 440 to 3400 Hz |
| 3 | `FM1 (5 bits) AMPL (bits 3..1)` | first formant: 32 values from 150 to 1047 Hz; amplitude, high bits |
| 4 | `AMPL (bit 0) FD (2 bits) PI (5 bits)` | amplitude, low bit; frame duration: 0 = 8 ms, 1 = 16, 2 = 32, 3 = 64; pitch increment: 0..15 = +0..+15 Hz, 17..31 = −15..−1 Hz, 16 = noise excitation |

The fourth resonator is fixed at 3500 Hz. The amplitude codes 0..15 follow a logarithmic
law (TP101 table 2); code 0 is silence. The pitch increment applies per 8 ms and scales
with the frame duration. The frequency values are those of the 3.84 MHz reference clock;
at 4 MHz (Thomson) everything is 4.2 % higher and shorter.

## The chip's protocol (TP101 fig. 19-20 and 28)

Two registers: data (A0 = 0) and command/status (A0 = 1). To play a speech file: write
the command `$1A` (stop, slow-stop procedure, REQ pin off), then the starting pitch to
the data register, then every frame byte, each once the status byte's bit 7 (REQ) says
the chip accepts a byte — after the fourth byte of a frame the chip is busy until that
frame starts, between the bytes of a frame it recovers within 3 µs. Once the dummy frame
has started, the file is over: write `$1A` again, or the next file's starting pitch.
