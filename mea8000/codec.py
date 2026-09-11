"""Frame bit layout and `.mea` container.

Two container layouts exist for the same 4-byte frames:

* ``philips`` — TP101 p. 14 ROM file and the legacy Java tools: ``length:u16be`` (whole file
  including this word), one extra byte, the starting pitch byte, then frames up to
  ``length``. Several files may be concatenated.
* ``rom`` — the Philips/Cedic-Nathan vocabulary image (TP101 fig. 27, the Cedic demo
  cartridge, the phoneme sets): a table of ``u16be`` offsets from the start of the image,
  terminated by ``FF FF``, then the speech files above, one per entry.
* ``player`` — the legacy 6809 routine ``mea8000.digitalized.read`` (2024, this project's
  own variant, not a period format): ``length:u16be`` then groups of ``pitch + frames``; a
  frame with AMPL = 0 ends a group and the next byte, if any, is the next group's pitch.

The layouts cannot be told apart from the bytes with certainty (a file in one layout
usually parses, misaligned, in another): the caller must know the origin of the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import tables as T


@dataclass(frozen=True)
class Frame:
    bw: tuple[int, int, int, int]  # codes 0..3
    fm: tuple[int, int, int]       # codes fm1 0..31, fm2 0..31, fm3 0..7
    ampl: int                      # code 0..15
    fd: int                        # code 0..3
    pi: int                        # code 0..31 (16 = noise)

    @classmethod
    def from_bytes(cls, b: bytes | bytearray | memoryview) -> "Frame":
        b0, b1, b2, b3 = b[0], b[1], b[2], b[3]
        return cls(
            bw=(b0 >> 6, (b0 >> 4) & 3, (b0 >> 2) & 3, b0 & 3),
            fm=(b2 >> 3, b1 & 0x1F, b1 >> 5),
            ampl=((b2 & 7) << 1) | (b3 >> 7),
            fd=(b3 >> 5) & 3,
            pi=b3 & 0x1F,
        )

    def to_bytes(self) -> bytes:
        bw1, bw2, bw3, bw4 = self.bw
        fm1, fm2, fm3 = self.fm
        return bytes((
            (bw1 << 6) | (bw2 << 4) | (bw3 << 2) | bw4,
            (fm3 << 5) | fm2,
            (fm1 << 3) | (self.ampl >> 1),
            ((self.ampl & 1) << 7) | (self.fd << 5) | self.pi,
        ))

    @property
    def noise(self) -> bool:
        return self.pi == T.NOISE_CODE

    @property
    def bw_hz(self) -> tuple[int, int, int, int]:
        return tuple(T.BW_HZ[c] for c in self.bw)  # type: ignore[return-value]

    @property
    def fm_hz(self) -> tuple[int, int, int, int]:
        return (T.FM1_HZ[self.fm[0]], T.FM2_HZ[self.fm[1]], T.FM3_HZ[self.fm[2]], T.FM4_HZ)

    @property
    def ampl_permille(self) -> int:
        return T.AMPL_PERMILLE[self.ampl]

    @property
    def fd_ms(self) -> int:
        return T.FD_MS[self.fd]

    @property
    def pitch_increment(self) -> int:
        """Total pitch change over the frame, in nominal Hz (increment per 8 ms x FD)."""
        return T.PI_HZ[self.pi] << self.fd

    def describe(self) -> str:
        src = "noise" if self.noise else f"pi={T.PI_HZ[self.pi]:+d}"
        fm = self.fm_hz
        bw = self.bw_hz
        return (f"fd={self.fd_ms}ms {src} ampl={self.ampl_permille / 1000:.3f} "
                f"fm1={fm[0]}/{bw[0]} fm2={fm[1]}/{bw[1]} fm3={fm[2]}/{bw[2]} fm4={fm[3]}/{bw[3]}")


@dataclass
class Utterance:
    pitch: int                    # starting pitch byte (Hz / 2)
    frames: list[Frame] = field(default_factory=list)
    extra: int = 0                # the unused header byte (philips layout only)

    @property
    def pitch_hz(self) -> int:
        return 2 * self.pitch

    def duration_ms(self) -> int:
        return sum(f.fd_ms for f in self.frames)


LAYOUTS = ("speech", "vocabulary")


def parse_stream(data: bytes, layout: str = "speech") -> list[Utterance]:
    """`speech`: speech files one after the other; `vocabulary`: a vocabulary image."""
    if layout == "speech":
        return _parse_speech(data)
    if layout == "vocabulary":
        return _parse_vocabulary(data)
    raise ValueError(f"unknown layout {layout!r}")


def detect_layout(data: bytes) -> str:
    """The layout a stream is in, from its content: a speech file is a chain of lengths
    that ends exactly at the end of the data, a vocabulary image an offset table ended
    by $FFFF whose entries all point at valid speech files. Raises if neither or both."""
    found = []
    for layout in LAYOUTS:
        try:
            parse_stream(data, layout)
            found.append(layout)
        except ValueError:
            pass
    if not found:
        raise ValueError("neither a speech file nor a vocabulary image")
    if len(found) == 2:
        # a small offset table is itself a valid speech file (three entries make 8 bytes:
        # a header and one frame); an image whose first entry lands right after its
        # table, as the encoder writes them, is taken as the image
        n = 0
        while 2 * n + 1 < len(data) and ((data[2 * n] << 8) | data[2 * n + 1]) != 0xFFFF:
            n += 1
        first = (data[0] << 8) | data[1]
        return "vocabulary" if first == 2 * n + 2 else "speech"
    return found[0]


def _parse_vocabulary(data: bytes) -> list[Utterance]:
    offsets = []
    i = 0
    while i + 1 < len(data):
        o = (data[i] << 8) | data[i + 1]
        i += 2
        if o == 0xFFFF:
            break
        offsets.append(o)
    else:
        raise ValueError("vocabulary index not terminated by FF FF")
    table_end = 2 * len(offsets) + 2
    # the entries need not be increasing (the Cedic demonstration cartridge lists its
    # words in menu order), but every one points past the table and inside the image
    if not offsets or any(o < table_end or o + 4 > len(data) for o in offsets):
        raise ValueError("vocabulary index does not point past itself into the image")
    out = []
    for o in offsets:
        if o + 4 > len(data):
            raise ValueError(f"index entry {o:#06x} outside the image")
        length = (data[o] << 8) | data[o + 1]
        out.append(_parse_speech(data[o:o + length])[0])
    return out


def _parse_speech(data: bytes) -> list[Utterance]:
    out: list[Utterance] = []
    i = 0
    while i + 4 <= len(data):
        length = (data[i] << 8) | data[i + 1]
        if length < 4 or i + length > len(data):
            raise ValueError(f"bad chunk length {length} at offset {i}")
        end = i + length
        utt = Utterance(pitch=data[i + 3], extra=data[i + 2])
        j = i + 4
        while j + 4 <= end:
            utt.frames.append(Frame.from_bytes(data[j:j + 4]))
            j += 4
        if j != end:
            raise ValueError(f"chunk at offset {i}: {end - j} trailing byte(s)")
        out.append(utt)
        i = end
    if i != len(data):
        raise ValueError(f"{len(data) - i} trailing byte(s) after last chunk")
    return out


def build_stream(utterances: list[Utterance], layout: str = "speech") -> bytes:
    """The free byte of every speech file header holds a copy of the starting pitch (its
    only known use, the Cedic-Nathan convention; the chip never reads it)."""
    def file_bytes(u: Utterance) -> bytes:
        body = b"".join(f.to_bytes() for f in u.frames)
        length = 4 + len(body)
        if length > 0xFFFF:
            raise ValueError("speech file longer than 65535 bytes")
        return bytes((length >> 8, length & 0xFF, u.pitch, u.pitch)) + body

    if layout == "speech":
        return b"".join(file_bytes(u) for u in utterances)
    if layout == "vocabulary":
        files = [file_bytes(u) for u in utterances]
        index_len = 2 * len(files) + 2
        if index_len + sum(len(f) for f in files) > 0xFFFF:
            raise ValueError("vocabulary image longer than 65535 bytes")
        out = bytearray()
        pos = index_len
        for f in files:
            out += bytes((pos >> 8, pos & 0xFF))
            pos += len(f)
        out += b"\xff\xff"
        for f in files:
            out += f
        return bytes(out)
    raise ValueError(f"unknown layout {layout!r}")


def describe_stream(utterances: list[Utterance]) -> str:
    lines = []
    for k, u in enumerate(utterances):
        lines.append(f"speech file {k}: starting pitch {u.pitch_hz} Hz, {len(u.frames)} frames, {u.duration_ms()} ms")
        for n, f in enumerate(u.frames):
            lines.append(f"  {n:3d} {f.to_bytes().hex(' ')}  {f.describe()}")
    return "\n".join(lines)
