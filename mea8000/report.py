"""The HTML report of `encode --report`: one self-contained page, no dependency beyond
the standard library — spectrograms as PNG (zlib) embedded in base64, lanes as SVG, the
two sounds embedded (16 kHz) to play with an A/B switch, the profile's pins, a summary.
The two sounds are also written beside it as WAV files: the recording after the
profile's filters (what the encoder saw) and the chip's rendering."""

from __future__ import annotations

import base64
import html
import io
import struct
import wave
import zlib
from pathlib import Path

import numpy as np
from scipy.signal import resample, stft

from .analysis import HOP, RATE, to_8k
from .codec import Utterance, describe_stream
from .fastsim import render_fast
from .sim import DEFAULT, with_model
from .wav import write_wav


# ----------------------------------------------------------------- PNG

def png_bytes(rgb: np.ndarray) -> bytes:
    """A PNG from an (h, w, 3) uint8 array, standard library only."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, body: bytes) -> bytes:
        c = tag + body
        return struct.pack(">I", len(body)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def spectrogram_rgb(x8: np.ndarray, floor_db: float = -70.0) -> np.ndarray:
    """Log-magnitude STFT of an 8 kHz signal, 4 kHz at the top, one column per 8 ms."""
    if len(x8) < 256:
        x8 = np.pad(x8, (0, 256 - len(x8)))
    _, _, z = stft(x8, fs=RATE, nperseg=256, noverlap=256 - HOP, boundary=None)
    mag = 20 * np.log10(np.abs(z) + 1e-9)
    mag -= mag.max()
    v = np.clip((mag - floor_db) / -floor_db, 0, 1)[::-1]
    r = np.clip(v * 1.6, 0, 1)
    g = np.clip(v * 1.6 - 0.5, 0, 1)
    b = np.clip(0.3 + v * 0.2 - (v > 0.7) * (v - 0.7) * 2, 0, 1) * (v > 0.02)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _img(rgb: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(rgb)).decode()


EMBED_RATE = 16000


def _audio(y: np.ndarray, rate: int) -> str:
    """A sound as a data URI: 16-bit mono WAV at 16 kHz."""
    y = np.asarray(y, dtype=np.float64)
    if rate != EMBED_RATE:
        y = resample(y, int(round(len(y) * EMBED_RATE / rate)))
    pcm = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(EMBED_RATE)
        w.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------- lanes

def _polyline(xs, ys, x0, x1, y0, y1, width, height, color, dashed=False):
    pts = []
    for x, y in zip(xs, ys):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        px = (x - x0) / (x1 - x0) * width
        py = height - (y - y0) / (y1 - y0) * height
        pts.append(f"{px:.1f},{py:.1f}")
    if not pts:
        return ""
    dash = ' stroke-dasharray="3,3"' if dashed else ""
    return f'<polyline fill="none" stroke="{color}" stroke-width="1"{dash} points="{" ".join(pts)}"/>'


def _dots(xs, ys, x0, x1, y0, y1, width, height, color):
    out = []
    for x, y in zip(xs, ys):
        if y is None or np.isnan(y):
            continue
        px = (x - x0) / (x1 - x0) * width
        py = height - (y - y0) / (y1 - y0) * height
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.2" fill="{color}"/>')
    return "".join(out)


def _lane(title, inner, width, height, y_labels):
    labels = "".join(f'<text x="2" y="{height - (v - y_labels[0]) / (y_labels[-1] - y_labels[0]) * height + 4:.0f}" '
                     f'font-size="9" fill="#888">{v:g}</text>' for v in y_labels)
    return (f'<div class="lane"><div class="title">{title}</div>'
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none">'
            f'<rect width="{width}" height="{height}" fill="#1b1b1b"/>{inner}{labels}</svg></div>')


# ----------------------------------------------------------------- resonators

FM_COLORS = ("#5bd0ff", "#6ee06e", "#ffffff")    # readable over the orange of the spectrogram
BW_WIDTH = {0: 3.5, 1: 2.5, 2: 1.6, 3: 1.0}     # 726, 309, 125, 50 Hz


def _resonator_overlay(utts: list[Utterance], dur: float, scale: float) -> str:
    """The three formants of every frame as ramps over the chip's spectrogram: the chip
    interpolates from the previous frame's values to the new ones over the frame, so each
    frame is a straight segment; the first frame of a file is preset (flat). Silent frames
    break the traces. Frequencies are those the chip renders at its clock (tables / scale),
    on the recording's time axis (viewBox: seconds by Hz, 4 kHz at the top)."""
    from . import tables as T
    from .encoder import FM_TABLES

    lines = []
    t = 0.0
    for u in utts:
        prev = None
        for f in u.frames[:-1]:
            length = (1 << f.fd) * HOP / RATE * scale
            fm = tuple(FM_TABLES[g][f.fm[g]] / scale for g in range(3))
            if f.ampl == 0:
                prev = None
                t += length
                continue
            start = prev if prev is not None else fm
            for g in range(3):
                seg = f'x1="{t:.4f}" y1="{4000 - start[g]:.0f}" x2="{t + length:.4f}" y2="{4000 - fm[g]:.0f}" vector-effect="non-scaling-stroke"'
                # a dark halo under the trace keeps it readable whatever the spectrogram's colour
                lines.append(f'<line {seg} stroke="#000" stroke-opacity="0.55" stroke-width="{BW_WIDTH[f.bw[g]] + 2.5}"/>'
                             f'<line {seg} stroke="{FM_COLORS[g]}" stroke-width="{BW_WIDTH[f.bw[g]]}"/>')
            prev = fm
            t += length
    return (f'<svg id="resonators" viewBox="0 0 {dur:.4f} 4000" preserveAspectRatio="none" opacity="0.9">'
            + "".join(lines) + "</svg>")


# ----------------------------------------------------------------- the page

def write_report(page: Path, source_wav: Path, chip_wav: Path, source_name: str, prepared,
                 utts: list[Utterance], result, prof, size_bytes: int) -> None:
    """`prepared` is the recording after the filters (`filters.Prepared`); it is written to
    `source_wav`, the chip's rendering to `chip_wav`, and both are embedded in the page."""
    x, rate = prepared.x, prepared.rate
    write_wav(source_wav, x, rate)
    model = with_model(DEFAULT, clock_hz=prof.clock_hz)
    y = render_fast(utts, model, "chunk").astype(np.float64)
    write_wav(chip_wav, y, model.sample_rate)
    # both signals on the recording's time, at 8 kHz, for the pictures
    x8 = to_8k(x, rate)
    y8 = to_8k(y, model.sample_rate)
    n = len(x8)
    y8 = np.pad(y8, (0, max(0, n - len(y8))))[:n]
    dur = n / RATE
    an = result.analysis
    scale = prof.rate_scale
    slots = np.arange(an.n_frames) * (HOP / RATE) * scale     # analysis slots on recording time
    src_wav = Path(source_name)

    # frames per slot (dummy frames take no time), on recording time
    starts, pitch, ampl, voiced, t = [], [], [], [], 0.0
    from . import tables as T
    for u in utts:
        starts.append(t)
        p = 2 * u.pitch
        for f in u.frames[:-1]:
            if f.pi != T.NOISE_CODE:
                p = (p + (T.PI_HZ[f.pi] << f.fd)) & 0xFFFF
            for _ in range(1 << f.fd):
                pitch.append(p)
                ampl.append(f.ampl)
                voiced.append(f.pi != T.NOISE_CODE and f.ampl > 0)
                t += HOP / RATE * scale
    ft = np.arange(len(pitch)) * (HOP / RATE) * scale
    pitch = np.array(pitch, float)
    voiced = np.array(voiced, bool)
    resonators = _resonator_overlay(utts, dur, scale)

    W, H = 1200, 110
    lanes = []
    f0 = np.where(an.voiced, an.f0, np.nan)
    inner = _dots(slots, f0, 0, dur, 50, 400, W, H, "#5b9bff") + _dots(ft, np.where(voiced, pitch, np.nan), 0, dur, 50, 400, W, H, "#ff9f43")
    inner += "".join(f'<line x1="{s / dur * W:.1f}" y1="0" x2="{s / dur * W:.1f}" y2="{H}" stroke="#6c6" stroke-width="1"/>' for s in starts)
    lanes.append(_lane("pitch (Hz) — blue: the recording, orange: the chip, green: a speech file starts", inner, W, H, [50, 200, 400]))
    energy = np.clip(an.energy_db, -80, 0)
    inner = _polyline(slots, energy, 0, dur, -80, 0, W, H, "#7c7")
    va = np.where(an.voiced, -78.0, np.nan)
    inner += _dots(slots, va, 0, dur, -80, 0, W, H, "#5b9bff") + _dots(ft, np.where(voiced, -74.0, np.nan), 0, dur, -80, 0, W, H, "#ff9f43")
    lanes.append(_lane("level (dB) and voicing — bands: blue the recording, orange the chip", inner, W, H, [-80, -40, 0]))
    inner = _polyline(ft, np.array(ampl, float), 0, dur, 0, 16, W, H, "#ccc")
    lanes.append(_lane("amplitude code of the frames (0-15)", inner, W, H, [0, 8, 16]))

    pins_html = ""
    if prof.pins:
        rows = "".join(f"<tr><td>{p.start:.3f}</td><td>{p.end:.3f}</td><td>{html.escape(', '.join(f'{k} = {v}' for k, v in p.as_dict().items() if k not in ('from', 'to')))}</td></tr>" for p in prof.pins)
        pins_html = f"<h2>Pins of the profile</h2><table><tr><th>from (s)</th><th>to (s)</th><th>decision</th></tr>{rows}</table>"
    files_rows = "".join(f"<tr><td>{k}</td><td>{s:.3f}</td><td>{2 * u.pitch}</td><td>{len(u.frames)}</td><td>{u.duration_ms() * scale / 1000:.3f}</td></tr>"
                         for k, (u, s) in enumerate(zip(utts, starts)))
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(src_wav.stem)} — MEA8000 report</title>
<style>
body{{font-family:system-ui,sans-serif;background:#111;color:#ddd;margin:0;padding:16px 24px}}
h1{{font-size:18px;margin:0 0 8px}} h2{{font-size:14px;margin:18px 0 6px;color:#aaa}}
.spec{{position:relative}} .spec img{{width:100%;height:220px;image-rendering:pixelated;display:block}}
.spec svg{{position:absolute;left:0;top:0;width:100%;height:220px;pointer-events:none}}
.toggle{{font-size:12px;font-weight:normal;color:#aaa;margin-left:12px}} .toggle input{{vertical-align:middle}}
.title{{font-size:12px;color:#aaa;margin:8px 0 2px}} .lane{{margin-bottom:4px}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #333;padding:2px 8px;text-align:left}}
.ab button{{font-size:14px;padding:4px 10px;margin-right:8px}} audio{{vertical-align:middle}}
.summary{{font-size:13px;color:#bbb}} code{{color:#eee}}
</style></head><body>
<h1>{html.escape(src_wav.name)} → MEA8000</h1>
<div class="summary">profile <code>{html.escape(prof.name)}</code> · clock {prof.clock_hz / 1e6:.2f} MHz · format {prof.format} · pitch {prof.pitch} · frame {prof.frame_ms} ms · quality {prof.quality}
 · {html.escape(prepared.describe())} · high-pass {prof.highpass_hz:g} Hz · {len(utts)} word group{'s' if len(utts) != 1 else ''} · {sum(len(u.frames) for u in utts)} frames · {size_bytes} bytes · {dur:.2f} s</div>
<h2>Listen</h2>
<div class="ab">
<button onclick="play('a')">A: the recording, after the filters</button><audio id="a" controls src="{_audio(x, rate)}"></audio><br>
<button onclick="play('b')">B: the chip</button><audio id="b" controls src="{_audio(y, model.sample_rate)}"></audio>
<span style="font-size:12px;color:#888">&nbsp; TAB while playing switches A/B at the same position · the same sounds as {html.escape(source_wav.name)} and {html.escape(chip_wav.name)} beside this page</span>
</div>
<h2>The recording, after the filters</h2><div class="spec"><img src="{_img(spectrogram_rgb(x8))}" alt="spectrogram of the recording, 0-4 kHz"></div>
<h2>The chip <label class="toggle"><input type="checkbox" id="res" checked onchange="document.getElementById('resonators').style.display=this.checked?'':'none'"> resonators
 <span style="color:{FM_COLORS[0]}">FM1</span> <span style="color:{FM_COLORS[1]}">FM2</span> <span style="color:{FM_COLORS[2]}">FM3</span>, thicker = wider bandwidth</label></h2>
<div class="spec"><img src="{_img(spectrogram_rgb(y8))}" alt="spectrogram of the chip's rendering, 0-4 kHz">{resonators}</div>
{''.join(lanes)}
<h2>Speech files</h2>
<table><tr><th>#</th><th>starts at (s)</th><th>starting pitch (Hz)</th><th>frames</th><th>duration (s)</th></tr>{files_rows}</table>
{pins_html}
<h2>Frames</h2><details><summary>{sum(len(u.frames) for u in utts)} frames, as <code>mea8000 inspect</code> prints them</summary><pre style="font-size:11px">{html.escape(describe_stream(utts))}</pre></details>
<script>
let cur=null;
function play(id){{const a=document.getElementById('a'),b=document.getElementById('b');const from=id==='a'?b:a,to=id==='a'?a:b;
 if(!from.paused){{to.currentTime=from.currentTime;from.pause();}} to.play();cur=id;}}
document.addEventListener('keydown',e=>{{if(e.key==='Tab'){{e.preventDefault();play(cur==='a'?'b':'a');}}}});
</script>
</body></html>
"""
    page.write_text(doc)
