# The formats

`mea8000 encode` writes the original, official formats of the chip's era, and nothing
else:

- [speech-file.md](speech-file.md) — the **speech file** (Philips Technical Publication
  101, 1983; the format of every Cedic-Nathan speech product for Thomson): a header and
  frames. `--format speech` writes speech files one after the other, extension `.mea`.
- [vocabulary-image.md](vocabulary-image.md) — the **vocabulary image**: an offset table,
  then speech files, each reachable by its number (the layout of the Cedic-Nathan
  cartridges). `--format vocabulary`, extension `.voc.mea`.

Both share the [frame](speech-file.md#the-frame), the chip's unit of sound. The
extensions are a convention: `mea8000 render` and `mea8000 inspect` recognise the two
formats from their content.

References: Philips, *MEA8000 voice synthesizer: principles and interfacing*, Technical
Publication 101 (1983); Signetics, *MEA8000 datasheet* (1985); *Parole et Micros*
(Cedic-Nathan, 1985).
