# The vocabulary image

The layout of the Cedic-Nathan cartridges (the demonstration cartridge, the phoneme
sets; TP101 fig. 27): a table gives where each speech file starts, so that a program
reaches a word by its number.

| offset | size | content |
|---|---|---|
| 0 | 2 × n | n offsets, big endian, each the position of a speech file from the start of the image |
| 2 × n | 2 | `$FF $FF`, the end of the table |
| … | | the speech files, in the [speech file](speech-file.md) format |

The offsets need not be increasing (the demonstration cartridge lists its words in the
order of its menu); every one points past the table, inside the image. The whole image
fits in 65 535 bytes, the reach of a 16-bit offset.

`mea8000 encode --format vocabulary` writes one entry per word group of the recording,
in their order (entry 0 is the first word group), extension `.voc.mea`. To play word
*k*, read entry *k* of the table and play the speech file it points at.
