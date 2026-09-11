# Samples

Three sentences from [Mozilla Common Voice](https://commonvoice.mozilla.org/) 17.0,
released by their speakers under the CC0 1.0 Universal public domain dedication, taken
from the `test` split (clip identifiers below). For each: the recording (`.wav`, 48 kHz
mono, decoded from the original MP3) and what `mea8000 encode --report` with the default
`thomson` profile makes of it: the speech data (`.mea`), the recording after the
profile's filters (`-source.wav`: normalized, the silence around the sentence dropped),
the chip's rendering (`-chip.wav`, at the chip's own rate) and the report (`.html`, open
it in a browser).

| file | speaker | sentence |
|---|---|---|
| `fr-female` | `common_voice_fr_17963678` | C'est peut-être le moment le plus important de notre débat parlementaire. |
| `fr-male` | `common_voice_fr_17311365` | Il faut veiller à ce qu'elles soient généralisables, chaque fois que c'est possible. |
| `en-female` | `common_voice_en_18309498` | Good evening ladies and gentlemen of the press. |

```
mea8000 encode samples/fr-female.wav /tmp/fr-female.mea --report
```
