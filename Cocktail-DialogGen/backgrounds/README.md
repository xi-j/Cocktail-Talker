# Background recordings

10 background recordings for each of the **18 predefined environment labels**, so you can
generate dialogs by naming a label instead of supplying your own recording:

```python
from cocktail_dialoggen import generate_dialogs, LABELS
print(LABELS)  # BUS CAFE CAFETER CAR FIELD HALLWAY KITCHEN LIVING MEETING METRO
               # OFFICE PARK RESTO RIVER SQUARE STATION TRAFFIC WASHING

generate_dialogs(environment_audio="CAFETER", ...)   # picks a bundled CAFETER recording
generate_dialogs(environment_audio="my_cafe.wav", ...)  # or your own recording
```

You can also add your own: drop `.wav` files into `backgrounds/<LABEL>/`, or point
`$COCKTAIL_BACKGROUNDS` at a different root.

## Layout

```
backgrounds/
├── manifest.json          # label -> [{file, freesound_id, freesound_url, original_name, duration_sec}]
└── <LABEL>/<freesound_id>.wav   # mono, 24 kHz, trimmed to <=60 s
```

## Provenance & licensing  ⚠️

These clips are **derived from [Freesound](https://freesound.org/) preview audio**, downloaded
by relevance and human-verified for each environment category (as described in the paper). Each
clip's filename is its Freesound **sound id**; the source page is `https://freesound.org/s/<id>/`
(also in `manifest.json`).

**Freesound sounds carry per-sound Creative Commons licenses (CC0, CC-BY, CC-BY-NC, …).** Before
redistributing this bundle or using it beyond your own research, you must check and comply with each
sound's individual license (attribution, share-alike, non-commercial terms). We did **not** capture
per-sound license/author metadata at download time, so this bundle is provided for research
convenience only. To assemble proper attribution, query the Freesound API for each id, or replace
these clips with CC0-only recordings.
