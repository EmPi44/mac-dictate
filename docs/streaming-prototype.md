# Isolated MLX Whisper live-output probe

Date: 2026-09-19

`streaming_prototype.py` replays a 16 kHz mono PCM WAV at natural speed. It
prewarms the same `mlx-community/whisper-small-mlx` model used by `dictate_app.py`,
decodes the latest available audio prefix every three seconds, and decodes the
complete recording once at the end. It prints JSON events. It does not capture
the microphone, type into another app, or change the running dictation service.

Atlas Capture supplied the scheduling idea: keep one inference in flight and
replace obsolete pending snapshots with the newest accumulated audio. Its Swift
meeting archive, two-track handling, timestamp agreement, and native
`whisper.cpp` runtime are not copied into this short-dictation probe.

## Reproduce

Use the repository's existing conda Python, as specified in `CLAUDE.md`. A
representative synthetic fixture can be generated with macOS `say` and `ffmpeg`:

```bash
say -v Anna -r 160 -o /tmp/mac-dictate-mixed.aiff 'Ich möchte heute testen, ob Whisper schon während der Aufnahme transkribieren kann. Dabei benutze ich Deutsch und englische Begriffe wie Streaming, Feedback und Release. Am Ende sollte der vollständige Satz sauber im Textfeld stehen.'
ffmpeg -nostdin -loglevel error -y -i /tmp/mac-dictate-mixed.aiff -ar 16000 -ac 1 /tmp/mac-dictate-mixed.wav
/opt/homebrew/Caskroom/miniconda/base/bin/python3 streaming_prototype.py /tmp/mac-dictate-mixed.wav
/opt/homebrew/Caskroom/miniconda/base/bin/python3 -m unittest -v test_streaming_prototype test_dictate_app
```

The actual probe used a 13.770-second `Anna` fixture with the same wording.
Times below are wall-clock seconds from the beginning of paced playback, after
model warmup. Model cache/download time is excluded, matching the running app's
startup warmup.

| Audio available | Event appeared | Decode time | Kind |
| ---: | ---: | ---: | --- |
| 3.000 s | 3.484 s | 0.478 s | provisional |
| 6.000 s | 6.336 s | 0.334 s | provisional |
| 9.000 s | 9.397 s | 0.391 s | provisional |
| 12.000 s | 12.442 s | 0.434 s | provisional |
| 13.770 s | 14.221 s | 0.448 s | final |

The first three-second hypothesis had an incorrect ending (`Aufwand ist`),
which the six-second hypothesis corrected to `Aufnahme transkribieren kann`.
Provisional text must therefore remain replaceable. The final hypothesis still
spelled the product name as `Wisper`; this fixture does not establish accuracy
for Emir's voice or code-switching.

The test worker keeps at most one pending snapshot. Automated checks cover
replacement of obsolete pending audio, exactly one final result, digital
silence, and a normal provisional-to-final sequence. All eight targeted tests,
including the existing dictation tests, passed.

## Boundary

This proves that the installed, warm model can keep up with this synthetic
13.8-second utterance and emit early hypotheses. It does not prove microphone
noise handling, real speech accuracy, text-field insertion, or a shorter final
wait in the active app. The later app integration uses a separate floating
preview so unstable hypotheses do not corrupt text in the focused application;
that integration still needs a real microphone acceptance test.
