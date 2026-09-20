# mac-dictate

One-click dictation for macOS (Apple Silicon optimized).

Press a key → speak with a live preview → press it again → final text appears in whatever app is focused.

Works globally: terminal, browser, editor, chat - anywhere.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## Setup

```bash
brew install portaudio
pip install rumps pynput sounddevice mlx-whisper numpy pyobjc-framework-ApplicationServices
```

## Run

```bash
python3 dictate_app.py
```

On first run, macOS will ask for permissions:
- System Settings → Privacy → Accessibility → allow `mac-dictate`
- System Settings → Privacy → Microphone → allow `mac-dictate`

## Usage

Press **Right Option** once to start speaking, then press it again to finish. A floating preview appears after about two seconds
and updates while you keep speaking. The final text is inserted at
your cursor. The preview is provisional and may correct itself; it never edits
the focused app. Short recordings still finish normally without a preview.

For a task with a screenshot, press **Right Option** once and click a screenshot button while speaking. A small
picture icon with **Vollbild** and **Ausschnitt** buttons appears beside the mouse
pointer. Vollbild captures the display under the pointer. Ausschnitt opens the
macOS selector so you can mark the area to capture.
Press Escape to cancel the selection. The two buttons stay near the pointer's
starting spot so they are easy to click. **F13** is a shortcut for the whole display; you can map a
second mouse side button to it. The microphone keeps recording, and the review card opens
as soon as the image is ready.
The card shows a provisional live transcript. Click **Fertig & kopieren** when you are done
speaking. Recording stops, Whisper finishes the full transcript, and the prompt is copied.
Paste it into
either a new or an existing Codex task. You can also copy the PNG separately and
paste it into that task. Copying one clipboard item replaces the other, so paste
each item before copying the next. The prompt includes the local PNG path for
tasks that can access this Mac's filesystem.

macOS asks for Screen Recording permission on the first explicit capture. Grant
access to mac-dictate in System Settings → Privacy & Security → Screen Recording,
then restart the app. The current version records the app and window title; it
does not retrieve browser URLs. Screenshots are removed if a draft is discarded
before copying. Captures older than 24 hours are removed on the next app start.

## Configuration

Edit the `CONFIG` block at the top of `dictate_app.py`:

| Setting | Default | Options |
|---|---|---|
| `HOTKEY` | `keyboard.Key.alt_r` | `ctrl_r`, `cmd_r`, or any key |
| `SCREENSHOT_HOTKEY` | `keyboard.Key.f13` | any key not used by dictation |
| `LANGUAGE` | `None` (auto-detect) | `"de"`, `"en"`, etc. |
| `MODEL` | `mlx-community/whisper-small-mlx` | any compatible MLX Whisper model |
| `ADD_TRAILING_SPACE` | `True` | `False` |
| `MIN_DURATION_SECONDS` | `0.5` | float |

## How it works

- `sounddevice` keeps a live audio stream open (near-zero CPU when not recording)
- `pynput` listens globally for the hotkey
- During recording, one MLX worker transcribes the latest accumulated audio
  every 1.5 seconds after the first two seconds. Obsolete waiting previews are
  replaced instead of queued, and a non-activating panel shows the latest text.
- On the second press: the 16 kHz `float32` audio array is queued directly for `mlx-whisper`
- `mlx-whisper` runs on the Apple Silicon GPU through MLX
- Dictations are transcribed serially in capture order
- `pynput.keyboard.Controller` types the result at the current cursor position
