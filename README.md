# mac-dictate

Push-to-talk dictation for macOS (Apple Silicon optimized).

Hold a key → speak → release → text appears in whatever app is focused.

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

Hold **Right Option** → speak → release → text is typed at your cursor.

## Configuration

Edit the `CONFIG` block at the top of `dictate_app.py`:

| Setting | Default | Options |
|---|---|---|
| `HOTKEY` | `keyboard.Key.alt_r` | `ctrl_r`, `cmd_r`, or any key |
| `LANGUAGE` | `None` (auto-detect) | `"de"`, `"en"`, etc. |
| `MODEL` | `mlx-community/whisper-small-mlx` | any compatible MLX Whisper model |
| `ADD_TRAILING_SPACE` | `True` | `False` |
| `MIN_DURATION_SECONDS` | `0.5` | float |

## How it works

- `sounddevice` keeps a live audio stream open (near-zero CPU when not recording)
- `pynput` listens globally for the hotkey
- On release: the 16 kHz `float32` audio array is queued directly for `mlx-whisper`
- `mlx-whisper` runs on the Apple Silicon GPU through MLX
- Dictations are transcribed serially in capture order
- `pynput.keyboard.Controller` types the result at the current cursor position
