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
pip install pynput sounddevice mlx-whisper numpy scipy
```

## Run

```bash
python3 dictate.py
```

On first run, macOS will ask for permissions:
- System Settings → Privacy → Accessibility → allow Terminal
- System Settings → Privacy → Microphone → allow Terminal

## Usage

Hold **Right Ctrl** → speak → release → text is typed at your cursor.

## Configuration

Edit the `CONFIG` block at the top of `dictate.py`:

| Setting | Default | Options |
|---|---|---|
| `HOTKEY` | `keyboard.Key.ctrl_r` | `alt_r`, `cmd_r`, or any key |
| `LANGUAGE` | `None` (auto-detect) | `"de"`, `"en"`, etc. |
| `MODEL` | `whisper-large-v3-turbo` | any mlx-whisper model |
| `ADD_TRAILING_SPACE` | `True` | `False` |
| `MIN_DURATION_SECONDS` | `0.5` | float |

## How it works

- `sounddevice` keeps a live audio stream open (near-zero CPU when not recording)
- `pynput` listens globally for the hotkey
- On release: audio is written to a temp WAV and passed to `mlx-whisper`
- `mlx-whisper` runs on the Neural Engine - typically 1-2s transcription
- `pynput.keyboard.Controller` types the result at the current cursor position
