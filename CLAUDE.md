# CLAUDE.md - mac-dictate

Push-to-talk Diktat für macOS (Apple Silicon). Rechte Option-Taste (⌥ rechts) halten →
sprechen → loslassen → Whisper transkribiert lokal und tippt den Text an der Cursor-Position.

> **Status: FUNKTIONIERT (verifiziert 2026-06-06, end-to-end vom User getestet).**
> Auto-Start beim Login ist scharf. Wenn es mal wieder klemmt, lies zuerst
> "Die Kernursache" und dann "Reparatur / Neu-Setup".

---

## TL;DR - was woran lag und wie es läuft

Es gab **ZWEI unabhängige Fehlerquellen**, die nacheinander auftraten:

### Problem 1 - "not trusted" (Hotkey/Tippen tat gar nichts)
- **Symptom:** App lief, Icon `Dic` da, aber Hotkey/Tippen tat nichts. Log voll mit
  `This process is not trusted!`. Schalter in den Bedienungshilfen standen auf "an" - trotzdem nichts.
- **Wahre Ursache:** Das conda-`python3` ist **adhoc-signiert**. macOS (TCC) gewährt einem
  adhoc-Binary unter **launchd** seine Accessibility-/Input-Monitoring-Rechte **niemals über
  seine eigene Freigabe** - auch wenn der Schalter "an" ist und der csreq sogar passt. Trust
  gibt es für so ein Binary nur **geerbt von einer vertrauenswürdigen, gültig signierten
  GUI-App** als verantwortlichem Eltern-Prozess (deshalb klappte es testweise aus dem Terminal,
  aber nie unter dem LaunchAgent).
- **Lösung:** Ein winziger, **gültig (ad-hoc) signierter Mach-O-Launcher** ist der Kern der
  `mac-dictate.app`. Er startet das conda-`python3` als **Kindprozess**. Das python **erbt**
  den Trust der signierten App. Start per `open -a` (LaunchServices), damit macOS sie als echte
  App registriert. Ein einmaliges Aktivieren des **einen** Schalters `mac-dictate` in den
  Bedienungshilfen reicht - Accessibility, Input Monitoring UND Mikrofon greifen dann alle über
  die App-Identität `com.emir.mac-dictate`. (Details: Gotcha #1)

### Problem 2 - Aufnahme ok, aber kein Text (ffmpeg)
- **Symptom:** Hotkey + Aufnahme funktionierten (Icon `Dic ●`, Audio kam an), aber beim
  Loslassen kam **kein Text**.
- **Wahre Ursache:** Whisper rief intern `ffmpeg` auf, das unter dem launchd-Minimal-PATH nicht
  gefunden wird → `FileNotFoundError`.
- **Lösung:** Audio direkt als `float32`-Array an Whisper geben statt über eine WAV-Datei -
  `ffmpeg` wird gar nicht mehr gebraucht. (Details: Gotcha #0)

---

## Architektur (so hängt es zusammen)

```
launchd (LaunchAgent: com.emir.mac-dictate.plist)
  └─ /usr/bin/open -W -a mac-dictate.app        (Start via LaunchServices => echte App-Identität)
       └─ mac-dictate.app/Contents/MacOS/mac-dictate   (SIGNIERTER Mach-O-Launcher, aus launcher.c)
            └─ python3 dictate_app.py            (Kindprozess - ERBT den Trust der signierten App)
```

Wichtige Dateien:
- `launcher.c` - Quelle des signierten Launchers (spawnt python als Kind, leitet Signale weiter, `wait`et).
- `mac-dictate.app/Contents/MacOS/mac-dictate` - das **kompilierte + signierte** Launcher-Binary.
- `mac-dictate.app/Contents/Info.plist` - Bundle-ID `com.emir.mac-dictate`, `LSUIElement` (kein Dock-Icon).
- `dictate_app.py` - die eigentliche Menüleisten-App (Aufnahme + Whisper + Tippen). **Liegt außerhalb
  des Bundles**, wird per absolutem Pfad referenziert. Editierbar, **ohne** die App-Signatur zu brechen.
- `com.emir.mac-dictate.plist` - LaunchAgent, startet via `open -W -a`.
- `install.sh` - kopiert die plist nach `~/Library/LaunchAgents/` und bootstrappt sie.

---

## Normaler Betrieb

- **Läuft es?** `launchctl list | grep mac-dictate` → es sollten ZWEI Einträge da sein:
  `com.emir.mac-dictate` (der `open -W`-Job) UND `application.com.emir.mac-dictate.*`
  (= von LaunchServices als App erkannt, das ist das Erfolgssignal). Plus `pgrep -fl dictate_app`.
- **Manuell starten:** `open -a /Users/emir/repo_workspaces/mac-dictate/mac-dictate.app`
- **Stoppen:** `pkill -f dictate_app` (LaunchAgent startet via KeepAlive neu) bzw. ganz aus:
  `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.emir.mac-dictate.plist`
- **Bedienung:** rechte Option-Taste (⌥ rechts) halten → Icon `Dic ●` (Aufnahme) → sprechen →
  loslassen → `Dic ...` (transkribiert) → Text erscheint am Cursor.
- **Leeres Log ist gut:** `~/Library/Logs/mac-dictate.log` ist im Normalbetrieb leer/nicht vorhanden.
  Steht dort `This process is not trusted!`, ist der Trust kaputt → siehe Reparatur.

---

## Reparatur / Neu-Setup (wenn Trust kaputt ist oder auf neuem Mac)

Trust kann brechen, wenn: die `.app` verschoben/umbenannt wird, der Launcher neu kompiliert/neu
signiert wird, conda-Python an einen anderen Pfad wandert, oder macroupdate die TCC-Einträge zerlegt.
Dann diese Schritte (alle Pfade ggf. an neuen conda-Pfad anpassen):

```bash
REPO=/Users/emir/repo_workspaces/mac-dictate
APP="$REPO/mac-dictate.app"

# 1. Signierten Launcher (neu) bauen und ins Bundle legen
clang -O2 -o "$APP/Contents/MacOS/mac-dictate" "$REPO/launcher.c"
chmod +x "$APP/Contents/MacOS/mac-dictate"

# 2. App ad-hoc signieren (stabile, gültige Identität - DAS ist der entscheidende Schritt)
codesign --force --sign - --identifier com.emir.mac-dictate "$APP"
codesign --verify --strict "$APP" && echo "Signatur GUELTIG"

# 3. Alte/kaputte TCC-Grants für das Bundle zurücksetzen (sauberer Neustart)
tccutil reset Accessibility com.emir.mac-dictate
tccutil reset ListenEvent  com.emir.mac-dictate
tccutil reset Microphone   com.emir.mac-dictate

# 4. LaunchAgent (neu) installieren und starten
cp "$REPO/com.emir.mac-dictate.plist" "$HOME/Library/LaunchAgents/com.emir.mac-dictate.plist"
launchctl bootout   gui/$(id -u)/com.emir.mac-dictate 2>/dev/null
launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/com.emir.mac-dictate.plist"
# (oder einfach: cd "$REPO" && ./install.sh)
```

**5. EINMALIG manuell freigeben** (das ist der einzige Schritt, der einen Klick braucht):
- Systemeinstellungen öffnen sich automatisch (die App fragt beim Start nach):
  `open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"`
- In **Datenschutz & Sicherheit → Bedienungshilfen** den frischen Eintrag **`mac-dictate`** auf **AN**.
- Das reicht: Input Monitoring und Mikrofon greifen automatisch über dieselbe App-Identität.
  (Mikrofon wird sonst beim ersten Aufnehmen abgefragt → erlauben.)

**6. Verifizieren** (statt raten - hat sich bewährt). Temporär in `dictate_app.py` direkt im
`if __name__ == "__main__":`-Block ein `_diag()` einbauen, das schreibt:

```python
from ApplicationServices import AXIsProcessTrusted
import Quartz, os
open(os.path.expanduser("~/Library/Logs/mac-dictate-trust.log"), "a").write(
    "AX=%s Listen=%s Post=%s\n" % (
        AXIsProcessTrusted(),
        Quartz.CGPreflightListenEventAccess(),
        Quartz.CGPreflightPostEventAccess()))
```

App neu starten und `~/Library/Logs/mac-dictate-trust.log` lesen → **alle drei müssen `True`
sein**. Danach das Diag wieder entfernen. (Achtung: aus dem Terminal/Harness gestartet liefert
`AXIsProcessTrusted` fälschlich `True` durch Vererbung - **immer über den LaunchAgent / `open -a`
testen**, sonst misst man Mist.)

---

## Bekannte Stolperfallen (Gotchas)

0. **Whisper ruft `ffmpeg` auf - das fehlt im launchd-PATH** (2. Fehlerquelle, diagnostiziert 2026-06-06).
   Symptom: Hotkey + Aufnahme funktionieren (Icon wird zu `Dic ●`, Audio kommt an), aber beim
   Loslassen kommt **kein Text**. Im Log: `transcription failed: FileNotFoundError(2, ...)`.
   Ursache: `_model.transcribe("datei.wav")` (Pfad-Variante) ruft intern `whisper.load_audio()`
   → startet **`ffmpeg`**. Unter launchd ist der `PATH` minimal (`/usr/bin:/bin:/usr/sbin:/sbin`),
   und `ffmpeg` liegt in `/opt/homebrew/bin` → nicht gefunden → `FileNotFoundError`.
   **Lösung (umgesetzt):** Das Audio liegt eh schon als `float32`-Array in 16 kHz vor - direkt an
   `mlx_whisper.transcribe(audio_array, path_or_hf_repo=MODEL, fp16=True)` übergeben statt über
   eine WAV-Datei. Damit entfällt `ffmpeg` komplett (kein PATH-Problem, keine Temp-Datei, schneller).
   Siehe `transcribe_and_type()` in `dictate_app.py`. **NICHT** wieder auf die
   Datei-/Pfad-Variante zurückbauen.
   (Alternative wäre `EnvironmentVariables` mit `PATH` in der plist, aber die Array-Variante ist sauberer.)

1. **Adhoc-Python hat unter launchd KEINEN eigenen Trust** (Hauptursache, diagnostiziert 2026-06-06).
   - `/opt/homebrew/Caskroom/miniconda/base/bin/python3` → Symlink auf `python3.12`, `Signature=adhoc`.
   - Selbst mit gültigem TCC-Eintrag (`auth_value=2`, csreq matcht den aktuellen cdhash!) liefert
     `AXIsProcessTrusted()` unter einem echten gui-LaunchAgent `False`. Hart per Probe bewiesen.
   - `exec python3 …` aus dem Bundle macht es schlimmer: der signierte App-Prozess wird durch das
     unsignierte python ersetzt, Identität futsch. Deshalb macht der Launcher **kein exec**, sondern
     spawnt python als **Kind** und bleibt als verantwortlicher (signierter) Eltern-Prozess am Leben.
   - Auch die `__CFBundleIdentifier`-Env-Variable allein reicht NICHT - TCC bewertet das laufende
     Binary/den verantwortlichen Prozess, nicht die Env-Var.
   - **Einzige zuverlässige Lösung:** signierte `.app` (siehe Architektur) + Start via `open -a`.

2. **Drei Berechtigungen, aber ein Schalter genügt** (dank App-Identität):
   - **Bedienungshilfen** (Accessibility) - fürs Tippen via `pynput` (`CGPreflightPostEventAccess`).
   - **Input Monitoring** - für den globalen Hotkey-Listener (`CGPreflightListenEventAccess`).
   - **Mikrofon** - für die Aufnahme.
   Über `com.emir.mac-dictate` greifen alle drei. Manuell nötig ist nur der Bedienungshilfen-Schalter.

3. **Nicht das Binary, sondern die App freigeben.** Die alten Listen-Einträge `python`, `python3.12`,
   `python.app` (und ggf. ein generisches `bin`) sind verwaiste Fehlversuche - bringen nichts und
   können mit dem `-` entfernt werden. Nur `mac-dictate` zählt.

4. **App nicht verschieben/umbenennen.** TCC bindet den Grant an Pfad + Signatur des Bundles.
   Verschieben/Umbenennen/Neu-Signieren bricht den Trust → Reparatur-Schritte 2-5 wiederholen.
   `dictate_app.py` editieren ist OK (liegt außerhalb des Bundles, bricht die Signatur nicht).

5. **Richtiges Python verwenden - NICHT das `venv`:**
   `/opt/homebrew/Caskroom/miniconda/base/bin/python3` (conda, 3.12) hat alle Dependencies
   (`rumps`, `mlx-whisper`, `sounddevice`, `numpy`, `pynput`, `pyobjc`).
   Das `venv/` im Repo (3.14) ist unvollständig (kein `rumps`) - verwaist, nicht benutzen.

6. **Import von `AXIsProcessTrustedWithOptions`** liegt in `ApplicationServices` (oder `HIServices`),
   **NICHT in `Quartz`**. Falscher Import → sofortiger `ImportError` beim Start
   (siehe `dictate_app.py`).

---

## Konfiguration

Im `CONFIG`-Block oben in `dictate_app.py`:
- `HOTKEY` = `keyboard.Key.alt_r` (rechte Option-Taste; auf `ctrl_r`/`cmd_r`/etc. änderbar)
- `LANGUAGE` = `None` (Auto de+en); auf `"de"` oder `"en"` setzen für mehr Tempo/Genauigkeit
- `MODEL` = `"mlx-community/whisper-small-mlx"` (multilinguales Whisper-small-Modell für MLX)
