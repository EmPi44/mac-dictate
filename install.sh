#!/bin/bash
# Installs mac-dictate: builds + ad-hoc-signs the launcher, then registers the
# LaunchAgent (auto-starts at login). See CLAUDE.md "Reparatur / Neu-Setup".
#
# WHY the signed launcher: the conda python3 is adhoc-signed and macOS (TCC)
# never honors its OWN Accessibility/Input-Monitoring grant under launchd. Trust
# is only inherited from a validly-signed parent app. So mac-dictate.app's main
# executable is a tiny SIGNED Mach-O launcher (launcher.c) that spawns python as
# a child; python inherits the app's trust. Launch must go through LaunchServices
# (`open -a`) so macOS treats it as a real app.
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
APP="$REPO/mac-dictate.app"
PLIST_DST="$HOME/Library/LaunchAgents/com.emir.mac-dictate.plist"

echo "[1/4] Building signed launcher..."
clang -O2 -o "$APP/Contents/MacOS/mac-dictate" "$REPO/launcher.c"
chmod +x "$APP/Contents/MacOS/mac-dictate"

echo "[2/4] Ad-hoc signing the app bundle..."
codesign --force --sign - --identifier com.emir.mac-dictate "$APP"
codesign --verify --strict "$APP" && echo "      signature OK"

echo "[3/4] Resetting old TCC grants (clean slate)..."
tccutil reset Accessibility com.emir.mac-dictate 2>/dev/null || true
tccutil reset ListenEvent  com.emir.mac-dictate 2>/dev/null || true
tccutil reset Microphone   com.emir.mac-dictate 2>/dev/null || true

echo "[4/4] Installing + starting LaunchAgent..."
launchctl bootout gui/"$(id -u)"/com.emir.mac-dictate 2>/dev/null || true
cp "$REPO/com.emir.mac-dictate.plist" "$PLIST_DST"
launchctl bootstrap gui/"$(id -u)" "$PLIST_DST"

echo ""
echo "Done. mac-dictate is running and will start at login."
echo ""
echo ">>> ONE manual step left: grant Accessibility once."
echo "    System Settings -> Privacy & Security -> Accessibility -> enable 'mac-dictate'."
echo "    (Input Monitoring + Microphone then apply automatically via the app identity.)"
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility" 2>/dev/null || true
echo ""
echo "Useful commands:"
echo "  Status:  launchctl list | grep mac-dictate   (expect a 'application.com.emir.mac-dictate.*' entry too)"
echo "  Stop:    launchctl bootout gui/\$(id -u) $PLIST_DST"
echo "  Start:   launchctl bootstrap gui/\$(id -u) $PLIST_DST"
echo "  Logs:    tail -f ~/Library/Logs/mac-dictate.log   (empty = healthy)"
