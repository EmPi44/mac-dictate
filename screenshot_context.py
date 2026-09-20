"""Explicit screen and region screenshots for a dictated prompt."""

import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import Quartz
from AppKit import NSEvent, NSScreen, NSBitmapImageRep, NSPNGFileType, NSWorkspace
from ApplicationServices import (
    AXUIElementCopyAttributeValue,
    AXUIElementCreateApplication,
    kAXFocusedWindowAttribute,
    kAXTitleAttribute,
)


CAPTURE_DIR = Path.home() / "Library" / "Caches" / "mac-dictate" / "Screenshots"
RETENTION_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class WindowContext:
    app: str
    bundle_id: str
    title: str
    window_id: int
    url: str = ""

    def as_dict(self):
        return {"app": self.app, "title": self.title, "url": self.url}


def front_window():
    """Return the foreground app's foremost ordinary window."""
    front = NSWorkspace.sharedWorkspace().frontmostApplication()
    if front is None:
        raise RuntimeError("Keine aktive App gefunden.")

    pid = front.processIdentifier()
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    window_id = None
    for info in windows:
        if info.get(Quartz.kCGWindowOwnerPID) != pid or info.get(Quartz.kCGWindowLayer) != 0:
            continue
        bounds = info.get(Quartz.kCGWindowBounds, {})
        if bounds.get("Width", 0) < 100 or bounds.get("Height", 0) < 100:
            continue
        window_id = int(info[Quartz.kCGWindowNumber])
        break
    if window_id is None:
        raise RuntimeError("Kein sichtbares Fenster der aktiven App gefunden.")

    title = ""
    try:
        element = AXUIElementCreateApplication(pid)
        error, focused = AXUIElementCopyAttributeValue(element, kAXFocusedWindowAttribute, None)
        if error == 0 and focused is not None:
            error, value = AXUIElementCopyAttributeValue(focused, kAXTitleAttribute, None)
            if error == 0 and value:
                title = str(value)
    except Exception:
        pass  # Window title is useful context but not required for capture.

    return WindowContext(
        app=str(front.localizedName() or ""),
        bundle_id=str(front.bundleIdentifier() or ""),
        title=title,
        window_id=window_id,
    )


def _check_capture_access():
    if not Quartz.CGPreflightScreenCaptureAccess():
        # The user's screenshot action is the request for access. macOS may
        # require restarting this app after the grant becomes effective.
        granted = Quartz.CGRequestScreenCaptureAccess()
        if granted:
            raise PermissionError("Screen Recording allowed. Restart mac-dictate, then try again.")
        raise PermissionError("Screen Recording denied. Enable it in macOS System Settings.")


def _new_capture_path():
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    return CAPTURE_DIR / ("dictation-" + uuid.uuid4().hex + ".png")


def display_under_pointer():
    """Identify the display containing the pointer when capture is chosen."""
    cursor = NSEvent.mouseLocation()
    for screen in NSScreen.screens():
        frame = screen.frame()
        if (frame.origin.x <= cursor.x < frame.origin.x + frame.size.width
                and frame.origin.y <= cursor.y < frame.origin.y + frame.size.height):
            return int(screen.deviceDescription()["NSScreenNumber"])
    return int(NSScreen.mainScreen().deviceDescription()["NSScreenNumber"])


def capture_screen(display_id):
    """Capture the entire display chosen at the moment the user clicked."""
    _check_capture_access()
    image = Quartz.CGDisplayCreateImage(display_id)
    if image is None:
        raise RuntimeError("Could not capture the screen.")
    path = _new_capture_path()
    data = NSBitmapImageRep.alloc().initWithCGImage_(image).representationUsingType_properties_(
        NSPNGFileType, {}
    )
    if data is None or not data.writeToFile_atomically_(str(path), True):
        raise RuntimeError("Could not save the screenshot.")
    return str(path)


def capture_region():
    """Let macOS draw the selection rectangle; None means Escape/cancel."""
    _check_capture_access()
    path = _new_capture_path()
    result = subprocess.run(
        ["/usr/sbin/screencapture", "-i", "-s", "-x", str(path)],
        capture_output=True,
        check=False,
    )
    if path.is_file() and path.stat().st_size > 0:
        return str(path)
    path.unlink(missing_ok=True)
    if result.stderr:
        raise RuntimeError("Could not capture the region.")
    return None


def discard(path):
    """Delete one of our own screenshots after the draft is discarded."""
    if not path:
        return
    candidate = Path(path)
    if candidate.parent == CAPTURE_DIR and candidate.name.startswith("dictation-"):
        candidate.unlink(missing_ok=True)


def prune_old_captures(now=None):
    """Do not retain unneeded screenshots indefinitely."""
    if not CAPTURE_DIR.exists():
        return
    cutoff = (time.time() if now is None else now) - RETENTION_SECONDS
    for path in CAPTURE_DIR.glob("dictation-*.png"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue
