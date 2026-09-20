"""Native review card for a dictated prompt with an optional screenshot."""

from AppKit import (
    NSApp,
    NSBackingStoreBuffered,
    NSBezelStyleRounded,
    NSButton,
    NSColor,
    NSFont,
    NSImage,
    NSImageScaleProportionallyUpOrDown,
    NSImageView,
    NSMakeRect,
    NSPanel,
    NSPasteboard,
    NSPasteboardTypePNG,
    NSScreen,
    NSScrollView,
    NSFloatingWindowLevel,
    NSTextField,
    NSTextView,
    NSView,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
)
from Foundation import NSObject


class _Actions(NSObject):
    def copy_(self, _sender):
        self.owner._copy()

    def copyImage_(self, _sender):
        self.owner._copy_image()

    def discard_(self, _sender):
        self.owner._discard()

    def windowWillClose_(self, _notification):
        self.owner._discard()


class ReviewPanel:
    """Call on the AppKit main thread; callbacks may start background work."""

    def __init__(self, on_copy, on_discard):
        self.on_copy = on_copy
        self.on_discard = on_discard
        self._actions = _Actions.alloc().init()
        self._actions.owner = self
        self._context = None
        self._image_path = None
        self._closing = False
        self._copied = False
        self._image_pending = False
        self._text_pending = False
        self._image_error = None
        self._text_error = None
        self._recording = False

        screen = NSScreen.mainScreen().visibleFrame()
        width, height = 620, 500
        x = screen.origin.x + (screen.size.width - width) / 2
        y = screen.origin.y + (screen.size.height - height) / 2
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, width, height),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
            NSBackingStoreBuffered,
            False,
        )
        self.panel.setTitle_("Diktat prüfen")
        self.panel.setReleasedWhenClosed_(False)
        self.panel.setDelegate_(self._actions)
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setLevel_(NSFloatingWindowLevel)
        self.panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        content.setWantsLayer_(True)
        content.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.10, 0.12, 0.15, 1).CGColor()
        )
        self.panel.setContentView_(content)

        self.heading = self._label("Diktat mit Screenshot", 20, 457, 580, 24, 18)
        content.addSubview_(self.heading)
        self.hint = self._label("Prompt kopieren, im gewünschten Task einfügen.", 20, 433, 580, 18, 12)
        content.addSubview_(self.hint)

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(20, 235, 580, 185))
        scroll.setHasVerticalScroller_(True)
        scroll.setDrawsBackground_(False)
        self.text_view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 580, 185))
        self.text_view.setFont_(NSFont.systemFontOfSize_(15))
        self.text_view.setTextColor_(NSColor.whiteColor())
        self.text_view.setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(0.15, 0.18, 0.22, 1))
        self.text_view.setRichText_(False)
        self.text_view.setVerticallyResizable_(True)
        self.text_view.setHorizontallyResizable_(False)
        self.text_view.textContainer().setWidthTracksTextView_(True)
        scroll.setDocumentView_(self.text_view)
        content.addSubview_(scroll)

        self.image_view = NSImageView.alloc().initWithFrame_(NSMakeRect(20, 90, 155, 130))
        self.image_view.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        content.addSubview_(self.image_view)
        self.context_label = self._label("Bild wird aufgenommen ...", 190, 125, 410, 95, 12)
        self.context_label.setUsesSingleLineMode_(False)
        content.addSubview_(self.context_label)

        self.status = self._label("", 20, 58, 580, 24, 12)
        content.addSubview_(self.status)
        self.discard_button = self._button("Verwerfen", 20, 20, 110, "discard:")
        self.copy_image_button = self._button("Bild kopieren", 328, 20, 120, "copyImage:")
        self.copy_button = self._button("Prompt kopieren", 452, 20, 148, "copy:")
        self.copy_button.setEnabled_(True)
        self.copy_image_button.setEnabled_(False)
        for button in (self.discard_button, self.copy_image_button, self.copy_button):
            content.addSubview_(button)

    @staticmethod
    def _label(value, x, y, width, height, size):
        label = NSTextField.labelWithString_(value)
        label.setFrame_(NSMakeRect(x, y, width, height))
        label.setFont_(NSFont.systemFontOfSize_(size))
        label.setTextColor_(NSColor.whiteColor())
        return label

    def _button(self, title, x, y, width, action):
        button = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, width, 32))
        button.setTitle_(title)
        button.setBezelStyle_(NSBezelStyleRounded)
        button.setTarget_(self._actions)
        button.setAction_(action)
        return button

    def show(self, text, image_path, context, error=None, pending=False, text_pending=False, text_error=None, recording=False):
        self._closing = False
        self._copied = False
        self.discard_button.setTitle_("Verwerfen")
        self._context = context
        self._image_path = image_path
        self._image_pending = pending
        self._text_pending = text_pending
        self._image_error = error
        self._text_error = text_error
        self._recording = recording
        self.text_view.setString_(text)
        self.text_view.setEditable_(not text_pending)
        self.set_image(image_path, context, error, pending=pending)
        self.panel.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        self.panel.makeFirstResponder_(self.text_view)

    def set_image(self, image_path, context, error=None, pending=False):
        self._image_path = image_path
        self._context = context
        self._image_pending = pending
        self._image_error = error
        if image_path:
            self.image_view.setImage_(NSImage.alloc().initWithContentsOfFile_(image_path))
            parts = [context.get("scope", ""), context.get("app", ""), context.get("title", ""), context.get("url", "")]
            self.context_label.setStringValue_("\n".join(part for part in parts if part))
        else:
            self.image_view.setImage_(None)
            self.context_label.setStringValue_(
                "Bild wird aufgenommen ..." if pending else (error or "Kein Screenshot vorhanden")
            )
        self.copy_image_button.setEnabled_(bool(image_path))
        self._refresh_ready_state()

    def set_transcript(self, text, error=None):
        self.text_view.setString_(text)
        self.text_view.setEditable_(True)
        self._text_pending = False
        self._text_error = error
        self._recording = False
        self._refresh_ready_state()

    def set_live_text(self, text):
        if self._recording:
            self.text_view.setString_(text)

    def set_recording(self, recording):
        self._recording = recording
        self._refresh_ready_state()

    def _refresh_ready_state(self):
        self.copy_button.setTitle_("Fertig & kopieren" if self._recording else "Prompt kopieren")
        self.copy_button.setEnabled_(not self._image_pending and (self._recording or not self._text_pending))
        if self._image_pending:
            self.set_status("Warte auf Screenshot ...")
        elif self._recording:
            self.set_status("Aufnahme läuft. Fertig & kopieren beendet das Diktat.")
        elif self._text_pending:
            self.set_status("Diktat wird verarbeitet ...")
        else:
            self.set_status(self._text_error or self._image_error or "Bereit zum Einfügen")

    def set_status(self, message):
        self.status.setStringValue_(message)

    def close(self):
        self._closing = True
        self.panel.orderOut_(None)

    def _copy(self):
        if self.on_copy(str(self.text_view.string()), self._image_path, self._context) is not False:
            self._copied = True
            self.discard_button.setTitle_("Schließen")

    def _copy_image(self):
        if not self._image_path:
            return
        from Foundation import NSData
        data = NSData.dataWithContentsOfFile_(self._image_path)
        if data is None:
            self.set_status("Bilddatei fehlt. Bitte Screenshot erneut aufnehmen.")
            return
        pasteboard = NSPasteboard.generalPasteboard()
        pasteboard.clearContents()
        if not pasteboard.setData_forType_(data, NSPasteboardTypePNG):
            self.set_status("Bild konnte nicht kopiert werden.")
            return
        self._copied = True
        self.discard_button.setTitle_("Schließen")
        self.set_status("Bild kopiert. Im gewünschten Task mit ⌘V einfügen.")

    def _discard(self):
        if self._closing:
            return
        self.close()
        self.on_discard(self._image_path, self._copied)
