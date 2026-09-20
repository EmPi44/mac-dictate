"""Native review card for a dictated prompt with an optional screenshot."""

from AppKit import (
    NSApp,
    NSBackingStoreBuffered,
    NSBezelStyleRounded,
    NSButton,
    NSColor,
    NSFont,
    NSFontWeightMedium,
    NSImage,
    NSImageScaleProportionallyUpOrDown,
    NSImageView,
    NSLineBreakByTruncatingTail,
    NSMakeRect,
    NSMakeSize,
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
        self._displayed_text = ""

        screen = NSScreen.mainScreen().visibleFrame()
        width = min(1040, screen.size.width - 48)
        height = min(680, screen.size.height - 48)
        rail_x = width - 270
        rail_width = 246
        image_width = rail_x - 40
        preview_height = height - 205
        x = screen.origin.x + (screen.size.width - width) / 2
        y = screen.origin.y + (screen.size.height - height) / 2
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, width, height),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
            NSBackingStoreBuffered,
            False,
        )
        self.panel.setTitle_("Review Dictation")
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

        self.heading = self._label("Dictation with Screenshot", 24, height - 48, width - 48, 28, 20)
        content.addSubview_(self.heading)
        self.hint = self._label("Speak while reviewing the image. Finish & Copy when ready.", 24, height - 72, width - 48, 18, 12)
        content.addSubview_(self.hint)

        image_heading = self._label("SCREENSHOT", 24, height - 97, image_width, 18, 11)
        image_heading.setFont_(NSFont.systemFontOfSize_weight_(11, NSFontWeightMedium))
        content.addSubview_(image_heading)
        prompt_heading = self._label("DESCRIPTION", rail_x, height - 97, rail_width, 18, 11)
        prompt_heading.setFont_(NSFont.systemFontOfSize_weight_(11, NSFontWeightMedium))
        content.addSubview_(prompt_heading)

        image_canvas = NSView.alloc().initWithFrame_(NSMakeRect(24, 105, image_width, preview_height))
        image_canvas.setWantsLayer_(True)
        image_canvas.layer().setCornerRadius_(10)
        image_canvas.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.15, 0.18, 0.22, 1).CGColor()
        )
        content.addSubview_(image_canvas)
        self.image_view = NSImageView.alloc().initWithFrame_(
            NSMakeRect(8, 8, image_width - 16, preview_height - 16)
        )
        self.image_view.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        image_canvas.addSubview_(self.image_view)

        scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(rail_x, 105, rail_width, preview_height)
        )
        scroll.setHasVerticalScroller_(True)
        scroll.setDrawsBackground_(False)
        scroll.setWantsLayer_(True)
        scroll.layer().setCornerRadius_(10)
        scroll.layer().setMasksToBounds_(True)
        self.text_view = NSTextView.alloc().initWithFrame_(
            NSMakeRect(0, 0, rail_width, preview_height)
        )
        self.text_view.setFont_(NSFont.systemFontOfSize_(13))
        self.text_view.setTextColor_(NSColor.whiteColor())
        self.text_view.setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(0.15, 0.18, 0.22, 1))
        self.text_view.setTextContainerInset_(NSMakeSize(12, 10))
        self.text_view.setRichText_(False)
        self.text_view.setVerticallyResizable_(True)
        self.text_view.setHorizontallyResizable_(False)
        self.text_view.textContainer().setWidthTracksTextView_(True)
        scroll.setDocumentView_(self.text_view)
        content.addSubview_(scroll)

        self.context_label = self._label("Capturing screenshot ...", 24, 78, width - 48, 20, 11)
        self.context_label.setLineBreakMode_(NSLineBreakByTruncatingTail)
        content.addSubview_(self.context_label)

        self.status = self._label("", 24, 49, width - 48, 22, 12)
        content.addSubview_(self.status)
        self.discard_button = self._button("Discard", 24, 12, 110, "discard:")
        self.copy_image_button = self._button("Copy Image", width - 302, 12, 120, "copyImage:")
        self.copy_button = self._button("Copy Prompt", width - 174, 12, 150, "copy:")
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
        self.discard_button.setTitle_("Discard")
        self._context = context
        self._image_path = image_path
        self._image_pending = pending
        self._text_pending = text_pending
        self._image_error = error
        self._text_error = text_error
        self._recording = recording
        self._displayed_text = text
        self._display_text(text)
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
            app = context.get("app", "")
            title = context.get("title", "")
            parts = [context.get("scope", ""), app]
            if title and title != app:
                parts.append(title)
            parts.append(context.get("url", ""))
            context_text = " · ".join(part for part in parts if part)
            self.context_label.setStringValue_(context_text)
            self.context_label.setToolTip_(context_text)
        else:
            self.image_view.setImage_(None)
            context_text = "Capturing screenshot ..." if pending else (error or "No screenshot available")
            self.context_label.setStringValue_(context_text)
            self.context_label.setToolTip_(context_text)
        self.copy_image_button.setEnabled_(bool(image_path))
        self._refresh_ready_state()

    def set_transcript(self, text, error=None):
        self._text_pending = False
        self._text_error = error
        self._recording = False
        self._displayed_text = text
        self._display_text(text)
        self.text_view.setEditable_(True)
        self._refresh_ready_state()

    def set_live_text(self, text):
        if self._recording:
            self._displayed_text = text
            self._display_text(text)

    def set_recording(self, recording):
        self._recording = recording
        self._display_text(self._displayed_text)
        self._refresh_ready_state()

    def _display_text(self, text):
        if text:
            self.text_view.setTextColor_(NSColor.whiteColor())
            self.text_view.setString_(text)
        elif self._text_pending:
            self.text_view.setTextColor_(NSColor.colorWithCalibratedWhite_alpha_(0.68, 1))
            self.text_view.setString_(
                "Keep speaking. Your words will appear here."
                if self._recording else "Finalizing dictation ..."
            )
        else:
            self.text_view.setTextColor_(NSColor.whiteColor())
            self.text_view.setString_("")

    def _refresh_ready_state(self):
        self.copy_button.setTitle_("Finish & Copy" if self._recording else "Copy Prompt")
        self.copy_button.setEnabled_(not self._image_pending and (self._recording or not self._text_pending))
        if self._image_pending:
            self.set_status("Waiting for screenshot ...")
        elif self._recording:
            self.set_status("Recording. Finish & Copy ends dictation.")
        elif self._text_pending:
            self.set_status("Processing dictation ...")
        else:
            self.set_status(self._text_error or self._image_error or "Ready to paste")

    def set_status(self, message):
        self.status.setStringValue_(message)

    def close(self):
        self._closing = True
        self.panel.orderOut_(None)

    def _copy(self):
        if self.on_copy(str(self.text_view.string()), self._image_path, self._context) is not False:
            self._copied = True
            self.discard_button.setTitle_("Close")

    def _copy_image(self):
        if not self._image_path:
            return
        from Foundation import NSData
        data = NSData.dataWithContentsOfFile_(self._image_path)
        if data is None:
            self.set_status("Image file missing. Capture the screenshot again.")
            return
        pasteboard = NSPasteboard.generalPasteboard()
        pasteboard.clearContents()
        if not pasteboard.setData_forType_(data, NSPasteboardTypePNG):
            self.set_status("Could not copy the image.")
            return
        self._copied = True
        self.discard_button.setTitle_("Close")
        self.set_status("Image copied. Paste into your task with ⌘V.")

    def _discard(self):
        if self._closing:
            return
        self.close()
        self.on_discard(self._image_path, self._copied)
