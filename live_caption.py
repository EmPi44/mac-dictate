"""Non-activating caption panel. Call only from the AppKit main thread."""

from AppKit import (
    NSBackingStoreBuffered,
    NSBezelStyleRounded,
    NSButton,
    NSColor,
    NSEvent,
    NSFont,
    NSFontWeightMedium,
    NSImage,
    NSImageView,
    NSLineBreakByWordWrapping,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSTextField,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
    NSStatusWindowLevel,
)
from Foundation import NSObject


class _CameraAction(NSObject):
    def captureScreen_(self, _sender):
        self.owner._capture("screen")

    def captureRegion_(self, _sender):
        self.owner._capture("region")


def position_near_cursor(cursor, frame, width, height, gap=2):
    """Keep the stationary capture button close to the cursor and on screen."""
    left = frame.origin.x
    bottom = frame.origin.y
    right = left + frame.size.width
    top = bottom + frame.size.height
    x = cursor.x + gap
    y = cursor.y + gap
    if x + width > right:
        x = cursor.x - width - gap
    if y + height > top:
        y = cursor.y - height - gap
    return max(left, min(x, right - width)), max(bottom, min(y, top - height))


class LiveCaptionOverlay:
    def __init__(self, on_capture=None):
        self.on_capture = on_capture
        screen = NSScreen.mainScreen()
        frame = screen.visibleFrame()
        width = min(560, frame.size.width - 32)
        height = 112
        x = frame.origin.x + (frame.size.width - width) / 2
        y = frame.origin.y + 42
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, width, height),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(NSStatusWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(14)
        content.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.10, 0.12, 0.15, 0.94).CGColor()
        )
        panel.setContentView_(content)

        header = NSTextField.labelWithString_("WHISPER  ·  LIVE")
        header.setFrame_(NSMakeRect(20, height - 33, width - 150, 16))
        header.setFont_(NSFont.systemFontOfSize_weight_(11, NSFontWeightMedium))
        header.setTextColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(0.61, 0.82, 0.98, 1))
        content.addSubview_(header)

        self.screen_button = None
        self.region_button = None
        self.camera_panel = None
        self.camera_visible = False
        self.capture_queued = False
        if on_capture is not None:
            chip_width, chip_height = 222, 42
            chip = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, chip_width, chip_height),
                NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
                NSBackingStoreBuffered,
                False,
            )
            chip.setLevel_(NSStatusWindowLevel)
            chip.setOpaque_(False)
            chip.setBackgroundColor_(NSColor.clearColor())
            chip.setHasShadow_(True)
            chip.setHidesOnDeactivate_(False)
            chip.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )
            chip_content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, chip_width, chip_height))
            chip_content.setWantsLayer_(True)
            chip_content.layer().setCornerRadius_(12)
            chip_content.layer().setBackgroundColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.10, 0.12, 0.15, 0.96).CGColor()
            )
            chip.setContentView_(chip_content)
            self._camera_action = _CameraAction.alloc().init()
            self._camera_action.owner = self
            icon = NSImageView.alloc().initWithFrame_(NSMakeRect(10, 11, 21, 20))
            icon.setImage_(NSImage.imageWithSystemSymbolName_accessibilityDescription_("photo", "Image"))
            icon.setContentTintColor_(NSColor.whiteColor())
            chip_content.addSubview_(icon)

            screen_button = NSButton.alloc().initWithFrame_(NSMakeRect(36, 4, 82, 34))
            screen_button.setTitle_("Screen")
            screen_button.setBezelStyle_(NSBezelStyleRounded)
            screen_button.setTarget_(self._camera_action)
            screen_button.setAction_("captureScreen:")
            chip_content.addSubview_(screen_button)

            region_button = NSButton.alloc().initWithFrame_(NSMakeRect(120, 4, 96, 34))
            region_button.setTitle_("Region")
            region_button.setBezelStyle_(NSBezelStyleRounded)
            region_button.setTarget_(self._camera_action)
            region_button.setAction_("captureRegion:")
            chip_content.addSubview_(region_button)
            self.camera_panel = chip
            self.screen_button = screen_button
            self.region_button = region_button

        text = NSTextField.labelWithString_("")
        text.setFrame_(NSMakeRect(20, 15, width - 40, height - 50))
        text.setFont_(NSFont.systemFontOfSize_(15))
        text.setTextColor_(NSColor.whiteColor())
        text.setUsesSingleLineMode_(False)
        text.setMaximumNumberOfLines_(3)
        text.setLineBreakMode_(NSLineBreakByWordWrapping)
        text.cell().setWraps_(True)
        content.addSubview_(text)

        self.panel = panel
        self.text = text
        self.max_chars = min(140, max(60, int((width - 40) / 7.5) * 2))
        self.visible = False
        self.last_text = None

    def show(self, message, can_capture=False):
        if self.camera_panel is not None:
            if can_capture and not self.camera_visible and not self.capture_queued:
                cursor = NSEvent.mouseLocation()
                screen = next(
                    (screen for screen in NSScreen.screens()
                     if screen.frame().origin.x <= cursor.x < screen.frame().origin.x + screen.frame().size.width
                     and screen.frame().origin.y <= cursor.y < screen.frame().origin.y + screen.frame().size.height),
                    NSScreen.mainScreen(),
                )
                chip_frame = self.camera_panel.frame()
                x, y = position_near_cursor(
                    cursor, screen.visibleFrame(), chip_frame.size.width, chip_frame.size.height
                )
                self.camera_panel.setFrameOrigin_((x, y))
                self.camera_panel.orderFrontRegardless()
                self.camera_visible = True
            elif not can_capture and self.camera_visible:
                self.camera_panel.orderOut_(None)
                self.camera_visible = False
        # Whisper revises its latest words. Keep those visible as the sentence
        # grows rather than clipping the newest text beyond the panel edge.
        visible = message
        if len(visible) > self.max_chars:
            visible = visible[-self.max_chars:]
            boundary = visible.find(" ")
            visible = "…" + visible[boundary + 1:] if boundary >= 0 else "…" + visible
        if visible != self.last_text:
            self.text.setStringValue_(visible)
            self.last_text = visible
        if not self.visible:
            self.panel.orderFrontRegardless()
            self.visible = True

    def hide(self):
        self.capture_queued = False
        if self.camera_visible:
            self.camera_panel.orderOut_(None)
            self.camera_visible = False
        if self.visible:
            self.panel.orderOut_(None)
            self.visible = False

    def _capture(self, mode):
        self.capture_queued = True
        if self.camera_visible:
            self.camera_panel.orderOut_(None)
            self.camera_visible = False
        if self.on_capture is not None:
            self.on_capture(mode)
        self.capture_queued = False
