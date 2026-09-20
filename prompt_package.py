"""Format a reviewed dictation for a new or existing local coding task."""


def format_prompt(text, image_path, context):
    spoken = text.strip()
    if not image_path:
        return spoken
    lines = [spoken, "", "Screenshot context:"]
    if context.get("scope"):
        lines.append("Capture: " + context["scope"])
    if context.get("app"):
        lines.append("App: " + context["app"])
    if context.get("title"):
        lines.append("Window: " + context["title"])
    if context.get("url"):
        lines.append("URL: " + context["url"])
    lines.append("Image file: " + image_path)
    lines.append("Please inspect the image before working on the task.")
    return "\n".join(lines)
