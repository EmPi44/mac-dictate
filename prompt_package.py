"""Format a reviewed dictation for a new or existing local coding task."""


def format_prompt(text, image_path, context):
    spoken = text.strip()
    if not image_path:
        return spoken
    lines = [spoken, "", "Kontext zum Screenshot:"]
    if context.get("scope"):
        lines.append("Aufnahme: " + context["scope"])
    if context.get("app"):
        lines.append("App: " + context["app"])
    if context.get("title"):
        lines.append("Fenster: " + context["title"])
    if context.get("url"):
        lines.append("URL: " + context["url"])
    lines.append("Bilddatei: " + image_path)
    lines.append("Bitte sieh dir die Bilddatei an, bevor du die Aufgabe bearbeitest.")
    return "\n".join(lines)
