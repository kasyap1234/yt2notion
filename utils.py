"""Small shared helpers."""


def timestamp_to_seconds(ts):
    """Parse 'MM:SS' or 'HH:MM:SS' (also tolerates a bare number of seconds)
    into a float number of seconds."""
    ts = str(ts).strip()
    if ts.replace(".", "", 1).isdigit():
        return float(ts)
    parts = [float(p) for p in ts.split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds
