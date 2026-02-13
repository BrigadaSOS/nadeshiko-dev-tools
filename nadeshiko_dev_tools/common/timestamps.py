"""Shared timestamp parsing utilities."""

import re


def parse_timestamp_to_ms(timestamp: str) -> int:
    """Convert 'H:MM:SS.ffffff' or 'H:MM:SS' timestamp to integer milliseconds."""
    match = re.match(r"(\d+):(\d+):(\d+)(?:\.(\d+))?", timestamp.strip())
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    hours, minutes, seconds, frac = match.groups()
    frac = frac or "0"
    # Normalize fractional part to microseconds (6 digits)
    frac = frac.ljust(6, "0")[:6]
    total_ms = (
        int(hours) * 3600000
        + int(minutes) * 60000
        + int(seconds) * 1000
        + int(frac) // 1000
    )
    return total_ms
