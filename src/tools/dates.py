"""Current-date helper for prompts (AR-7 task 1).

"Latest X" queries fly blind when the model has no idea what today is.
Every prompt that benefits from recency awareness gets a date line, and the
date is an injectable argument so tests can freeze the clock.
"""

from datetime import UTC, datetime


def current_date_line(today: str | None = None) -> str:
    """`Current date: YYYY-MM-DD.` with an injectable clock (AR-7)."""
    if not today:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"Current date: {today}."
