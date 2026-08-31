"""
Terminal display helpers — raw ANSI escape codes only, no colorama, no rich.

See STDLIB.md: colorama / rich -> raw ANSI escapes.

Honors two conventions the cheat sheet calls out explicitly for terminal
colour: the NO_COLOR env var (https://no-color.org/), and checking whether
stdout is actually a TTY before emitting any escape codes at all (so piping
`noembed search ... > results.txt` produces clean, colour-free text).
"""

import os
import sys

# SGR (Select Graphic Rendition) codes. Named instead of left as bare
# strings so call sites read as intent, not magic numbers.
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"

_BAR_FULL = "█"
_BAR_EMPTY = "░"
_BAR_WIDTH = 24


def _colour_enabled() -> bool:
    """
    True only if stdout is an interactive terminal AND the user hasn't
    opted out via NO_COLOR. Both checks matter: a TTY check alone would
    still colour output for someone who's explicitly asked not to have
    ANSI codes in their environment.
    """
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _wrap(text: str, code: str) -> str:
    if not _colour_enabled():
        return text
    return f"{code}{text}{_RESET}"


def bold(text: str) -> str:
    return _wrap(text, _BOLD)


def dim(text: str) -> str:
    return _wrap(text, _DIM)


def _score_colour(score: float) -> str:
    """Green for a strong match, yellow for a moderate one, red for weak."""
    if score >= 0.3:
        return _GREEN
    if score >= 0.1:
        return _YELLOW
    return _RED


def score_bar(score: float, width: int = _BAR_WIDTH) -> str:
    """
    Render a score in [0, 1] as a coloured horizontal bar of block
    characters, e.g. "██████████░░░░░░░░░░░░░░" — this exists purely to
    make relevance differences visually obvious at a glance, rather than
    requiring someone to compare four-decimal floats by eye.

    Scores are cosine similarities and can exceed 1.0 only in pathological
    edge cases that shouldn't occur with normalized TF-IDF vectors; this
    clamps defensively rather than producing a bar longer than `width`.
    """
    clamped = max(0.0, min(1.0, score))
    filled = round(clamped * width)
    bar = _BAR_FULL * filled + _BAR_EMPTY * (width - filled)
    return _wrap(bar, _score_colour(score))


def score_label(score: float) -> str:
    """Fixed-width, coloured numeric score for aligned columns."""
    text = f"{score:.4f}"
    return _wrap(text, _score_colour(score))


def contribution_bar(contribution: float, max_contribution: float, width: int = 12) -> str:
    """
    A smaller bar for per-term contributions in --explain output, scaled
    relative to the strongest contribution in that result so the bars are
    comparable within one document's breakdown.
    """
    if max_contribution <= 0:
        filled = 0
    else:
        filled = round((contribution / max_contribution) * width)
    bar = _BAR_FULL * filled + _BAR_EMPTY * (width - filled)
    return _wrap(bar, _CYAN)