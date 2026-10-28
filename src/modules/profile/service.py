"""Pure preference validation and resolution (no database access).

`clamp_to_model` implements decision 12: stored thinking/effort survive
only when the effective model supports them; anything else clamps to the
model's default (null) with a warning. `validate_sources` drops unknown
source types the same way.
"""

from uuid import UUID

from .model import KNOWN_SOURCE_TYPES


def validate_sources(source_types: list[str]) -> tuple[list[str], list[str]]:
    """Keep known source types in order; return (kept, warnings)."""
    kept = [s for s in source_types if s in KNOWN_SOURCE_TYPES]
    dropped = [s for s in source_types if s not in KNOWN_SOURCE_TYPES]
    warnings = (
        [f"Dropped unknown source types: {dropped}."] if dropped else []
    )
    return kept, warnings


def clamp_to_model(
    capabilities: list[str],
    effort_levels: list[str],
    thinking: bool | None,
    effort: str | None,
    model_id: UUID | None,
) -> tuple[bool | None, str | None, list[str]]:
    """Clamp stored thinking/effort to one model's capabilities.

    Returns (thinking, effort, warnings). A missing capability or an
    unsupported level resets the field to the model's default (null) —
    the "model switch resets thinking/effort" contract.
    """
    warnings: list[str] = []
    if thinking is not None and "thinking" not in capabilities:
        who = f" model {model_id}" if model_id else " the effective model"
        warnings.append(
            f"Reset thinking to the model default:{who} lacks the "
            f"'thinking' capability."
        )
        thinking = None
    if effort is not None:
        if "effort" not in capabilities:
            who = f" model {model_id}" if model_id else " the effective model"
            warnings.append(
                f"Reset effort to the model default:{who} lacks the "
                f"'effort' capability."
            )
            effort = None
        elif effort_levels and effort not in effort_levels:
            warnings.append(
                f"Reset effort to the model default: {effort} is not in "
                f"the model's supported levels {effort_levels}."
            )
            effort = None
    return thinking, effort, warnings
