from typing import Any

from app.services.resume_pdf_renderers.base import BaseResumeRenderer

# The smallest layout-detection confidence we trust. The layout analyzer uses the same value when it
# decides between "preserve_layout" and "request_template_choice", so change them together.
MIN_LAYOUT_CONFIDENCE = 0.65


def get_renderer(layout_metadata: dict[str, Any]) -> BaseResumeRenderer:
    """The renderer for a detected resume layout, or a ValueError saying why there isn't one."""
    if not isinstance(layout_metadata, dict):
        raise ValueError("Layout metadata must be a dictionary. Please select a resume template.")

    layout_type = str(layout_metadata.get("layout_type", "complex")).strip().lower()

    raw_confidence = layout_metadata.get("confidence", 0)
    try:
        if isinstance(raw_confidence, bool):
            raise TypeError("a boolean is not a confidence")
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        raise ValueError(
            "Layout detection confidence is missing or not a number. "
            "Please select a resume template."
        ) from None

    # Written so that NaN also fails: any comparison with NaN is False.
    if not confidence >= MIN_LAYOUT_CONFIDENCE:
        raise ValueError(
            "Layout detection confidence is too low. "
            "Please select a resume template."
        )

    if layout_type == "single_column":
        from app.services.resume_pdf_renderers.renderers.single_column import SingleColumnRenderer

        return SingleColumnRenderer()

    if layout_type == "two_column":
        from app.services.resume_pdf_renderers.renderers.two_column import TwoColumnRenderer

        return TwoColumnRenderer()

    raise ValueError(
        "This resume layout is not supported for automatic rendering."
    )
