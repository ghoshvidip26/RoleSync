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


# --------------------------------------------------------------------------------------------------
# Choosing how a tailored resume is drawn: the user's own design, or a template from the gallery.

class TemplateChoiceRequired(ValueError):
    """The user must pick a template (their own design is unavailable or cannot be reproduced)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


ORIGINAL = "original"


def select_renderer(template: str | None, original_pdf: bytes | None) -> tuple[BaseResumeRenderer, str]:
    """(renderer, name) for the requested template.

    ``template`` is a gallery template id, ``"original"`` (rebuild the uploaded design), or None, which
    means "the uploaded design if there is one". Raises TemplateChoiceRequired when the user has to pick,
    and ValueError for an unknown template id.
    """
    from app.services.resume_pdf_renderers.pdf_preserving_renderer import PdfPreservingRenderer
    from app.services.resume_pdf_renderers.template_renderer import TemplateRenderer

    name = (template or "").strip().lower()
    if name in {"", ORIGINAL}:
        if not original_pdf:
            raise TemplateChoiceRequired(
                "No original resume PDF is stored for this profile, so its own design cannot be used. "
                "Please select a resume template.")
        return PdfPreservingRenderer(original_pdf), ORIGINAL
    return TemplateRenderer(name), f"template:{name}"
