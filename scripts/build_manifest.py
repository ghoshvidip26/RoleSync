"""Build a layout manifest from a resume PDF and print a summary (a manual inspection tool, not a test).

    python3 scripts/build_manifest.py                      # tests/fixtures/Vidip_Resume.pdf
    python3 scripts/build_manifest.py path/to/resume.pdf

The full manifest is written to scratch/resume_layout_manifest.json (git-ignored).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.resume_pdf_renderers.layout_manifest import build_layout_manifest  # noqa: E402


def main() -> None:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "fixtures" / "Vidip_Resume.pdf"
    output_path = ROOT / "scratch" / "resume_layout_manifest.json"
    output_path.parent.mkdir(exist_ok=True)

    manifest = build_layout_manifest(pdf_path.read_bytes())
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Manifest extraction successful.")
    print("Pages:", manifest["document"]["page_count"])
    print("Layout:", manifest["document"]["layout_type"])
    print("Sections:")
    for section in manifest["sections"]:
        print(f"- {section['name']}: {len(section['lines'])} lines")
    print("Spans:", sum(len(p["spans"]) for p in manifest["pages"]))
    print("Lines:", sum(len(p["lines"]) for p in manifest["pages"]))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
