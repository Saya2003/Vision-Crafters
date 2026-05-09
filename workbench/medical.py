"""Educational medical imaging workflow — viewer + report text, no clinical claims."""

from __future__ import annotations

from PIL import Image

from workbench.audit import append_audit
from workbench.disclaimers import MEDICAL_DISCLAIMER
from workbench.pipelines import fmt_latency_md, run_clip_labels

# Workflow-oriented labels: software demo language, not diagnostic wording.
WORKFLOW_LABELS = [
    "chest X-ray style image with generally clear lung fields for educational viewing",
    "chest X-ray style image with focal opacity or consolidation patterns worth highlighting in a viewer",
    "chest X-ray style image with diffuse interstitial or reticular patterns worth highlighting in a viewer",
    "non-chest image or not suitable for the educational chest workflow",
]


def analyze_medical_workflow(image: Image.Image | None, agreed: bool) -> str:
    if not agreed:
        return f"{MEDICAL_DISCLAIMER}\n\n**Check the acknowledgment box** to enable this educational tab."

    if image is None:
        return "Upload a **public** chest X-ray style image for the educational viewer workflow."

    try:
        results, timings = run_clip_labels(image, WORKFLOW_LABELS)
    except Exception as exc:
        return (
            f"{MEDICAL_DISCLAIMER}\n\n"
            "### Educational workflow report\n\n"
            f"Model call failed: `{exc}`\n\n"
            "If this is the first run, model files may still be downloading. "
            "Retry once download/network activity settles."
        )

    if not results:
        return (
            f"{MEDICAL_DISCLAIMER}\n\n"
            "### Educational workflow report\n\n"
            "No model scores were returned. "
            "Try another sample image and run again."
        )
    top = results[0]

    lines = [MEDICAL_DISCLAIMER, "", "### Educational workflow report (non-diagnostic)\n"]
    lines.append("| Workflow tag | Score |")
    lines.append("| --- | ---: |")

    for r in results:
        label = str(r.get("label", ""))
        score = float(r.get("score", 0.0))
        lines.append(f"| {label} | {score:.3f} |")

    lines.append("\n#### Viewer notes (documentation only)\n")

    lines.append(
        f"- **Suggested UI emphasis:** the highest-ranked tag is `{top.get('label')}` — use this only to decide "
        "what *regions or overlays* your demo viewer should emphasize (e.g., pan/zoom to lung fields), not to infer disease.\n"
        "- **Report text (template):** “Educational demo: image reviewed in software workflow. "
        "No clinical interpretation. Follow institutional policies for real reads.”\n"
        "- **Data:** use public datasets (e.g., NIH Chest X-ray, CheXpert-style public releases) and cite the source in your README.\n"
    )

    blob = "\n".join(lines) + fmt_latency_md("Latency", timings)

    append_audit(
        {
            "mode": "medical_educational",
            "timings_ms": timings,
            "viewer_tag_top": str(top.get("label", "")),
        }
    )

    return blob
