"""Industrial inspection mode: zero-shot visual labels + structured explanation."""

from __future__ import annotations

from PIL import Image

from workbench.audit import append_audit
from workbench.pipelines import fmt_latency_md, run_clip_labels

DEFAULT_LABELS = [
    "normal product surface with no visible defect",
    "scratch, dent, or physical damage on the surface",
    "stain, contamination, or foreign material",
    "crack, fracture, or broken area",
    "misalignment, assembly error, or wrong orientation",
]


def analyze_industrial(image: Image.Image | None, custom_labels: str | None = None) -> str:
    if image is None:
        return "Upload an image to run inspection."

    labels = DEFAULT_LABELS

    if custom_labels and custom_labels.strip():
        labels = [x.strip() for x in custom_labels.split(",") if x.strip()]

    try:
        results, timings = run_clip_labels(image, labels)
    except Exception as exc:
        return (
            "### Industrial inspection\n\n"
            f"Model call failed: `{exc}`\n\n"
            "If this is your first run, model weights may still be downloading. "
            "Wait a bit and click **Run inspection** again."
        )

    if not results:
        return (
            "### Industrial inspection\n\n"
            "No scores were returned by the model.\n\n"
            "Try a different image, or retry after model download completes."
        )
    lines = ["### Industrial inspection (zero-shot)\n"]
    lines.append("| Label | Score |")
    lines.append("| --- | ---: |")

    for r in results[:8]:
        label = str(r.get("label", ""))
        score = float(r.get("score", 0.0))
        lines.append(f"| {label} | {score:.3f} |")

    top = results[0]
    lines.append("\n#### Explanation panel\n")

    lines.append(
        f"- **Top hypothesis:** `{top.get('label')}` ({float(top.get('score', 0)):.1%} CLIP similarity).\n"
        "- **How to read this:** CLIP scores are *not* calibrated defect probabilities; they rank how well "
        "each text description matches the image. Use them for triage and UX, then validate with labeled data "
        "or a fine-tuned detector before production.\n"
        "- **AMD angle:** batch many frames/images on MI300X to raise throughput for line inspection; "
        "pair with a fine-tuned ROCm model when you have domain labels.\n"
    )

    blob = "\n".join(lines) + fmt_latency_md("Latency", timings)

    append_audit(
        {
            "mode": "industrial",
            "timings_ms": timings,
            "labels": labels,
            "top_label": str(top.get("label", "")),
        }
    )

    return blob
