"""
Multimodal assistant:

- Portable path: BLIP caption + Qwen text conditioned on caption.
- Accelerator path (WB_ENABLE_VLM=1 + GPU): Qwen2-VL answers directly from pixels — stronger multimodal judging story.
"""

from __future__ import annotations

from PIL import Image

from workbench.audit import append_audit
from workbench.config import ENABLE_VLM
from workbench.pipelines import fmt_latency_md, run_blip_caption, run_qwen_answer, run_qwen_vl_answer
from workbench.runtime import use_gpu_inference

SYSTEM_QA_CAPTION = (
    "You are a careful technical assistant for a vision demo. "
    "You only answer based on the provided image description; if the question cannot be answered from it, say so. "
    "Do not invent fine-grained visual details not supported by the description. "
    "Keep answers concise and structured with bullets when helpful."
)


def describe_frame(image: Image.Image | None) -> tuple[str, str]:
    if image is None:
        return "", "Upload an image first."

    caption, timings = run_blip_caption(image)

    vl_hint = ""

    if ENABLE_VLM:
        if use_gpu_inference():
            vl_hint = "\n\nVision-language mode is **on** (`WB_ENABLE_VLM`) — GPU answers can use pixels directly.\n"
        else:
            vl_hint = (
                "\n\n`WB_ENABLE_VLM` is set but **no accelerator** detected — replies stay caption-conditioned "
                "until you run on ROCm/CUDA/MPS hardware.\n"
            )

    md = (
        f"### Frame description (BLIP)\n\n{caption}\n\n"
        "---\n"
        "- Ask questions below.\n"
        f"- Caption-conditioned path: **Qwen** text interprets BLIP prose.{vl_hint}"
    )

    md += fmt_latency_md("Latency", timings)

    append_audit({"mode": "assistant_describe", "timings_ms": timings, "caption_len": len(caption)})

    return caption, md


def answer_question(caption: str, question: str, image: Image.Image | None = None) -> str:
    stripped_q = (question or "").strip()

    if ENABLE_VLM and use_gpu_inference() and image is not None:
        try:
            text, timings = run_qwen_vl_answer(image, stripped_q or "Summarize notable objects or risks.")

            md = f"### Qwen2-VL (vision-grounded)\n\n{text}"
            md += fmt_latency_md("Latency", timings)

            append_audit({"mode": "assistant_vlm", "timings_ms": timings})

            return md

        except Exception as exc:  # pragma: no cover
            appendix = (
                f"\n\n**Vision-language step failed**, falling back to caption-conditioned reasoning: `{exc}`\n\n"
                "---"
            )
    else:
        appendix = ""

    if not stripped_q:
        return "Enter a question." + appendix

    if not caption.strip():
        if ENABLE_VLM and not use_gpu_inference():
            return (
                "GPU vision-language replies are unavailable on this runtime. Describe the frame first for "
                "caption-conditioned answers, or run with ROCm/CUDA/MPS (`WB_ENABLE_VLM`)."
                + appendix
            )

        return "Click **Describe this frame** first to build caption context." + appendix

    user_block = f"Image description (may be incomplete):\n{caption.strip()}\n\nUser question:\n{stripped_q}"

    try:
        ans, timings = run_qwen_answer(SYSTEM_QA_CAPTION, user_block, max_new_tokens=384)

        md = "### Qwen (caption-conditioned)\n\n" + ans + appendix
        md += fmt_latency_md("Latency", timings)

        append_audit({"mode": "assistant_caption_llm", "timings_ms": timings})

        return md

    except Exception as exc:
        append_audit({"mode": "assistant_error", "error": str(exc)})

        return (
            f"**Model error:** {exc}\n\n"
            "Install matching torch wheels for your platform, confirm RAM/VRAM budget, "
            "or tighten `WB_QWEN_TEXT_MODEL` to a smaller instruct checkpoint."
            + appendix
        )
