"""
Environment-driven model IDs and toggles — no rebuild required between CPU laptop and GPU cloud.

WB_CLIP_MODEL — HF model id for CLIP zero-shot (default: openai/clip-vit-base-patch32)
WB_BLIP_MODEL — HF model id for image captioning (default: Salesforce/blip-image-captioning-base)
WB_QWEN_TEXT_MODEL — text instruct model for caption-conditioned Q&A (default: Qwen/Qwen2.5-0.5B-Instruct)

WB_FORCE_CPU — set to 1/true to pin everything CPU (Spaces / smoke tests).

WB_ENABLE_VLM — set to 1/true on a CUDA/ROCm/MPS machine to answer assistant questions with Qwen2-VL on the pixels.
WB_VLM_MODEL — HF id (default: Qwen/Qwen2-VL-2B-Instruct). Use larger weights on MI300X if VRAM allows.

WB_AUDIT_JSONL — optional path (e.g. data/audit.jsonl). Appends one JSON object per inference for QC / demo traceability.

Picking models on Hugging Face:
  1. Open huggingface.co/models and filter by task (Zero-Shot Image Classification, Image-To-Text, etc.).
  2. Prefer models with permissive licenses and Spaces-compatible sizes.
  3. Paste the repo id (org/name) into the matching WB_* variable and restart the app.
"""

from __future__ import annotations

import os

TRUEISH = frozenset({"1", "true", "yes", "on"})


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in TRUEISH


CLIP_MODEL_ID = os.environ.get("WB_CLIP_MODEL", "openai/clip-vit-base-patch32")
BLIP_MODEL_ID = os.environ.get("WB_BLIP_MODEL", "Salesforce/blip-image-captioning-base")
QWEN_TEXT_MODEL_ID = os.environ.get("WB_QWEN_TEXT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

FORCE_CPU = _truthy(os.environ.get("WB_FORCE_CPU"))

ENABLE_VLM = _truthy(os.environ.get("WB_ENABLE_VLM"))
VLM_MODEL_ID = os.environ.get("WB_VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")

AUDIT_JSONL_PATH = (os.environ.get("WB_AUDIT_JSONL") or "").strip()


def summarize() -> dict[str, str | bool]:
    return {
        "WB_CLIP_MODEL": CLIP_MODEL_ID,
        "WB_BLIP_MODEL": BLIP_MODEL_ID,
        "WB_QWEN_TEXT_MODEL": QWEN_TEXT_MODEL_ID,
        "WB_FORCE_CPU": FORCE_CPU,
        "WB_ENABLE_VLM": ENABLE_VLM,
        "WB_VLM_MODEL": VLM_MODEL_ID,
        "WB_AUDIT_JSONL": AUDIT_JSONL_PATH or "(disabled)",
    }
