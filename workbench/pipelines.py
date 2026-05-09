"""
Lazy-loaded HF pipelines so import stays light and failures are localized.

All run_* helpers return inference timings alongside results for hackathon benchmarking UX.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_clip_pipe = None
_blip_pipe = None
_qwen_text_bundle = None
_qwen_vl_bundle = None


def _latency_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


def _torch_model_device(module):
    import torch

    p = next(module.parameters(), None)
    if p is None:
        return torch.device("cpu")

    return p.device


def _hf_pipeline_device():
    """Pick a transformers `pipeline(..., device=...)` argument."""

    from workbench.config import FORCE_CPU
    import torch

    if FORCE_CPU:
        return -1
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return -1


def _prepare_vision_image(image, max_side: int = 640):
    """
    Keep inference responsive on CPU by bounding image resolution.
    """
    if image is None:
        return image
    try:
        w, h = image.size
        if max(w, h) <= max_side:
            return image
        scale = max_side / float(max(w, h))
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size)
    except Exception:
        return image


def fmt_latency_md(prefix: str, timings_ms: dict[str, float]) -> str:
    if not timings_ms:
        return ""

    rows = "".join(f"<tr><td>{k}</td><td style='text-align:right'>{timings_ms[k]:.1f}&nbsp;ms</td></tr>" for k in sorted(timings_ms))
    return (
        f"\n\n#### {prefix}\n<table class='wb-lat'><thead><tr><th>Step</th><th style='text-align:right'>Time</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>\n"
    )


def get_clip_zero_shot():
    global _clip_pipe
    if _clip_pipe is None:
        from transformers import pipeline

        from workbench.config import CLIP_MODEL_ID

        _clip_pipe = pipeline(
            "zero-shot-image-classification",
            model=CLIP_MODEL_ID,
            device=_hf_pipeline_device(),
        )
    return _clip_pipe


def get_blip_captioner():
    global _blip_pipe
    if _blip_pipe is None:
        from transformers import pipeline

        from workbench.config import BLIP_MODEL_ID

        _blip_pipe = pipeline(
            "image-to-text",
            model=BLIP_MODEL_ID,
            device=_hf_pipeline_device(),
        )
    return _blip_pipe


def get_qwen_text_bundle():
    global _qwen_text_bundle
    if _qwen_text_bundle is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from workbench.config import QWEN_TEXT_MODEL_ID
        from workbench.runtime import use_gpu_inference

        model_id = QWEN_TEXT_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if use_gpu_inference():
            dtype = torch.float16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
                trust_remote_code=True,
            )

        model.eval()
        _qwen_text_bundle = (model, tokenizer)

    return _qwen_text_bundle


def get_qwen_vl_bundle():
    global _qwen_vl_bundle
    if _qwen_vl_bundle is None:
        import torch

        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Qwen2-VL requires a recent transformers with Qwen2-VL support. "
                "`pip install -U transformers` (often >= 4.45) fixes this."
            ) from e

        from workbench.config import VLM_MODEL_ID

        processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)

        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.eval()
        _qwen_vl_bundle = (model, processor)

    return _qwen_vl_bundle


def run_clip_labels(
    image,
    labels: list[str],
    hypothesis_template: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    pipe = get_clip_zero_shot()
    timings["pipeline_load_clip_ms"] = _latency_ms(t0)

    t1 = time.perf_counter()
    kwargs = {"candidate_labels": labels}
    if hypothesis_template:
        kwargs["hypothesis_template"] = hypothesis_template
    results = pipe(_prepare_vision_image(image), **kwargs)
    timings["clip_forward_ms"] = _latency_ms(t1)

    return results, timings


def run_blip_caption(image) -> tuple[str, dict[str, float]]:
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    pipe = get_blip_captioner()
    timings["pipeline_load_blip_ms"] = _latency_ms(t0)

    t1 = time.perf_counter()
    out = pipe(_prepare_vision_image(image))

    timings["blip_caption_ms"] = _latency_ms(t1)

    if isinstance(out, list) and out and isinstance(out[0], dict):
        text = out[0].get("generated_text", str(out[0]))
        return text, timings

    return str(out), timings


def run_qwen_answer(system: str, user: str, max_new_tokens: int = 256) -> tuple[str, dict[str, float]]:
    import torch

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    model, tokenizer = get_qwen_text_bundle()
    timings["qwen_text_load_bundle_ms"] = _latency_ms(t0)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    t1 = time.perf_counter()

    inputs = tokenizer(prompt, return_tensors="pt")
    target = _torch_model_device(model)
    inputs = {k: v.to(target) for k, v in inputs.items()}
    timings["qwen_prompt_tokenize_ms"] = _latency_ms(t1)

    t2 = time.perf_counter()

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    timings["qwen_generate_ms"] = _latency_ms(t2)

    new_tokens = out_ids[0, inputs["input_ids"].shape[1] :]

    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, timings


def run_qwen_vl_answer(image, user_prompt: str, max_new_tokens: int = 256) -> tuple[str, dict[str, float]]:
    import torch

    from workbench.runtime import use_gpu_inference

    if not use_gpu_inference():  # pragma: no cover
        raise RuntimeError("Vision-language inference is disabled without a GPU accelerator.")

    timings: dict[str, float] = {}

    sys_line = (
        "You are a careful multimodal assistant. Ground answers in the image; if uncertain, say so. "
        "Keep replies concise — bullets when helpful."
    )
    user_blob = sys_line.strip() + "\n\nQuestion:\n" + user_prompt.strip()

    t0 = time.perf_counter()
    model, processor = get_qwen_vl_bundle()
    timings["qwen_vl_load_bundle_ms"] = _latency_ms(t0)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_blob},
            ],
        }
    ]

    t1 = time.perf_counter()

    text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    proc_inputs = processor(
        text=[text_prompt],
        images=[_prepare_vision_image(image, max_side=896)],
        padding=True,
        return_tensors="pt",
    )

    timings["qwen_vl_preprocess_ms"] = _latency_ms(t1)

    t2 = time.perf_counter()

    proc_inputs = proc_inputs.to(_torch_model_device(model))

    with torch.inference_mode():
        out_ids = model.generate(
            **proc_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )

    timings["qwen_vl_generate_ms"] = _latency_ms(t2)

    trimmed = [o[len(i) :] for i, o in zip(proc_inputs.input_ids, out_ids)]

    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    ans = decoded[0].strip() if decoded else ""

    return ans, timings
