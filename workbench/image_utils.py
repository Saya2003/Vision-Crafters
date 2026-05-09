"""Normalize Gradio image inputs to RGB PIL."""

from __future__ import annotations

import numpy as np
from PIL import Image


def to_pil_rgb(image) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(np.asarray(image)).convert("RGB")
