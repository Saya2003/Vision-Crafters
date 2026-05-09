# AMD Multimodal Workbench

Single Gradio app combining three hackathon-aligned modes:

1. **Industrial inspection** — CLIP zero-shot labels + calibrated explanation copy (swap in a ROCm fine-tuned detector when you own labels).
2. **Medical workflow (educational)** — same CLIP stack as a **non-diagnostic** viewer/report workflow; use **public** chest X-ray data only.
3. **Multimodal assistant** — BLIP caption + instruct Qwen text. Set `WB_ENABLE_VLM=1` on a CUDA/ROCm/MPS box to answer with **Qwen2-VL** directly from pixels.

Each run reports a **latency table** (transparent benchmarking for judges). Optional `WB_AUDIT_JSONL` writes JSONL traces for audit-style demos without running a database.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860`.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `WB_CLIP_MODEL` | Hugging Face repo id for CLIP zero-shot (default `openai/clip-vit-base-patch32`). |
| `WB_BLIP_MODEL` | HF id for captioning (default `Salesforce/blip-image-captioning-base`). |
| `WB_QWEN_TEXT_MODEL` | Text instruct model id (default `Qwen/Qwen2.5-0.5B-Instruct`). Smaller = faster CPU laptops. |
| `WB_FORCE_CPU` | Set `1`/`true` to disable GPU paths (CPU-only Spaces). |
| `WB_ENABLE_VLM` | Set `1`/`true` on GPU hardware to route assistant answers through **Qwen2-VL** (`WB_VLM_MODEL`). |
| `WB_VLM_MODEL` | HF id such as `Qwen/Qwen2-VL-2B-Instruct` (scale up if VRAM permits). |
| `WB_AUDIT_JSONL` | Path like `data/audit.jsonl` to append UTC-stamped inference records. |

Choosing HF models:

1. Open [huggingface.co/models](https://huggingface.co/models) and filter by task (Zero-Shot Image Classification, Image-To-Text, etc.).
2. Confirm license + size constraints (Spaces/L4 versus MI300X).
3. Paste the canonical `organization/model` slug into `WB_*` vars and redeploy/restart — no code edits.

## Hugging Face Space

Set **SDK** to Gradio, entry file `app.py`, and ship `requirements.txt`. First boot downloads HF weights where configured (CLIP + BLIP + Qwen are multiple GB cumulative). For VL demos, provision a GPU Space or external runner and export `WB_ENABLE_VLM`.

## AMD Developer Cloud

Reuse the identical container; measure CLIP throughput (batch sizing) on ROCm MI300X, toggle `WB_ENABLE_VLM`, and cite latency tables exported from each tab inside your appendix.
