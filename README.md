---
title: Vision Crafters Workbench
emoji: 🐠
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: app.py
pinned: false
license: mit
short_description: A unified AMD-powered application for zero-shot industrial i
---

# 🚀 AMD Multimodal Workbench

Welcome to the **Multimodal Workbench**, built by Team Vision-Crafters for the AMD Developer Hackathon. 

This project is a versatile, unified AI application powered by AMD ROCm that merges three cutting-edge multimodal Vision-Language Model (VLM) workflows into a single, premium web interface. We bridge the gap between heavy AI inference and practical, real-world applications in manufacturing, healthcare education, and interactive analytical assistance.

## ✨ The Three Core Workflows

### 1. Industrial Quality Control (Zero-Shot)
- **Powered by:** CLIP (Contrastive Language-Image Pretraining)
- **How it works:** Upload an image of a manufactured part and define custom defect labels on the fly (e.g., "chipped edge", "intact"). The model ranks the probability of those defects **zero-shot**—meaning it requires absolutely no retraining or massive custom datasets.
- **Impact:** Scalable, instant visual quality assurance for factory floors.

### 2. Educational Medical Imaging Workflow
- **Powered by:** Qwen & BLIP
- **How it works:** Designed strictly for demonstrative and learning purposes using public datasets (like chest X-rays). It synthesizes complex medical imagery into structured, multi-part workflow reports.
- **Impact:** Demonstrates the future of AI as a supportive, educational tool in healthcare.

### 3. Interactive Multimodal Assistant
- **Powered by:** BLIP (Captioning) & Qwen2-VL (Reasoning)
- **How it works:** Users can upload any frame and ask intricate, pixel-grounded questions. The system intelligently routes the query based on available hardware, engaging the full visual reasoning power of Qwen2-VL when GPU acceleration is enabled.
- **Impact:** A dynamic conversational interface to query visual data instantly.

---

## 🛠️ Technology Stack & AMD Integration
- **Frontend:** Gradio (with responsive, custom dark-mode glassmorphism)
- **Framework:** PyTorch & Hugging Face Transformers
- **Hardware Acceleration:** **AMD ROCm**
- **Performance:** The entire pipeline is optimized to run on the **AMD Developer Cloud**, taking full advantage of accelerators like the MI300X. This ensures that heavy multimodal pipelines scale seamlessly and run highly efficiently on AMD infrastructure without requiring major code refactoring. 

*(Note: Each run reports a transparent latency benchmarking table directly in the UI for performance auditing).*

---

## 💻 Run Locally

To test this application on your local machine:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the server
python app.py
```
*Open `http://127.0.0.1:7860` in your browser.*

---

## ⚙️ Environment Variables

Customize the underlying models and hardware routing without changing a line of code:

| Variable | Purpose |
| --- | --- |
| `WB_CLIP_MODEL` | Hugging Face repo id for CLIP zero-shot (default: `openai/clip-vit-base-patch32`). |
| `WB_BLIP_MODEL` | HF id for captioning (default: `Salesforce/blip-image-captioning-base`). |
| `WB_QWEN_TEXT_MODEL` | Text instruct model id (default: `Qwen/Qwen2.5-0.5B-Instruct`). |
| `WB_FORCE_CPU` | Set `1` or `true` to disable GPU paths (Useful for CPU-only Spaces). |
| `WB_ENABLE_VLM` | Set `1` or `true` on GPU hardware to route assistant answers through **Qwen2-VL**. |
| `WB_VLM_MODEL` | HF id for Vision-Language model (default: `Qwen/Qwen2-VL-2B-Instruct`). |
| `WB_AUDIT_JSONL` | Path to append UTC-stamped inference records (e.g., `data/audit.jsonl`). |

---
*Built with ❤️ by Team Vision-Crafters.*
