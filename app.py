"""
AMD Multimodal Workbench — three modes in one Gradio app:
industrial inspection, educational medical workflow, multimodal assistant.
"""

from __future__ import annotations

import logging

import gradio as gr

try:
    # Optional convenience: load `.env` if present.
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:
    pass

from workbench.assistant import answer_question, describe_frame
from workbench.config import summarize
from workbench.image_utils import to_pil_rgb
from workbench.runtime import accelerator_kind, use_gpu_inference
from workbench.industrial import analyze_industrial
from workbench.medical import analyze_medical_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Polished dark UI: midnight surfaces, vibrant gradient accents, readable type, and glassmorphism.
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

/* --- Shell --- */
.gradio-container {
  font-family: 'Inter', sans-serif !important;
  max-width: 1180px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
footer { display: none !important; }

h1, h2, h3, h4, h5, h6, .wb-eyebrow, .wb-pill {
  font-family: 'Outfit', sans-serif !important;
}

/* --- Hero --- */
@keyframes gradientBg {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.wb-hero {
  position: relative;
  overflow: hidden;
  border-radius: 24px;
  padding: 2.5rem 2.5rem 2.25rem;
  margin-bottom: 2rem;
  border: 1px solid rgba(255,255,255,0.08);
  background:
    radial-gradient(1000px circle at 15% 20%, rgba(139, 92, 246, 0.25), transparent 50%),
    radial-gradient(800px circle at 85% 80%, rgba(14, 165, 233, 0.2), transparent 50%),
    linear-gradient(135deg, #09090b 0%, #18181b 50%, #09090b 100%);
  background-size: 200% 200%;
  animation: gradientBg 15s ease infinite;
  box-shadow: 0 32px 64px rgba(0,0,0,0.5), inset 0 1px 1px rgba(255,255,255,0.1);
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.wb-hero:hover {
  transform: translateY(-2px);
}

.wb-hero .wb-eyebrow {
  font-size: 0.75rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #a78bfa;
  margin: 0 0 0.75rem;
  font-weight: 700;
  text-shadow: 0 2px 10px rgba(167,139,250,0.3);
}

.wb-hero h1 {
  font-size: 2.25rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0 0 0.75rem;
  line-height: 1.1;
  color: #f8fafc;
  background: linear-gradient(to right, #ffffff, #a5b4fc);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.wb-hero .wb-sub {
  margin: 0;
  max-width: 56rem;
  color: rgba(226, 232, 240, 0.85);
  line-height: 1.6;
  font-size: 1.05rem;
}

.wb-hero .wb-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin-top: 1.5rem;
}

.wb-hero .wb-pill {
  font-size: 0.8rem;
  font-weight: 600;
  padding: 0.4rem 0.8rem;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.03);
  backdrop-filter: blur(8px);
  color: #e2e8f0;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: default;
}

.wb-hero .wb-pill:hover {
  background: rgba(255,255,255,0.1);
  border-color: rgba(255,255,255,0.25);
  transform: translateY(-1px);
  color: #ffffff;
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* --- Tab panels --- */
.wb-tab-intro {
  color: rgba(226, 232, 240, 0.78);
  line-height: 1.55;
  margin-bottom: 0.85rem !important;
  font-size: 0.95rem;
}
.wb-panel {
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  background: rgba(15, 23, 42, 0.35) !important;
  padding: 1rem !important;
}
.wb-panel-out {
  min-height: 200px;
}

/* --- Interactive Elements (Buttons & Inputs) --- */
button.primary {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
  border: none !important;
  box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
  transition: transform 0.2s, box-shadow 0.2s, filter 0.2s !important;
  color: white !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
}
button.primary:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
  filter: brightness(1.1) !important;
}
button.primary:active {
  transform: translateY(0) !important;
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4) !important;
}

button.secondary {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  color: #f1f5f9 !important;
  transition: transform 0.2s, box-shadow 0.2s, background 0.2s, border-color 0.2s !important;
}
button.secondary:hover {
  background: rgba(255,255,255,0.1) !important;
  border-color: rgba(255,255,255,0.2) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 16px rgba(0,0,0,0.2) !important;
}
button.secondary:active {
  transform: translateY(0) !important;
}

textarea, input {
  transition: border-color 0.2s, box-shadow 0.2s, background 0.2s !important;
  background: rgba(0,0,0,0.2) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
}
textarea:focus, input:focus {
  border-color: #8b5cf6 !important;
  box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.25) !important;
  background: rgba(0,0,0,0.3) !important;
}

/* --- Markdown results (tables + headings) --- */
.wb-panel-out .prose table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
.wb-panel-out .prose th, .wb-panel-out .prose td {
  border-bottom: 1px solid rgba(255,255,255,0.05);
  padding: 0.6rem 0.5rem;
}
.wb-panel-out .prose th { color: #a78bfa; font-weight: 600; text-align: left; }
.wb-panel-out .prose h3, .wb-panel-out .prose h4 { margin-top: 0.75rem; color: #f8fafc; font-weight: 600; }

.wb-panel-out table.wb-lat { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 0.5rem; }
.wb-panel-out table.wb-lat th, .wb-panel-out table.wb-lat td {
  border-bottom: 1px solid rgba(255,255,255,0.05);
  padding: 0.5rem 0.4rem;
}
.wb-panel-out table.wb-lat thead th { font-weight: 600; color: #a78bfa; text-align: left; }

/* --- Footer tip --- */
.wb-footer {
  margin-top: 1.5rem;
  padding: 1.25rem 1.5rem;
  border-radius: 16px;
  border: 1px dashed rgba(255,255,255,0.1);
  background: rgba(2, 6, 23, 0.4);
  color: #94a3b8;
  font-size: 0.95rem;
  line-height: 1.6;
  text-align: center;
}
"""

WB_THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_lg,
).set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_50",
    block_title_text_color="*neutral_100",
    block_label_text_color="*neutral_200",
    block_border_width="0px",
    input_border_width="1px",
    button_large_text_weight="600",
)


def tab_industrial(image, custom_labels):
    pil = to_pil_rgb(image)
    if pil is None:
        return "Upload an image to run inspection."
    # Stream a quick status update so the UI feels responsive on first model warmup.
    yield "### Industrial inspection\n\nStarting model inference... first run can take ~30-90s while weights warm up."
    yield analyze_industrial(pil, custom_labels)


def tab_medical(image, agreed):
    pil = to_pil_rgb(image)
    if pil is None and agreed:
        return "Upload a **public** chest X-ray style image for the educational viewer workflow."
    yield "### Educational workflow report\n\nStarting analysis... first run can take ~30-90s while weights warm up."
    yield analyze_medical_workflow(pil, agreed)


def tab_assistant_describe(image):
    pil = to_pil_rgb(image)
    caption, md = describe_frame(pil)
    return caption, md


def tab_assistant_ask(caption, question, image):
    pil = to_pil_rgb(image)
    return answer_question(caption or "", question or "", pil)


def runtime_info_md() -> str:
    cfg = summarize()
    accel = accelerator_kind()
    gpu = use_gpu_inference()
    bullets = (
        "| Key | Value |\n"
        "| --- | --- |\n"
        f"| Accelerator | `{accel}` |\n"
        f"| GPU inference allowed | `{gpu}` |\n"
        f"| `WB_CLIP_MODEL` | `{cfg['WB_CLIP_MODEL']}` |\n"
        f"| `WB_BLIP_MODEL` | `{cfg['WB_BLIP_MODEL']}` |\n"
        f"| `WB_QWEN_TEXT_MODEL` | `{cfg['WB_QWEN_TEXT_MODEL']}` |\n"
        f"| `WB_ENABLE_VLM` | `{cfg['WB_ENABLE_VLM']}` |\n"
        f"| `WB_VLM_MODEL` | `{cfg['WB_VLM_MODEL']}` |\n"
        f"| `WB_AUDIT_JSONL` | `{cfg['WB_AUDIT_JSONL']}` |\n"
    )
    hint = ""
    if bool(cfg["WB_ENABLE_VLM"]) and not gpu:
        hint = (
            "\nVision-language replies need ROCm/CUDA/MPS (`WB_FORCE_CPU` off). Caption-only Q&A still runs on CPU.\n"
        )

    return (
        "### Runtime snapshot\nReload the Space after tweaking env vars.\n\n"
        + bullets
        + hint
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AMD Multimodal Workbench") as demo:
        gr.HTML(
            """
<div class="wb-hero">
  <p class="wb-eyebrow">AMD Developer Hackathon · Vision &amp; multimodal</p>
  <h1>Multimodal Workbench</h1>
  <p class="wb-sub">
    One polished surface for three demos: industrial inspection, an educational imaging workflow,
    and a multimodal assistant. CLIP + BLIP + instruct Qwen text ship everywhere;
    toggle <strong>Qwen2-VL</strong> with <code>WB_ENABLE_VLM=1</code> on ROCm/CUDA for pixel-grounded answers.
  </p>
  <div class="wb-pills">
    <span class="wb-pill">Industrial QC</span>
    <span class="wb-pill">Educational medical viewer</span>
    <span class="wb-pill">Caption + Qwen Q&amp;A</span>
    <span class="wb-pill">ROCm / AMD Developer Cloud ready</span>
  </div>
</div>
            """.strip()
        )

        caption_state = gr.State("")

        with gr.Tabs():
            with gr.Tab("Industrial inspection", id="tab-ind"):
                gr.Markdown(
                    "Zero-shot CLIP ranking plus an explanation panel. Override labels for your own defect vocabulary.",
                    elem_classes=["wb-tab-intro"],
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5, min_width=280):
                        with gr.Group(elem_classes=["wb-panel"]):
                            ind_img = gr.Image(
                                type="pil",
                                label="Inspection image",
                                height=320,
                                image_mode="RGB",
                            )
                            ind_labels = gr.Textbox(
                                label="Custom labels (optional)",
                                placeholder="Comma-separated, e.g. chip on ceramic, intact glaze, kiln crack",
                                lines=2,
                            )
                            ind_btn = gr.Button("Run inspection", variant="primary", size="lg")
                    with gr.Column(scale=6, min_width=280):
                        with gr.Group(elem_classes=["wb-panel", "wb-panel-out"]):
                            gr.Markdown("#### Results")
                            ind_out = gr.Markdown()
                ind_btn.click(tab_industrial, [ind_img, ind_labels], ind_out)

            with gr.Tab("Medical workflow", id="tab-med"):
                gr.Markdown(
                    "Public data only. **Not for clinical use** — checkbox required before running.",
                    elem_classes=["wb-tab-intro"],
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5, min_width=280):
                        with gr.Group(elem_classes=["wb-panel"]):
                            med_agree = gr.Checkbox(
                                label="I understand this tab is educational and not for diagnosis",
                                value=False,
                            )
                            med_img = gr.Image(
                                type="pil",
                                label="Chest X-ray style image (public dataset)",
                                height=320,
                                image_mode="RGB",
                            )
                            med_btn = gr.Button("Generate workflow report", variant="primary", size="lg")
                    with gr.Column(scale=6, min_width=280):
                        with gr.Group(elem_classes=["wb-panel", "wb-panel-out"]):
                            gr.Markdown("#### Workflow report")
                            med_out = gr.Markdown()
                med_btn.click(tab_medical, [med_img, med_agree], med_out)

            with gr.Tab("Multimodal assistant", id="tab-asst"):
                gr.Markdown(
                    "**Step 1:** Describe (BLIP, optional warmup). "
                    "**Step 2:** Ask — on GPU + `WB_ENABLE_VLM`, **Qwen2-VL** reads the frame directly; "
                    "otherwise **Qwen** reasons from the caption.",
                    elem_classes=["wb-tab-intro"],
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5, min_width=280):
                        with gr.Group(elem_classes=["wb-panel"]):
                            asst_img = gr.Image(
                                type="pil",
                                label="Image or frame",
                                height=280,
                                image_mode="RGB",
                            )
                            asst_desc_btn = gr.Button("Describe this frame", variant="primary", size="lg")
                            asst_desc = gr.Markdown()
                            asst_q = gr.Textbox(
                                label="Your question",
                                placeholder="What materials or defects might relate to this scene?",
                                lines=3,
                            )
                            asst_ask_btn = gr.Button("Ask assistant (VLM if enabled)", variant="secondary", size="lg")
                    with gr.Column(scale=6, min_width=280):
                        with gr.Group(elem_classes=["wb-panel", "wb-panel-out"]):
                            gr.Markdown("#### Answer")
                            asst_ans = gr.Markdown()

                asst_desc_btn.click(
                    tab_assistant_describe,
                    [asst_img],
                    [caption_state, asst_desc],
                )
                asst_ask_btn.click(
                    tab_assistant_ask,
                    [caption_state, asst_q, asst_img],
                    asst_ans,
                )

        with gr.Accordion("Runtime & model routing", open=False):
            wb_runtime = gr.Markdown(runtime_info_md())

        demo.load(runtime_info_md, [], wb_runtime)

        gr.HTML(
            """
<div class="wb-footer">
  <strong>Ship checklist:</strong> export latency tables from each tab in your appendix, benchmark CLIP throughput on ROCm MI300X, publish as a HF Space with <code>WB_ENABLE_VLM=1</code> on an A100/L4/MI large runner, optional <code>WB_AUDIT_JSONL=data/audit.jsonl</code> for traceability demos.
</div>
            """.strip()
        )
    # Queue keeps UI responsive while model calls run (downloads/inference can take time).
    demo.queue(default_concurrency_limit=2)
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=WB_THEME,
        css=CUSTOM_CSS,
        show_error=True,
    )
