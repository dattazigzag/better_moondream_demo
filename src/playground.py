"""
Moondream Playground — direct access to every local vision capability.

Unlike the Chat interface (which uses an LLM to route requests),
the Playground lets you pick exactly which Moondream capability to
run. Upload an image, choose a task, fill in the inputs, and hit Run.

No LLM orchestrator, no interpretation — you control what happens.

Run with:
    python main.py --mode playground
"""

import json

import gradio as gr
from PIL import Image

from src.client import MoondreamClient
from src.logger import get_logger
from src.renderer import draw_bounding_boxes, draw_points

log = get_logger("playground")


# Same CSS as the Chat app for visual consistency
CUSTOM_CSS = """
/* Make output images display larger */
#output-image img {
    max-height: 70vh !important;
    max-width: 100% !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    border-radius: 8px;
}

/* Uploaded image preview — reasonable size */
#input-image img {
    max-height: 50vh !important;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 16px 0 8px;
    font-size: 0.85em;
    opacity: 0.6;
}
.app-footer a { text-decoration: none; }
.app-footer a:hover { text-decoration: underline; }
"""

# ── Task definitions ─────────────────────────────────────────────
# Each task maps to a Moondream capability. These descriptions show
# in the UI to help users understand what each task does.

TASK_CHOICES = ["Caption", "Query", "Detect", "Point", "OCR"]

TASK_DESCRIPTIONS = {
    "Caption": (
        "Generate a text description of the image. "
        "Choose short for a one-liner, normal for a sentence or two, "
        "or long for a detailed paragraph."
    ),
    "Query": (
        "Ask any question about the image — what's in it, what's "
        "happening, spatial relationships, counting, or extracting "
        "structured data (JSON, markdown, CSV). "
        "Enable reasoning for complex questions; disable it for "
        "simple facts to get faster responses."
    ),
    "Detect": (
        "Find all instances of something in the image and draw "
        "bounding boxes around them. Enter what you're looking for "
        "(e.g. \"cars\", \"people\", \"red buttons\"). Each result "
        "gets a colored box with coordinates."
    ),
    "Point": (
        "Locate objects by their center point. Similar to detect, "
        "but returns precise coordinates instead of bounding boxes. "
        "Useful when you need exact positions rather than regions."
    ),
    "OCR": (
        "Extract text from the image. Moondream reads signs, "
        "documents, labels, screens — anything with visible text. "
        "You can also ask it to format the output as markdown "
        "(great for tables and documents) or other formats."
    ),
}


def _raw_json_block(data: dict) -> str:
    """Format raw result data as pretty-printed JSON.

    If any value is itself a JSON string, parse it first so
    the output is fully expanded rather than a single escaped line.
    """
    cleaned = {}
    for k, v in data.items():
        if isinstance(v, str):
            try:
                cleaned[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                cleaned[k] = v
        else:
            cleaned[k] = v
    return json.dumps(cleaned, indent=2)


def create_app() -> gr.Blocks:
    """
    Build and return the Playground Gradio app.

    Uses gr.Blocks for full layout control — conditional visibility
    of input fields based on which task is selected.
    """
    client = MoondreamClient()

    with gr.Blocks() as app:
        gr.Markdown(
            "# Moondream Playground\n"
            "Direct access to every Moondream 3 vision capability running "
            "locally on your machine. Upload an image, pick a task, and run it. "
            "No LLM in the loop — you choose exactly what happens."
        )

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Image",
                    type="pil",
                    elem_id="input-image",
                    height=400,
                )

                task_radio = gr.Radio(
                    choices=TASK_CHOICES,
                    value="Caption",
                    label="Task",
                    info="What do you want Moondream to do with this image?",
                )

                task_description = gr.Markdown(
                    value=TASK_DESCRIPTIONS["Caption"],
                    elem_id="task-description",
                )

                # ── Caption inputs ───────────────────────────────
                with gr.Group(visible=True) as caption_inputs:
                    caption_length = gr.Radio(
                        choices=["short", "normal", "long"],
                        value="normal",
                        label="Description length",
                        info="Short = one line, normal = a few sentences, long = detailed paragraph",
                    )

                # ── Query inputs ─────────────────────────────────
                with gr.Group(visible=False) as query_inputs:
                    query_question = gr.Textbox(
                        label="Question",
                        placeholder="e.g. What breed is this dog? / How many people? / Extract a JSON with keys: name, color",
                        lines=2,
                        info="Ask anything — visual questions or structured data extraction",
                    )
                    query_reasoning = gr.Checkbox(
                        label="Enable reasoning",
                        value=True,
                        info="On = model thinks step-by-step (better for complex questions). Off = faster direct answer (better for simple facts, structured output).",
                    )

                # ── Detect inputs ────────────────────────────────
                with gr.Group(visible=False) as detect_inputs:
                    detect_subject = gr.Textbox(
                        label="What to find",
                        placeholder="e.g. cars, people, red buttons, street signs",
                        info="Enter one type of object. Moondream will find all instances and draw boxes around them.",
                    )

                # ── Point inputs ─────────────────────────────────
                with gr.Group(visible=False) as point_inputs:
                    point_subject = gr.Textbox(
                        label="What to locate",
                        placeholder="e.g. the dog, largest item, red button",
                        info="Enter what to point at. Moondream marks the center of each match.",
                    )

                # ── OCR inputs ───────────────────────────────────
                with gr.Group(visible=False) as ocr_inputs:
                    ocr_format = gr.Radio(
                        choices=["Plain text", "Markdown"],
                        value="Plain text",
                        label="Output format",
                        info="Plain text extracts raw text. Markdown preserves tables, headings, and structure.",
                    )

                run_btn = gr.Button(
                    "Run",
                    variant="primary",
                    size="lg",
                )

            # ── Right column: outputs ────────────────────────────
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Result",
                    lines=8,
                    interactive=False,
                )

                output_image = gr.Image(
                    label="Annotated image",
                    elem_id="output-image",
                    height=400,
                    visible=False,
                )

                with gr.Accordion("Raw JSON", open=False, visible=False) as raw_accordion:
                    output_raw = gr.Code(
                        label="Raw response data",
                        language="json",
                        interactive=False,
                    )

        # ── Task selection: toggle input visibility ──────────
        def on_task_change(task: str):
            """Show/hide input groups based on the selected task."""
            return (
                gr.update(visible=(task == "Caption")),
                gr.update(visible=(task == "Query")),
                gr.update(visible=(task == "Detect")),
                gr.update(visible=(task == "Point")),
                gr.update(visible=(task == "OCR")),
                TASK_DESCRIPTIONS.get(task, ""),
            )

        task_radio.change(
            fn=on_task_change,
            inputs=[task_radio],
            outputs=[
                caption_inputs,
                query_inputs,
                detect_inputs,
                point_inputs,
                ocr_inputs,
                task_description,
            ],
        )

        # ── Run button handler ───────────────────────────────
        def run_task(
            image: Image.Image | None,
            task: str,
            cap_length: str,
            q_question: str,
            q_reasoning: bool,
            det_subject: str,
            pt_subject: str,
            ocr_fmt: str,
        ):
            """
            Execute the selected Moondream task and return results.

            Returns a tuple of:
                (text_output, image_output, raw_json, accordion_visible)
            """
            if image is None:
                return (
                    "Please upload an image first.",
                    gr.update(visible=False, value=None),
                    "",
                    gr.update(visible=False),
                )

            image = image.convert("RGB")
            log.info(f"Running task: {task} on {image.size[0]}x{image.size[1]} image")

            # Encode image once for reuse
            encoded = client.encode_image(image)
            working_image = encoded if encoded is not None else image

            # ── Caption ──────────────────────────────────────
            if task == "Caption":
                result = client.caption(working_image, length=cap_length)
                if "error" in result:
                    return (
                        f"Error: {result['error']}",
                        gr.update(visible=False, value=None),
                        "",
                        gr.update(visible=False),
                    )
                return (
                    result["caption"],
                    gr.update(visible=False, value=None),
                    _raw_json_block(result),
                    gr.update(visible=True),
                )

            # ── Query ────────────────────────────────────────
            elif task == "Query":
                question = q_question.strip()
                if not question:
                    return (
                        "Please enter a question.",
                        gr.update(visible=False, value=None),
                        "",
                        gr.update(visible=False),
                    )
                result = client.query(working_image, question, reasoning=q_reasoning)
                if "error" in result:
                    return (
                        f"Error: {result['error']}",
                        gr.update(visible=False, value=None),
                        "",
                        gr.update(visible=False),
                    )
                return (
                    result["answer"],
                    gr.update(visible=False, value=None),
                    _raw_json_block(result),
                    gr.update(visible=True),
                )

            # ── Detect ───────────────────────────────────────
            elif task == "Detect":
                subject = det_subject.strip()
                if not subject:
                    return (
                        "Please enter what to detect.",
                        gr.update(visible=False, value=None),
                        "",
                        gr.update(visible=False),
                    )
                result = client.detect(working_image, subject)
                if "error" in result:
                    return (
                        f"Error: {result['error']}",
                        gr.update(visible=False, value=None),
                        "",
                        gr.update(visible=False),
                    )
                objects = result["objects"]
                if not objects:
                    return (
                        f'No "{subject}" found in the image.',
                        gr.update(visible=False, value=None),
                        _raw_json_block(result),
                        gr.update(visible=True),
                    )
                annotated = draw_bounding_boxes(image, objects, subject)
                count = len(objects)
                raw_data = {"objects": objects, "count": count}
                return (
                    f"Found {count} result{'s' if count != 1 else ''} for \"{subject}\"",
                    gr.update(visible=True, value=annotated),
                    _raw_json_block(raw_data),
                    gr.update(visible=True),
                )

            # ── Point ────────────────────────────────────────
            elif task == "Point":
                subject = pt_subject.strip()
                if not subject:
                    return (
                        "Please enter what to locate.",
                        gr.update(visible=False, value=None),
                        "",
                        gr.update(visible=False),
                    )
                result = client.point(working_image, subject)
                if "error" in result:
                    return (
                        f"Error: {result['error']}",
                        gr.update(visible=False, value=None),
                        "",
                        gr.update(visible=False),
                    )
                points = result["points"]
                if not points:
                    return (
                        f'Could not locate "{subject}" in the image.',
                        gr.update(visible=False, value=None),
                        _raw_json_block(result),
                        gr.update(visible=True),
                    )
                annotated = draw_points(image, points, subject)
                count = len(points)
                raw_data = {"points": points, "count": count}
                return (
                    f"Located {count} result{'s' if count != 1 else ''} for \"{subject}\"",
                    gr.update(visible=True, value=annotated),
                    _raw_json_block(raw_data),
                    gr.update(visible=True),
                )

            # ── OCR ──────────────────────────────────────────
            elif task == "OCR":
                if ocr_fmt == "Markdown":
                    question = "Convert all visible text in this image to well-formatted markdown."
                else:
                    question = "Extract all visible text in this image."
                result = client.query(working_image, question, reasoning=False)
                if "error" in result:
                    return (
                        f"Error: {result['error']}",
                        gr.update(visible=False, value=None),
                        "",
                        gr.update(visible=False),
                    )
                return (
                    result["answer"],
                    gr.update(visible=False, value=None),
                    _raw_json_block(result),
                    gr.update(visible=True),
                )

            # ── Shouldn't reach here ─────────────────────────
            return (
                "Unknown task selected.",
                gr.update(visible=False, value=None),
                "",
                gr.update(visible=False),
            )

        run_btn.click(
            fn=run_task,
            inputs=[
                input_image,
                task_radio,
                caption_length,
                query_question,
                query_reasoning,
                detect_subject,
                point_subject,
                ocr_format,
            ],
            outputs=[
                output_text,
                output_image,
                output_raw,
                raw_accordion,
            ],
        )

        # ── Footer ────────────────────────────────────────
        gr.HTML(
            '<div class="app-footer">'
            '<a href="https://zigzag.is/en" target="_blank">zigzag.is</a>'
            " &middot; "
            '<a href="mailto:datta@zigzag.is">Saurabh Datta</a>'
            "</div>"
        )

    return app


def main(host: str | None = None, port: int | None = None):
    """
    Launch the Playground interface.

    Args:
        host: Bind address (defaults to config.yaml → app.host).
        port: Port number (defaults to config.yaml → app.port).
    """
    from src.config import config

    app_cfg = config["app"]
    app = create_app()
    app.launch(
        server_name=host or app_cfg["host"],
        server_port=port or app_cfg["port"],
        share=app_cfg["share"],
        theme=app_cfg["theme"],
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
