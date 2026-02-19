"""
Gradio chat interface for Moondream 3.

Main entry point. Provides a chatbot UI where users upload images
and ask questions. Routes messages through the LLM orchestrator (or
regex fallback), calls the appropriate Moondream capability, and
renders results (text answers, annotated images with bounding boxes
or points).

Run with:
    python main.py
    or
    moondream-chat  (if installed via uv/pip)
"""

import json

import gradio as gr
from PIL import Image

from src.client import MoondreamClient
from src.logger import get_logger
from src.orchestrator import Action, orchestrate
from src.renderer import draw_bounding_boxes, draw_points

log = get_logger("app")


# Custom CSS for better image display and layout
CUSTOM_CSS = """
/* Make images in chat messages display larger */
.chatbot .message img {
    max-height: 70vh !important;
    max-width: 100% !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    border-radius: 8px;
}

/* Give the chatbot more breathing room */
.chatbot .messages {
    padding: 12px 16px;
}

/* Uploaded images in user messages — show at reasonable size */
.chatbot .user img {
    max-height: 40vh !important;
}
"""


def _raw_data_block(data: dict) -> str:
    """Format raw result data as a collapsible markdown details block."""
    formatted = json.dumps(data, indent=2)
    return (
        "\n<details><summary>Raw data</summary>\n\n"
        f"```json\n{formatted}\n```\n\n"
        "</details>"
    )


def _execute_action(
    action: Action, image, raw_image: Image.Image, client: MoondreamClient
) -> list[gr.ChatMessage]:
    """
    Execute a single orchestrator action against Moondream and return
    one or more ChatMessage objects for the response.

    Args:
        action: The parsed action from the orchestrator.
        image: Pre-encoded image (or raw PIL Image as fallback).
        raw_image: Original PIL Image for rendering annotations.
        client: The MoondreamClient instance.
    """
    if action.action == "caption":
        length = action.length or "normal"
        result = client.caption(image, length=length)
        if "error" in result:
            return [gr.ChatMessage(role="assistant", content=result["error"])]
        return [gr.ChatMessage(role="assistant", content=result["caption"])]

    elif action.action == "detect":
        subject = action.subject or "object"
        result = client.detect(image, subject)
        if "error" in result:
            return [gr.ChatMessage(role="assistant", content=result["error"])]

        objects = result["objects"]
        if not objects:
            return [
                gr.ChatMessage(
                    role="assistant",
                    content=f'No "{subject}" found in the image.',
                )
            ]

        annotated = draw_bounding_boxes(raw_image, objects, subject)
        count = len(objects)
        summary = f"Found {count} result{'s' if count != 1 else ''} for '{subject}'"
        raw_block = _raw_data_block({"objects": objects, "count": count})
        return [
            gr.ChatMessage(role="assistant", content=summary + raw_block),
            gr.ChatMessage(
                role="assistant",
                content=gr.Image(value=annotated),
            ),
        ]

    elif action.action == "point":
        subject = action.subject or "object"
        result = client.point(image, subject)
        if "error" in result:
            return [gr.ChatMessage(role="assistant", content=result["error"])]

        points = result["points"]
        if not points:
            return [
                gr.ChatMessage(
                    role="assistant",
                    content=f'Could not locate "{subject}" in the image.',
                )
            ]

        annotated = draw_points(raw_image, points, subject)
        count = len(points)
        summary = f"Located {count} result{'s' if count != 1 else ''} for '{subject}'"
        raw_block = _raw_data_block({"points": points, "count": count})
        return [
            gr.ChatMessage(role="assistant", content=summary + raw_block),
            gr.ChatMessage(
                role="assistant",
                content=gr.Image(value=annotated),
            ),
        ]

    else:
        # Default: query (includes OCR, structured output, general VQA)
        question = action.question or "What is in this image?"
        reasoning = action.reasoning
        result = client.query(image, question, reasoning=reasoning)
        if "error" in result:
            return [gr.ChatMessage(role="assistant", content=result["error"])]
        return [gr.ChatMessage(role="assistant", content=result["answer"])]


def _find_image_in_history(history: list[dict]) -> Image.Image | None:
    """
    Search chat history backwards for the most recent user-uploaded image.

    In Gradio 6, content is a list of content blocks:
        [{"type": "text", "text": "..."}, {"type": "file", "file": {"path": "..."}}]
    """
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            # Gradio 6 file block format
            if block.get("type") == "file":
                file_info = block.get("file", {})
                path = file_info.get("path") if isinstance(file_info, dict) else None
                if path:
                    try:
                        return Image.open(path).convert("RGB")
                    except Exception:
                        continue
            # Fallback: direct path key
            if "path" in block:
                try:
                    return Image.open(block["path"]).convert("RGB")
                except Exception:
                    continue
    return None


def create_app() -> gr.Blocks:
    """
    Build and return the Gradio app (without launching it).

    Keeps app creation separate from launching so it's easier to
    test or embed in other applications.
    """
    client = MoondreamClient()

    def handle_message(
        message: dict,
        history: list[dict],
    ):
        """
        Process a user message and yield the assistant's response.

        This is a generator so Gradio can show a stop button and
        cancel between pipeline stages. Each yield replaces the
        previous assistant response in the chat — Gradio keeps the
        last yielded value when the generator finishes or is stopped.

        Flow:
        1. Extract image from the message (or find in history)
        2. Yield a "thinking" status while the LLM parses intent
        3. For each action, yield a "running" status then the result
        4. Final yield contains all accumulated responses
        """
        text = message.get("text", "").strip()
        files = message.get("files", [])

        log.info(f"New message: \"{text}\" ({len(files)} file(s))")

        # --- Resolve the current image ---
        current_image = None

        if files:
            try:
                current_image = Image.open(files[0]).convert("RGB")
                log.info(f"Loaded uploaded image: {current_image.size[0]}x{current_image.size[1]}")
            except Exception as e:
                log.error(f"Failed to open uploaded image: {e}")
                yield [
                    gr.ChatMessage(
                        role="assistant",
                        content=f"Could not open that image: {e}",
                    )
                ]
                return

        if current_image is None:
            current_image = _find_image_in_history(history)
            if current_image:
                log.debug("Using image from conversation history")

        if current_image is None:
            log.warning("No image available")
            yield [
                gr.ChatMessage(
                    role="assistant",
                    content=(
                        "I need an image to work with. Please upload one "
                        "using the attachment button in the message box."
                    ),
                )
            ]
            return

        # No text but an image was uploaded — default to caption
        if not text:
            text = "Describe this image"

        # --- Encode image for reuse across multiple Moondream calls ---
        encoded_image = client.encode_image(current_image)
        # Use encoded version if available, fall back to raw PIL image
        working_image = encoded_image if encoded_image is not None else current_image

        # --- Orchestrate: parse intent via LLM (or regex fallback) ---
        orch_result = orchestrate(text, history)

        if orch_result.error:
            log.error(f"Orchestrator error: {orch_result.error}")
            yield [
                gr.ChatMessage(role="assistant", content=f"Sorry, something went wrong: {orch_result.error}")
            ]
            return

        # --- Execute each action ---
        all_responses: list[gr.ChatMessage] = []
        total_steps = len(orch_result.actions)

        for i, action in enumerate(orch_result.actions):
            step_label = f"Step {i + 1}/{total_steps}: " if total_steps > 1 else ""
            log.info(f"Executing {step_label}{action.action}")

            # Yield results collected so far before each Moondream call.
            # Gradio shows its native processing indicator while we block.
            # If the user presses stop, the generator stops here —
            # they keep the results collected so far.
            if all_responses:
                yield all_responses

            responses = _execute_action(action, working_image, current_image, client)
            all_responses.extend(responses)

        # Final yield — all results, no status indicator
        log.info(f"Returning {len(all_responses)} message(s) to chat")
        yield all_responses

    # --- Build the interface ---
    chatbot = gr.Chatbot(
        height="75vh",
        placeholder="Upload an image and start chatting...",
    )

    app = gr.ChatInterface(
        fn=handle_message,
        multimodal=True,
        chatbot=chatbot,
        title="Moondream Chat",
        description=(
            "Upload an image and chat with it. Ask questions, detect objects, "
            "extract text (OCR), get structured data (JSON/markdown), or "
            "locate specific things. Powered by Moondream 3 + Qwen 3 4B."
        ),
        textbox=gr.MultimodalTextbox(
            file_types=["image"],
            file_count="single",
            placeholder="Ask about your image, or upload a new one...",
            sources=["upload"],
            stop_btn="■",
        ),
        examples=[
            # Vision QA
            {"text": "What's happening in this image?"},
            {"text": "Describe this image in detail"},
            # Detection & pointing
            {"text": "Find all people"},
            {"text": "Point to the largest item"},
            {"text": "How many chairs are there?"},
            # OCR & text extraction
            {"text": "Read all the text in this image"},
            {"text": "Convert the text to markdown"},
            # Structured output
            {"text": "Extract a JSON array with keys: object, color, position"},
            # Multi-step
            {"text": "Find all the objects and describe the most interesting one"},
        ],
    )

    return app


def main(host: str | None = None, port: int | None = None):
    """
    Launch the Gradio chat interface.

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
        theme=app_cfg["theme"],
        share=app_cfg["share"],
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
