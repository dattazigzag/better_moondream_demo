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


def _execute_action(
    action: Action, image: Image.Image, client: MoondreamClient
) -> list[gr.ChatMessage]:
    """
    Execute a single orchestrator action against Moondream and return
    one or more ChatMessage objects for the response.
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

        annotated = draw_bounding_boxes(image, objects, subject)
        count = len(objects)
        summary = f"Found {count} {subject}"
        return [
            gr.ChatMessage(role="assistant", content=summary),
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

        annotated = draw_points(image, points, subject)
        count = len(points)
        summary = f"Located {count} {subject}" if count > 1 else f"Located {subject}"
        return [
            gr.ChatMessage(role="assistant", content=summary),
            gr.ChatMessage(
                role="assistant",
                content=gr.Image(value=annotated),
            ),
        ]

    else:
        # Default: query
        question = action.question or "What is in this image?"
        result = client.query(image, question)
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
    ) -> list[gr.ChatMessage]:
        """
        Process a user message and return the assistant's response.

        Flow:
        1. Extract image from the message (or find in history)
        2. Send text to the LLM orchestrator for intent parsing
        3. Execute each action against Moondream
        4. Return formatted results (text and/or annotated images)
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
                return [
                    gr.ChatMessage(
                        role="assistant",
                        content=f"Could not open that image: {e}",
                    )
                ]

        if current_image is None:
            current_image = _find_image_in_history(history)
            if current_image:
                log.debug("Using image from conversation history")

        if current_image is None:
            log.warning("No image available")
            return [
                gr.ChatMessage(
                    role="assistant",
                    content=(
                        "I need an image to work with. Please upload one "
                        "using the attachment button in the message box."
                    ),
                )
            ]

        # No text but an image was uploaded — default to caption
        if not text:
            text = "Describe this image"

        # --- Orchestrate: parse intent via LLM (or regex fallback) ---
        orch_result = orchestrate(text, history)

        if orch_result.error:
            log.error(f"Orchestrator error: {orch_result.error}")
            return [
                gr.ChatMessage(role="assistant", content=f"Sorry, something went wrong: {orch_result.error}")
            ]

        # --- Execute each action ---
        all_responses: list[gr.ChatMessage] = []

        for i, action in enumerate(orch_result.actions):
            if len(orch_result.actions) > 1:
                log.info(f"Executing step {i + 1}/{len(orch_result.actions)}: {action.action}")

            responses = _execute_action(action, current_image, client)
            all_responses.extend(responses)

        log.info(f"Returning {len(all_responses)} message(s) to chat")
        return all_responses

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
            "Upload an image and chat with it. Ask questions, request "
            "captions, detect objects, or locate specific things. "
            "Powered by Moondream 3 via Moondream Station."
        ),
        textbox=gr.MultimodalTextbox(
            file_types=["image"],
            file_count="single",
            placeholder="Ask about your image, or upload a new one...",
            sources=["upload"],
        ),
        examples=[
            {"text": "What's in this image?"},
            {"text": "Describe this image in detail"},
            {"text": "Find all people"},
            {"text": "Where's the main subject?"},
            {"text": "How many objects are there?"},
            {"text": "Point to the largest item"},
        ],
    )

    return app


def main():
    """Launch the Gradio chat interface."""
    app = create_app()
    app.launch(theme="hmb/amethyst", css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
