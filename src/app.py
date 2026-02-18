"""
Gradio chat interface for Moondream 3.

Main entry point. Provides a chatbot UI where users upload images
and ask questions. Routes messages through intent detection, calls
the appropriate Moondream capability, and renders results (text
answers, annotated images with bounding boxes or points).

Run with:
    python main.py
    or
    moondream-chat  (if installed via uv/pip)
"""

import gradio as gr
from PIL import Image

from src.client import MoondreamClient
from src.intent import Capability, parse_intent
from src.renderer import draw_bounding_boxes, draw_points


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
    ) -> gr.ChatMessage | list[gr.ChatMessage]:
        """
        Process a user message and return the assistant's response.

        This is the core function that ties everything together:
        1. Extract image from the message (if uploaded)
        2. Parse the user's text to detect intent
        3. Call the appropriate Moondream capability
        4. Format the result as text and/or annotated image

        Args:
            message: Dict with "text" and "files" keys from the
                    multimodal textbox.
            history: List of previous messages in OpenAI format.

        Returns:
            One or more ChatMessage objects for the assistant response.
        """
        text = message.get("text", "").strip()
        files = message.get("files", [])

        # --- Resolve the current image ---
        # If the user uploaded a new image with this message, use it.
        # Otherwise, look back through history for the most recent image.
        current_image = None

        if files:
            # Gradio gives us file paths for uploaded files
            try:
                current_image = Image.open(files[0]).convert("RGB")
            except Exception as e:
                return gr.ChatMessage(
                    role="assistant",
                    content=f"Could not open that image: {e}",
                )

        if current_image is None:
            # Search history backwards for the last user-uploaded image
            for msg in reversed(history):
                if msg.get("role") == "user":
                    content = msg.get("content")
                    # Image messages have a dict content with a "path" key
                    if isinstance(content, dict) and "path" in content:
                        try:
                            current_image = Image.open(content["path"]).convert("RGB")
                            break
                        except Exception:
                            continue
                    # Also check for file paths stored as tuples (older format)
                    if isinstance(content, (list, tuple)):
                        for item in content:
                            if isinstance(item, dict) and "path" in item:
                                try:
                                    current_image = Image.open(item["path"]).convert(
                                        "RGB"
                                    )
                                    break
                                except Exception:
                                    continue

        # No image anywhere — can't do vision tasks
        if current_image is None:
            return gr.ChatMessage(
                role="assistant",
                content=(
                    "I need an image to work with. Please upload one "
                    "using the attachment button (📎) in the message box."
                ),
            )

        # No text but an image was uploaded — give a helpful default
        if not text:
            text = "Describe this image"

        # --- Parse intent and call Moondream ---
        intent = parse_intent(text)

        if intent.capability == Capability.CAPTION:
            result = client.caption(current_image, length=intent.caption_length)
            if "error" in result:
                return gr.ChatMessage(role="assistant", content=result["error"])
            return gr.ChatMessage(role="assistant", content=result["caption"])

        elif intent.capability == Capability.DETECT:
            result = client.detect(current_image, intent.subject) # type: ignore
            if "error" in result:
                return gr.ChatMessage(role="assistant", content=result["error"])

            objects = result["objects"]
            if not objects:
                return gr.ChatMessage(
                    role="assistant",
                    content=f'No "{intent.subject}" found in the image.',
                )

            # Draw bounding boxes and return both text + annotated image
            annotated = draw_bounding_boxes(current_image, objects, intent.subject) # type: ignore
            count = len(objects)
            noun = intent.subject
            summary = f"Found {count} {noun}" if count > 1 else f"Found 1 {noun}"

            return [
                gr.ChatMessage(role="assistant", content=summary),
                gr.ChatMessage(
                    role="assistant",
                    content=gr.Image(value=annotated),
                ),
            ]

        elif intent.capability == Capability.POINT:
            result = client.point(current_image, intent.subject) # type: ignore
            if "error" in result:
                return gr.ChatMessage(role="assistant", content=result["error"])

            points = result["points"]
            if not points:
                return gr.ChatMessage(
                    role="assistant",
                    content=f'Could not locate "{intent.subject}" in the image.',
                )

            annotated = draw_points(current_image, points, intent.subject) # type: ignore
            count = len(points)
            noun = intent.subject
            summary = f"Located {count} {noun}" if count > 1 else f"Located {noun}"

            return [
                gr.ChatMessage(role="assistant", content=summary),
                gr.ChatMessage(
                    role="assistant",
                    content=gr.Image(value=annotated),
                ),
            ]

        else:
            # Default: query (visual question answering)
            result = client.query(current_image, intent.question or text)
            if "error" in result:
                return gr.ChatMessage(role="assistant", content=result["error"])
            return gr.ChatMessage(role="assistant", content=result["answer"])

    # --- Build the interface ---
    app = gr.ChatInterface(
        fn=handle_message,
        type="messages", # type: ignore
        multimodal=True,
        theme="hmb/amethyst", # type: ignore
        title="Moondream Chat",
        description=(
            "Upload an image and chat with it. Ask questions, request "
            "captions, detect objects, or locate specific things.\n\n"
            "**Powered by Moondream 3 via Moondream Station.**"
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
    app.launch()


if __name__ == "__main__":
    main()
