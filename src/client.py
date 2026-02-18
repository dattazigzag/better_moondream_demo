"""
Moondream client wrapper.

Thin layer around the `moondream` Python client that connects to
Moondream Station (localhost:2020). Exposes query, caption, detect,
and point with consistent error handling and connection health checks.
"""

import time

import moondream as md
from PIL import Image

from src.config import config
from src.logger import get_logger

log = get_logger("moondream")

DEFAULT_ENDPOINT = config["moondream"]["endpoint"]


class MoondreamClient:
    """
    Wrapper around the moondream Python client.

    Connects to a running Moondream Station instance and provides
    clean methods for each capability. All methods return dicts —
    either the result data or {"error": "description"} on failure.
    """

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT):
        self.endpoint = endpoint
        self.model = md.vl(endpoint=endpoint)

    def is_available(self) -> bool:
        """
        Check whether Moondream Station is reachable.

        Attempts a minimal caption call on a tiny dummy image.
        Returns True if Station responds, False otherwise.
        """
        try:
            # Create a tiny 1x1 image — cheapest possible inference call
            tiny_image = Image.new("RGB", (1, 1), color=(0, 0, 0))
            self.model.caption(tiny_image, length="short")
            return True
        except Exception:
            return False

    def encode_image(self, image: Image.Image):
        """
        Pre-encode an image for reuse across multiple calls.

        Moondream spends a significant portion of inference time encoding
        the image. Encoding once and reusing the result across query,
        caption, detect, and point calls on the same image gives a
        meaningful speedup.

        Returns:
            An encoded image object, or None on failure.
        """
        try:
            log.debug("Encoding image for reuse...")
            start = time.time()
            encoded = self.model.encode_image(image)
            elapsed = (time.time() - start) * 1000
            log.debug(f"Image encoded ({elapsed:.0f}ms)")
            return encoded
        except Exception as e:
            log.warning(f"encode_image failed, will use raw image: {e}")
            return None

    def query(self, image, question: str, reasoning: bool = True) -> dict:
        """
        Ask a question about an image (visual question answering).

        Uses reasoning mode by default — Moondream 3 will "think"
        before answering, which produces better results for complex
        questions. Disable reasoning for simple factual queries, OCR,
        and structured output requests (JSON, markdown, CSV) where
        chain-of-thought adds latency without improving quality.

        Args:
            image: PIL Image or pre-encoded image from encode_image().
            question: The question or prompt to send.
            reasoning: Whether to enable Moondream's reasoning mode.

        Returns:
            {"answer": "the model's response"}
            or {"error": "description"} on failure
        """
        try:
            mode = "reasoning" if reasoning else "direct"
            log.info(f'query("{question}") [{mode}]')
            start = time.time()
            result = self.model.query(image, question, reasoning=reasoning)
            elapsed = (time.time() - start) * 1000
            log.info(f"query result ({elapsed:.0f}ms): {str(result['answer'])[:120]}")  # type: ignore
            return {"answer": result["answer"]}  # type: ignore
        except Exception as e:
            log.error(f"Query failed: {e}")
            return {"error": f"Query failed: {e}"}

    def caption(self, image, length: str = "normal") -> dict:
        """
        Generate a text description of the image.

        Args:
            image: PIL Image to describe.
            length: One of "short", "normal", or "long".

        Returns:
            {"caption": "the generated description"}
            or {"error": "description"} on failure
        """
        if length not in ("short", "normal", "long"):
            return {
                "error": f"Invalid caption length '{length}'. Use short, normal, or long."
            }

        try:
            log.info(f'caption(length="{length}")')
            start = time.time()
            result = self.model.caption(image, length=length)
            elapsed = (time.time() - start) * 1000
            log.info(
                f"caption result ({elapsed:.0f}ms): {str(result['caption'])[:120]}"
            )
            return {"caption": result["caption"]}
        except Exception as e:
            log.error(f"Caption failed: {e}")
            return {"error": f"Caption failed: {e}"}

    def detect(self, image, subject: str) -> dict:
        """
        Detect objects in the image matching the subject.

        Returns bounding boxes with coordinates normalized to 0–1
        (relative to image dimensions).

        Args:
            image: PIL Image to search.
            subject: What to look for (e.g. "car", "person", "cat").

        Returns:
            {"objects": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}, ...]}
            or {"error": "description"} on failure
        """
        try:
            log.info(f'detect("{subject}")')
            start = time.time()
            result = self.model.detect(image, subject)
            elapsed = (time.time() - start) * 1000
            count = len(result["objects"])
            log.info(f"detect result ({elapsed:.0f}ms): found {count} object(s)")
            return {"objects": result["objects"]}
        except Exception as e:
            log.error(f"Detection failed: {e}")
            return {"error": f"Detection failed: {e}"}

    def point(self, image, subject: str) -> dict:
        """
        Locate objects by their center point.

        Returns point coordinates normalized to 0–1 (relative to
        image dimensions). Values go from top-left (0,0) to
        bottom-right (1,1).

        Args:
            image: PIL Image to search.
            subject: What to point at (e.g. "the red button").

        Returns:
            {"points": [{"x": ..., "y": ...}, ...]}
            or {"error": "description"} on failure
        """
        try:
            log.info(f'point("{subject}")')
            start = time.time()
            result = self.model.point(image, subject)
            elapsed = (time.time() - start) * 1000
            count = len(result["points"])
            log.info(f"point result ({elapsed:.0f}ms): found {count} point(s)")
            return {"points": result["points"]}
        except Exception as e:
            log.error(f"Pointing failed: {e}")
            return {"error": f"Pointing failed: {e}"}
