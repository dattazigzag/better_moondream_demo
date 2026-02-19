"""
Moondream client wrapper.

Thin layer around the `moondream` Python client that connects to
Moondream Station (localhost:2020). Exposes query, caption, detect,
point, and segment with consistent error handling and connection
health checks.

Note: The query() method makes HTTP requests directly to the station
instead of using the library's query() — the library assumes the
station always returns {"answer": ...}, but that's not guaranteed
across station versions and configurations. By calling the endpoint
ourselves we can log the raw response and handle any format.
"""

import json
import time
import urllib.request

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

        Note: We make the HTTP request directly instead of calling
        self.model.query() because the library assumes the station
        always returns {"answer": ...}. By calling the endpoint
        ourselves we can log the raw response and handle any format
        the station version returns.

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

            # Encode image using the library's encoder
            encoded = self.model.encode_image(image)
            image_url = encoded.image_url

            # Build payload — only include "reasoning" key when True,
            # matching the library's behaviour
            payload: dict = {
                "image_url": image_url,
                "question": question,
                "stream": False,
            }
            if reasoning:
                payload["reasoning"] = True

            # For OCR / heavy text extraction, request more tokens so
            # the station doesn't cut off mid-response and timeout.
            max_tokens = config["moondream"].get("max_tokens")
            if max_tokens:
                payload["settings"] = {"max_tokens": max_tokens}

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.endpoint}/query",
                data=data,
                headers={"Content-Type": "application/json"},
            )

            # Use configurable timeout — dense OCR on large images can
            # take well over the station's default 30s
            http_timeout = config["moondream"].get("timeout", 90)

            with urllib.request.urlopen(req, timeout=http_timeout) as response:
                raw = response.read().decode("utf-8")

            result = json.loads(raw)
            elapsed = (time.time() - start) * 1000

            # Log the raw keys so we can debug station response formats
            log.debug(f"Station response keys: {list(result.keys())}")

            # ── Check for station-level errors first ──────────────
            if "error" in result:
                err_msg = result["error"]
                log.error(f"Station error ({elapsed:.0f}ms): {err_msg}")
                # Give the user a helpful hint for timeouts
                if "timeout" in str(err_msg).lower():
                    return {
                        "error": (
                            f"Moondream Station timed out ({elapsed / 1000:.0f}s). "
                            "The image may contain too much text for a single pass. "
                            "Try asking about a specific section, or increase "
                            "moondream.timeout / moondream.max_tokens in config.yaml."
                        )
                    }
                return {"error": f"Moondream Station error: {err_msg}"}

            # ── Extract the answer ────────────────────────────────
            # Try common answer keys — station versions may vary
            answer = (
                result.get("answer")
                or result.get("text")
                or result.get("result")
                or result.get("response")
            )

            if answer is None:
                # Last resort: if there's a string value, use it
                string_vals = [
                    v for v in result.values()
                    if isinstance(v, str) and v
                ]
                if string_vals:
                    answer = string_vals[0]
                    log.warning(
                        f"No 'answer' key in station response. "
                        f"Used first string value from keys: {list(result.keys())}"
                    )
                else:
                    log.error(
                        f"Station returned no usable answer. "
                        f"Raw response: {raw[:500]}"
                    )
                    return {
                        "error": (
                            "Moondream returned an unexpected response "
                            f"format: {list(result.keys())}"
                        )
                    }

            log.info(f"query result ({elapsed:.0f}ms): {str(answer)[:120]}")
            return {"answer": answer}

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

        Note: Like query(), we make the HTTP request directly instead
        of calling self.model.detect() because the library blindly
        accesses result["objects"] which crashes if Station returns
        an unexpected format (e.g. on cold-start or timeout).

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

            encoded = self.model.encode_image(image)
            image_url = encoded.image_url

            payload: dict = {
                "image_url": image_url,
                "object": subject,
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.endpoint}/detect",
                data=data,
                headers={"Content-Type": "application/json"},
            )

            http_timeout = config["moondream"].get("timeout", 90)

            with urllib.request.urlopen(req, timeout=http_timeout) as response:
                raw = response.read().decode("utf-8")

            result = json.loads(raw)
            elapsed = (time.time() - start) * 1000

            log.debug(f"Station detect response keys: {list(result.keys())}")

            if "error" in result:
                err_msg = result["error"]
                log.error(f"Station detect error ({elapsed:.0f}ms): {err_msg}")
                if "timeout" in str(err_msg).lower():
                    return {
                        "error": (
                            f"Moondream Station timed out during detection ({elapsed / 1000:.0f}s). "
                            "Try increasing moondream.timeout in config.yaml."
                        )
                    }
                return {"error": f"Detection failed: {err_msg}"}

            objects = result.get("objects", [])
            count = len(objects)
            log.info(f"detect result ({elapsed:.0f}ms): found {count} object(s)")
            return {"objects": objects}

        except Exception as e:
            log.error(f"Detection failed: {e}")
            return {"error": f"Detection failed: {e}"}

    def point(self, image, subject: str) -> dict:
        """
        Locate objects by their center point.

        Returns point coordinates normalized to 0–1 (relative to
        image dimensions). Values go from top-left (0,0) to
        bottom-right (1,1).

        Note: Like query() and detect(), we make the HTTP request
        directly for defensive response handling.

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

            encoded = self.model.encode_image(image)
            image_url = encoded.image_url

            payload: dict = {
                "image_url": image_url,
                "object": subject,
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.endpoint}/point",
                data=data,
                headers={"Content-Type": "application/json"},
            )

            http_timeout = config["moondream"].get("timeout", 90)

            with urllib.request.urlopen(req, timeout=http_timeout) as response:
                raw = response.read().decode("utf-8")

            result = json.loads(raw)
            elapsed = (time.time() - start) * 1000

            log.debug(f"Station point response keys: {list(result.keys())}")

            if "error" in result:
                err_msg = result["error"]
                log.error(f"Station point error ({elapsed:.0f}ms): {err_msg}")
                if "timeout" in str(err_msg).lower():
                    return {
                        "error": (
                            f"Moondream Station timed out during pointing ({elapsed / 1000:.0f}s). "
                            "Try increasing moondream.timeout in config.yaml."
                        )
                    }
                return {"error": f"Pointing failed: {err_msg}"}

            points = result.get("points", [])
            count = len(points)
            log.info(f"point result ({elapsed:.0f}ms): found {count} point(s)")
            return {"points": points}

        except Exception as e:
            log.error(f"Pointing failed: {e}")
            return {"error": f"Pointing failed: {e}"}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FUTURE: Segment — currently cloud-only (not available in
    # Moondream Station). Station returns:
    #   {'error': "Function 'segment' not available"}
    # This method is kept for when Station adds local segment
    # support. Nothing calls it right now.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def segment(self, image, prompt: str) -> dict:
        """
        Segment an object in the image using a text prompt.

        Returns an SVG path mask and bounding box for the described
        object. The SVG path is in pixel coordinates matching the
        original image dimensions. The bounding box uses normalized
        0-1 coordinates like detect/point.

        This is a newer Moondream 3 capability and may not be
        available on all Station versions. If Station doesn't
        support it, the error is returned cleanly.

        Args:
            image: PIL Image or pre-encoded image from encode_image().
            prompt: What to segment (e.g. "the person in blue").

        Returns:
            {"path": "M...", "bbox": {"x_min": ..., ...}}
            or {"error": "description"} on failure
        """
        try:
            log.info(f'segment("{prompt}")')
            start = time.time()

            # Encode image to get the image_url for the HTTP call
            encoded = self.model.encode_image(image)
            image_url = encoded.image_url

            payload: dict = {
                "image_url": image_url,
                "object": prompt,
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.endpoint}/segment",
                data=data,
                headers={"Content-Type": "application/json"},
            )

            http_timeout = config["moondream"].get("timeout", 90)

            with urllib.request.urlopen(req, timeout=http_timeout) as response:
                raw = response.read().decode("utf-8")

            result = json.loads(raw)
            elapsed = (time.time() - start) * 1000

            log.debug(f"Station segment response keys: {list(result.keys())}")

            if "error" in result:
                err_msg = result["error"]
                log.error(f"Station segment error ({elapsed:.0f}ms): {err_msg}")
                return {"error": f"Segment failed: {err_msg}"}

            # Extract path and bounding box from response
            path = result.get("path") or result.get("mask")
            bbox = result.get("bbox") or result.get("bounding_box")

            if not path:
                log.error(f"No path in segment response: {list(result.keys())}")
                return {
                    "error": (
                        "Segment returned no mask path. "
                        "This capability may not be available on your "
                        "Moondream Station version."
                    )
                }

            log.info(f"segment result ({elapsed:.0f}ms): path length={len(path)}")
            return {"path": path, "bbox": bbox}

        except urllib.error.HTTPError as e:
            log.error(f"Segment HTTP error: {e.code} {e.reason}")
            if e.code == 404:
                return {
                    "error": (
                        "Segment endpoint not found (HTTP 404). "
                        "This capability may not be available on your "
                        "Moondream Station version."
                    )
                }
            return {"error": f"Segment failed: HTTP {e.code} {e.reason}"}
        except Exception as e:
            log.error(f"Segment failed: {e}")
            return {"error": f"Segment failed: {e}"}
