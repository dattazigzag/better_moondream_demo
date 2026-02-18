# Moondream Chat

A chat interface for talking to your images. Upload a photo, ask questions about it, find objects, get descriptions — all powered by [Moondream 3](https://moondream.ai/blog/moondream-3-preview), a vision-language model that runs locally on your machine.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## What It Does

You upload an image and have a conversation about it. The app figures out what you're asking for and picks the best tool for the job:

**Ask a question** — "What breed is this dog?" → You get a text answer. The model reasons about your question before responding, which helps with complex or nuanced queries.

**Get a description** — "Describe this image" → You get a caption. You can ask for short, normal, or long descriptions depending on how much detail you need.

**Find objects** — "Find all cars" → You get an annotated image with bounding boxes drawn around each car, plus a count. Useful for inventory, counting, or just seeing what the model identifies.

**Locate something** — "Where's the cat?" → You get the image with crosshair markers pointing to the location. Helpful when you want precise positioning rather than a bounding region.

The routing between these modes happens automatically based on how you phrase your message. You don't need to pick a mode manually — just talk naturally. If the auto-detection gets it wrong, the fallback is always visual question answering, which handles pretty much anything.

## How It Works Under the Hood

The project has four modules that each handle one concern:

```
src/
├── app.py        ← Gradio chat UI, ties everything together
├── client.py     ← Talks to Moondream Station over HTTP
├── intent.py     ← Figures out what you're asking for
└── renderer.py   ← Draws boxes and markers on images
```

**client.py** wraps the official [moondream Python client](https://pypi.org/project/moondream/). It connects to a running Moondream Station instance on `localhost:2020` and exposes clean methods for each capability — query, caption, detect, point. Every method returns a dict, either with the result data or an `{"error": "..."}` key, so the UI always has something sensible to display.

**intent.py** is a keyword-based message parser. It scans your message for trigger phrases ("find all...", "where's the...", "describe this...") and routes to the appropriate capability. No ML involved here — just regex patterns checked in a deliberate order. Caption triggers are checked first, then detect (plural/spatial phrasing), then point (singular location phrasing), and anything that doesn't match falls through to query. Since query is Moondream's most flexible mode, misclassified messages still produce useful answers.

**renderer.py** takes the raw normalized coordinates from detect/point results (values between 0 and 1) and draws them onto a copy of the original image using Pillow. Bounding boxes get semi-transparent fills with solid borders and labeled tags. Point markers get crosshair-and-ring indicators with text outlines for contrast. Everything scales proportionally with image dimensions, so annotations look right whether your image is 200px or 4000px wide.

**app.py** is the Gradio chat interface that wires it all together. It handles image persistence across messages (upload once, ask many questions), routes through intent detection, calls the appropriate client method, and returns either plain text or annotated images depending on what makes sense.

## Prerequisites

Before running this project, you need **Moondream Station** installed and running on your machine.

### What is Moondream Station?

Moondream Station is a local inference server for the Moondream vision-language model. It downloads and manages the model weights, runs inference using your hardware, and exposes an HTTP API on `localhost:2020`. Think of it as the engine — this project is just the dashboard.

On Apple Silicon Macs, Station uses MLX for native acceleration. You'll need at least 16GB of memory. On an M1 Max with 64GB, expect 35+ tokens per second. Detection and pointing feel near-instantaneous.

### Installing Moondream Station

```bash
pip install moondream-station
```

Then start it:

```bash
moondream-station
```

The first run downloads the model weights (a few GB). After that, it starts in seconds. Leave it running in a terminal tab while you use this project.

For more details, see the [Moondream Station docs](https://docs.moondream.ai/station/).

## Setup

Clone the repo and install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/your-username/better_moondream_demo.git
cd better_moondream_demo
uv sync
```

If you don't have uv, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Running

Make sure Moondream Station is running in another terminal:

```bash
moondream-station
```

Then launch the chat interface:

```bash
uv run python main.py
```

Or if you've installed the project:

```bash
uv run moondream-chat
```

The app opens in your browser at `http://localhost:7860`. Upload an image and start chatting.

## Example Conversations

Here are some things you can try once an image is uploaded:

| You type | What happens |
|---|---|
| *(just upload, no text)* | Auto-generates a description of the image |
| "What's happening in this photo?" | Visual question answering with reasoning |
| "Describe this image in detail" | Long-form caption |
| "Give me a short caption" | Brief one-liner description |
| "Find all people" | Bounding boxes drawn around each person |
| "How many cars are there?" | Detection + count |
| "Where's the dog?" | Crosshair marker on the dog's location |
| "Point to the red button" | Precise location marker |
| "What text is visible?" | OCR via the query capability |
| "Is this photo taken indoors or outdoors?" | Reasoning-based answer |

You can keep asking follow-up questions about the same image without re-uploading. Upload a new image at any time to switch context.

## Project Structure

```
better_moondream_demo/
├── main.py              Entry point (python main.py)
├── pyproject.toml       Project config and dependencies
├── README.md
├── .gitignore
└── src/
    ├── __init__.py
    ├── app.py           Gradio chat interface
    ├── client.py        Moondream Station client wrapper
    ├── intent.py        Message-to-capability routing
    └── renderer.py      Bounding box and point annotation
```

## Dependencies

| Package | Purpose |
|---|---|
| [gradio](https://gradio.app) | Chat interface with multimodal input |
| [moondream](https://pypi.org/project/moondream/) | Python client for the Moondream API |
| [moondream-station](https://pypi.org/project/moondream-station/) | Local inference server |
| [Pillow](https://python-pillow.org/) | Image loading and annotation drawing |

## About Moondream 3

Moondream 3 is a 9-billion-parameter Mixture-of-Experts vision-language model that activates roughly 2 billion parameters per inference. It routes tokens dynamically across 64 experts and activates only 8 for each input, which keeps it fast despite the large total parameter count.

Key capabilities that this project uses:

- **Visual Question Answering** with optional reasoning mode — the model can "think" before answering, producing better results on complex questions
- **Captioning** at three detail levels (short, normal, long)
- **Object Detection** returning normalized bounding box coordinates
- **Pointing** returning normalized center-point coordinates
- **OCR** for reading text within images
- **32K context window** for handling complex, multi-turn conversations about images

The model runs entirely on your machine through Moondream Station. No data leaves your computer.

For more on the model architecture: [Moondream 3 Preview announcement](https://moondream.ai/blog/moondream-3-preview)

For the model weights: [moondream/moondream3-preview on HuggingFace](https://huggingface.co/moondream/moondream3-preview)

## License

MIT
