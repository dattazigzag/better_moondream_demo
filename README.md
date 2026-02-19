# Moondream Chat & Playground

## PLayground

| Captioning | VQA (Query) | BBOx Detection | Pointing | OCR Extraction |
| --- | --- | --- | --- | --- |
| ![alt text](<assets/Screenshot 2026-02-19 at 12.49.24.png>) | ![alt text](<assets/Screenshot 2026-02-19 at 12.49.08.png>) | ![alt text](<assets/Screenshot 2026-02-19 at 12.47.33.png>) | ![alt text](<assets/Screenshot 2026-02-19 at 12.47.00.png>) | |


## Chat Interface

| Description | Object Detection | Object Detection | Object Pointing |
| --- | --- | --- | --- |
| ![alt text](<assets/Screenshot 2026-02-18 at 19.58.06.png>) | ![alt text](<assets/Screenshot 2026-02-18 at 20.01.10.png>) | ![alt text](<assets/Screenshot 2026-02-18 at 20.01.49.png>) | ![alt text](<assets/Screenshot 2026-02-18 at 20.02.31.png>) |


| Basic OCR | Explicit structure data query response | Default structured data dump | Default structured data dump |
| --- | --- | --- | --- |
| ![alt text](<assets/Screenshot 2026-02-18 at 21.12.04.png>) | ![alt text](<assets/Screenshot 2026-02-18 at 20.54.56.png>) | ![alt text](<assets/Screenshot 2026-02-18 at 20.50.56.png>) | ![alt text](<assets/Screenshot 2026-02-18 at 20.50.58.png>) |

Two interfaces for visual understanding, both running entirely on your machine with no data leaving your computer.

**Chat** is the conversational mode. Two AI models work together: [Moondream 3](https://moondream.ai/blog/moondream-3-preview) handles vision tasks (understanding images, detecting objects, reading text), while [Qwen 3 4B Instruct](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M) acts as a language orchestrator that interprets what you're asking for and routes to the right capability. You just talk naturally — "find all the cars and describe the red one" — and the system figures out the rest.

**Playground** is the direct mode. No LLM in the loop. You pick exactly which Moondream capability to run (caption, query, detect, point, or OCR), fill in the inputs, and hit Run. It's the tool for when you know what you want and don't need an orchestrator guessing.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![Gradio 6](https://img.shields.io/badge/gradio-6.x-orange.svg)
![Ollama](https://img.shields.io/badge/ollama-0.16+-blueviolet.svg)
![macOS](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Architecture Overview

The system uses a two-model architecture. This separation exists because vision models are great at seeing but not great at understanding conversational nuance, while language models are great at parsing intent but can't see images. By combining them, we get the best of both.

```mermaid
graph LR
    U[User] -->|text + image| G[Gradio UI]
    G -->|user message + history| O[Qwen 3 4B Instruct<br/>Orchestrator]
    O -->|structured JSON| G
    G -->|image + action| M[Moondream 3<br/>Vision Model]
    M -->|results| G
    G -->|text + annotated images| U

    style O fill:#0891b2,color:#fff
    style M fill:#16a34a,color:#fff
    style G fill:#ca8a04,color:#fff
```

**Qwen 3 4B Instruct** (via [Ollama](https://ollama.com)) is the orchestrator. It receives the user's natural language message along with recent conversation history, and returns structured JSON specifying which vision capability to invoke. It handles ambiguous references ("what is it?"), compound requests ("find the cars and describe the red one"), and natural phrasing that a regex parser would miss. It runs with low temperature for fast, deterministic classification — typically responding in under 500ms.

> [!IMPORTANT]
> We use the **instruct** variant (`qwen3:4b-instruct-2507-q4_K_M`), not the default `qwen3:4b`. The default Qwen 3 model uses a "thinking" mode that conflicts with Ollama's structured JSON output, producing empty responses. The instruct variant skips chain-of-thought and gives direct, schema-compliant output. See [ollama/ollama#10929](https://github.com/ollama/ollama/issues/10929) and [ollama/ollama#12917](https://github.com/ollama/ollama/issues/12917) for background.

**Moondream 3** (via [Moondream Station](https://docs.moondream.ai/station/)) is the vision model. It's a 9-billion-parameter Mixture-of-Experts model that activates roughly 2 billion parameters per inference, keeping it fast despite its size. It supports four distinct capabilities: visual question answering, image captioning, object detection with bounding boxes, and object pointing with center coordinates. On Apple Silicon, Station uses MLX for native acceleration.

If Ollama isn't running, the system gracefully degrades to a built-in regex-based intent parser. You lose context resolution and multi-step support, but single queries still work fine.

## Request Flow

Here's what happens when you type a message:

```mermaid
sequenceDiagram
    participant U as User
    participant G as Gradio App
    participant O as Qwen 3 4B Instruct (Ollama)
    participant M as Moondream 3 (Station)
    participant R as Renderer

    U->>G: "Find all hammers and describe the biggest one"
    G->>G: Resolve current image (uploaded or from history)
    G->>O: Send message + last 3 conversation exchanges
    O->>G: {"action":"multi","steps":[{"action":"detect","subject":"hammers"},{"action":"query","question":"Describe the biggest hammer"}]}
    G->>M: detect(image, "hammers")
    M->>G: {objects: [{x_min, y_min, x_max, y_max}, ...]}
    G->>R: draw_bounding_boxes(image, objects)
    R->>G: annotated image with boxes
    G->>M: query(image, "Describe the biggest hammer")
    M->>G: {answer: "The largest hammer is a..."}
    G->>U: "Found 3 hammers" + annotated image + description
```

The orchestrator is the key piece that makes compound requests like this work. Without it, "find all hammers and describe the biggest one" would just go to a single query call and return a text answer with no bounding boxes.

## Vision Capabilities

Moondream 3 exposes four distinct API methods, plus powerful prompt-driven features like OCR and structured output. The orchestrator picks the right approach based on what you're asking.

```mermaid
graph TD
    Q[User Message] --> O{Orchestrator<br/>Qwen 3 4B Instruct}
    O -->|"What breed is this?"| VQA[query<br/>Visual QA]
    O -->|"Describe this image"| CAP[caption<br/>Description]
    O -->|"Find all cars"| DET[detect<br/>Bounding Boxes]
    O -->|"Where's the cat?"| PT[point<br/>Location Markers]
    O -->|"Read the text"| OCR[query<br/>OCR / Text Extraction]
    O -->|"Extract JSON with keys: ..."| STR[query<br/>Structured Output]

    VQA --> T1[Text answer]
    CAP --> T2[Text description]
    DET --> B[Annotated image + raw coords]
    PT --> P[Annotated image + raw coords]
    OCR --> T3[Extracted text / markdown]
    STR --> T4[JSON / CSV / XML]

    style O fill:#0891b2,color:#fff
    style VQA fill:#16a34a,color:#fff
    style CAP fill:#16a34a,color:#fff
    style DET fill:#16a34a,color:#fff
    style PT fill:#16a34a,color:#fff
    style OCR fill:#16a34a,color:#fff
    style STR fill:#16a34a,color:#fff
```

**Query** is the most flexible — it handles any natural language question about the image. By default it uses Moondream 3's reasoning mode, where the model "thinks" before answering. The orchestrator disables reasoning for simple factual questions (like "what color is the sky?") to save time.

**Caption** generates a text description at three levels of detail: short (one-liner), normal (a sentence or two), or long (detailed paragraph). The orchestrator picks the length based on how you phrase it ("brief description" → short, "describe in detail" → long).

**Detect** finds all instances of a given object and returns bounding box coordinates normalized to 0–1. The renderer converts these to pixel coordinates and draws semi-transparent colored rectangles on a copy of the image. Each box gets a label, and colors cycle through 8 distinct values when multiple objects are found. Raw coordinate data is shown in a collapsible block below the annotated image.

**Point** locates objects by their center point, also normalized to 0–1. The renderer draws crosshair markers with contrasting rings (white outer, red inner) so they're visible on any background. Useful when you want precise location rather than a bounding region. Raw point data is also shown in a collapsible block.

**OCR & Text Extraction** — Moondream 3 has strong OCR capabilities. Ask "read the text", "OCR this", or "what does the sign say?" and the orchestrator routes to query with reasoning disabled for fast extraction. You can also ask "convert to markdown" to get formatted output from tables, documents, or screenshots.

**Structured Output** — Moondream 3 natively generates JSON, CSV, XML, and markdown when you include the format in your prompt. Ask "extract a JSON array with keys: name, color, position" and it returns structured data directly. This is handled via query — Moondream parses the prompt format instructions itself.

> [!TIP]
> For detect and point results, expand the "Raw data" section below the annotated image to see the exact normalized coordinates returned by Moondream. This is useful for building on top of the API or debugging detection accuracy.

All coordinates from Moondream are normalized — values between 0 and 1, where (0,0) is top-left and (1,1) is bottom-right. The renderer handles the conversion to actual pixel positions based on the image dimensions.

## What Moondream Can and Can't Do Locally

Moondream 3's full API surface is broader than what's available through Moondream Station. Knowing the boundaries matters because the orchestrator can only route to capabilities that actually work locally.

**Available locally (our toolkit):** `query`, `detect`, `point`, `caption`. These four cover the vast majority of visual understanding tasks — question answering, object localization, text extraction, structured output, and image description. Both Chat and Playground use these. The Playground also surfaces OCR as a dedicated task (internally it's `query` with a text extraction prompt and reasoning disabled).

**Cloud-only (not available in Station):** `segment` returns SVG path masks for pixel-level object boundaries. It's a cloud preview feature — Station returns `Function 'segment' not available` when you try it locally. If a user asks "outline the car" or "mask the background" in Chat, the system falls back to `detect` (bounding boxes) as the closest local approximation.

**Experimental:** `gaze` estimates where a person is looking. It exists in the API docs but isn't reliable enough to route to by default.

**Single-subject constraint:** `detect` and `point` accept exactly one subject per call. "Find all cars" works; "find the car and the dog" requires two separate calls. The orchestrator handles this by splitting multi-subject requests into a `multi` action with one detect/point step per item. This is a Moondream API constraint, not an orchestrator limitation.

**OCR timeout risk:** Dense text images (legal documents, full-page screenshots) can push inference past Station's default 30-second timeout, especially on lower-memory machines. The `config.yaml` exposes `moondream.timeout` (default 90s) and `moondream.max_tokens` (default 2048) to handle these cases. If OCR times out, increase these values.

## Routing Disambiguation

The hardest problem in this system isn't vision — it's figuring out what the user actually wants. Natural language is ambiguous, and similar-sounding requests can require completely different action pipelines.

**The core tension:** "Highlight all clothes with their colors" and "What colors are the clothes?" sound almost identical but need different treatment. The first wants bounding boxes drawn on the image AND a text answer about colors (multi: detect + query). The second just wants a text answer (query alone). A third variation, "highlight cloth colors", is ambiguous — does "highlight" mean draw boxes or emphasize in text?

**How we mitigate this:** The orchestrator's system prompt uses few-shot examples rather than verbose rules. Small models (4B parameters) degrade with long instructions due to attention dilution — they follow examples better than explanations. The prompt includes ~28 examples covering:

- **Action keywords** → capability mapping: "show me", "find", "highlight", "mark" route to detect/point, not caption or query.
- **Compound requests** → multi-step pipelines: "find X and describe Y" splits into detect + query.
- **Context resolution** → pronoun/reference handling: "show me" after discussing glasses resolves to detect glasses.
- **Attribute + localization** → multi: "highlight clothes with colors" means detect clothing, then query about colors. The visual localization comes first, the textual answer second.
- **Reasoning toggle** → speed optimization: simple yes/no questions, OCR, and structured output skip the reasoning step. Complex spatial or analytical questions enable it.

**Where it still gets tricky:** Edge cases exist. "Read the text and highlight the title" is a multi (query for OCR + detect for title), but a 4B model might collapse it to just query. Very long or nested compound requests sometimes lose a step. The practical mitigation: if the result isn't what you expected, rephrase with explicit action verbs. "Find the title" is unambiguous in a way "highlight the title" sometimes isn't.

> [!TIP]
> For best results, use direct action verbs: "find" / "detect" for bounding boxes, "point to" for location markers, "describe" for captions, "read" / "OCR" for text extraction. Compound requests work best when structured as "do X and then Y" — e.g., "find all people and describe what they're wearing".

## Recommended Chat Flow

A good conversation with this tool follows a natural escalation pattern — start broad, then drill into specifics. The image persists across messages, so you upload once and keep exploring.

**1. Orient** — Start with a caption or open-ended question to understand the image.
```
[upload image]
"What's happening here?"
```

**2. Locate** — Use detect or point to find specific objects.
```
"Find all the people"
"Point to the red car"
```

**3. Drill down** — Ask follow-up questions. The orchestrator resolves "it" / "them" from context.
```
"What is the person on the left wearing?"
"Is there a dog in the background?"
"Show me the dog"
```

**4. Extract** — Pull structured data or text from the image.
```
"Read all the text"
"Extract a JSON with keys: object, color, position"
"Convert to markdown"
```

**5. Compound** — Combine actions when you need both visual and textual answers.
```
"Find all the tools and describe the biggest one"
"Highlight his sweater, pants, and glasses"
"Detect all vehicles and tell me their colors"
```

Each step builds on context from previous messages. The orchestrator sees the last 3 exchanges (6 messages), so references like "it", "that one", and "show me" resolve correctly within that window. If you switch to a new image, the context resets naturally.

> [!NOTE]
> The stop button (■) appears in the message box while processing. It cancels between pipeline steps — if you're midway through a 3-step multi action and press stop, you keep the results from completed steps. It can't interrupt a single in-flight Moondream call (that's a network-level limitation), but it prevents the next step from starting.

## Playground Mode

The Playground is the other way to use this project. Where Chat interprets your natural language and decides which capability to call, the Playground puts you in direct control. You pick the task, you provide the inputs, you see exactly what Moondream returns.

```bash
uv run python main.py --mode playground --port 4030
```

The interface is a single page with everything visible at once — no tabs hiding capabilities, no buried settings. The layout:

- **Image upload** at the top (shared across all tasks — upload once, switch tasks freely)
- **Task selector** as radio buttons: Caption · Query · Detect · Point · OCR
- **Task-specific inputs** that appear based on your selection — each with descriptions and placeholder text explaining what to enter
- **Run button** — one click, one result
- **Output panel** — text result on top, annotated image below (for detect/point), and a collapsible raw JSON section with the exact data Moondream returned

**When to use Playground over Chat:**

The Playground is better when you already know which capability you want. "I want to run detect on 'street signs'" is faster in Playground — no orchestrator overhead, no risk of misrouting. The dedicated OCR task with plain text / markdown format toggle is also more discoverable here than in Chat, where you'd need to phrase your request as "read the text" or "convert to markdown".

Chat is better for exploration and compound tasks. "Find all the cars and tell me about the red one" requires two Moondream calls (detect + query) stitched together — Chat handles that automatically, while Playground would need you to run each step manually.

Both apps share the same theme (`hmb/amethyst`), the same Moondream client, and the same renderer for annotated images. They're two views of the same underlying system.

> [!TIP]
> The Playground is a great way to understand what Moondream can actually do before relying on the Chat orchestrator to route for you. Try each capability manually to build intuition for what kinds of prompts work best with each task.

## How the Orchestrator Works

The orchestrator sends a lean system prompt to Qwen 3 4B Instruct with few-shot examples that teach the JSON output format — including OCR, structured output, and reasoning flag examples. Ollama's structured output feature constrains the response to match a predefined JSON schema, so parsing never fails.

```mermaid
graph LR
    subgraph Orchestrator
        SP[System Prompt<br/>~180 tokens] --> MSG[User Message<br/>+ 3 recent exchanges]
        MSG --> LLM[Qwen 3 4B]
        LLM --> JSON[Structured JSON]
    end

    JSON --> |single| A1[One Action]
    JSON --> |multi| A2[Action Sequence]

    style LLM fill:#0891b2,color:#fff
```

The system prompt is intentionally small. Research shows that small models (1–4B parameters) degrade significantly with long system prompts due to attention dilution — the more instructions you stuff in, the worse the model follows any of them. Instead of verbose rules, the prompt uses few-shot examples that demonstrate the pattern. The model learns the mapping from natural language to structured actions by seeing examples rather than reading explanations.

Conversation history is limited to the last 6 messages (3 exchanges). This gives the model enough context to resolve references like "it" or "that one" without overloading its context window. Only text content is passed — images are stripped since the orchestrator doesn't need to see them.

### Regex Fallback (`intent.py`)

When Ollama is unreachable (connection refused or timeout), the orchestrator automatically falls back to [`src/intent.py`](src/intent.py), a regex-based parser. This keeps the app functional without any external LLM dependency — you just lose the smarter features.

The fallback checks patterns in a fixed priority order — first match wins:

1. **Caption** — triggers on keywords like `describe`, `caption`, `what's in this image`, `tell me about this photo`. Also detects length modifiers (`brief` → short, `detailed` → long).
2. **Detect** — triggers on `find all X`, `detect X`, `where are the X`, `how many X`, `count X`. Extracts the subject from the matched group (e.g., "find all **cars**" → subject is `cars`).
3. **Point** — triggers on `point to X`, `where's the X`, `show me the X`. Extracts a singular subject.
4. **Query** — catch-all fallback. Any message that doesn't match the above patterns goes to Moondream's visual QA as a direct question.

> [!NOTE]
> The regex parser has no conversation memory, so context-dependent messages like "what is it?" will go straight to query without resolving "it". It also can't handle compound requests ("find the cars and describe the red one" would match detect only). These limitations are exactly why we added the LLM orchestrator.

## Image Persistence

You don't need to re-upload the image with every message. The app maintains image context across the conversation:

1. If you upload an image with your message, that becomes the current image.
2. If you send text without an image, the app searches backward through the chat history to find the most recent uploaded image.
3. If no image exists anywhere, you get a friendly prompt to upload one.

This means you can upload once, then ask 10 different questions about the same image. Upload a new image at any time to switch context.

## Terminal Logging

Every request flow is logged to the terminal with colored output using [loguru](https://github.com/Delgan/loguru). Each component gets its own tag so you can trace the full pipeline:

```
18:42:01 [APP] New message: "find all hammers" (1 file(s))
18:42:01 [APP] Loaded uploaded image: 1920x1080
18:42:01 [ORCHESTRATOR] User message: "find all hammers"
18:42:01 [ORCHESTRATOR] Sending to qwen3:4b-instruct-2507-q4_K_M via Ollama...
18:42:02 [ORCHESTRATOR] LLM response: {"action":"detect","subject":"hammers"}
18:42:02 [ORCHESTRATOR] Action 1: detect | subject=hammers
18:42:02 [MOONDREAM] detect("hammers")
18:42:03 [MOONDREAM] detect result (842ms): found 2 object(s)
18:42:03 [APP] Returning 2 message(s) to chat
```

For even more visibility into what Ollama is doing (token generation speed, model loading, memory usage), start Ollama in debug mode:

```bash
OLLAMA_DEBUG=1 ollama serve
```

## Prerequisites

You need two local services running before starting the app:


**1. Ollama with Qwen 3 4B Instruct** — the language orchestrator

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama serve
```

> [!WARNING]
> Do **not** use `ollama pull qwen3:4b` — that pulls the thinking variant which produces empty JSON responses when used with structured output. Always use the instruct variant above.

Runs on `localhost:11434`. If you skip this step, the app still works using the regex fallback — just without smart intent parsing.

For more details: [Moondream Station docs](https://docs.moondream.ai/station/) · [Ollama docs](https://ollama.com)

**2. Setup**

Clone the repo and install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/your-username/better_moondream_demo.git
cd better_moondream_demo
uv sync
```

> [!Tip]
> If you don't have uv:
> 
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```

**3. Moondream Station** — the vision model server

```bash
# Note: Source the venv of the project first  
moondream-station
```

First run downloads the model weights (quantized, a few GB). After that it starts in seconds. Runs on `localhost:2020`.


## Configuration

All settings live in [`config.yaml`](config.yaml) at the project root. Edit it to change endpoints, models, ports, or theme without touching Python code:

```yaml
moondream:
  endpoint: "http://localhost:2020/v1"

ollama:
  url: "http://localhost:11434"
  model: "qwen3:4b-instruct-2507-q4_K_M"
  temperature: 0.1
  num_predict: 256
  timeout: 30

app:
  host: "0.0.0.0"       # LAN-accessible by default
  port: 7860
  theme: "hmb/amethyst"
  share: false
```

Every key has a sensible default — the app runs fine even if `config.yaml` is missing.

## Running

Three terminals for Chat mode (the default), two for Playground:

```bash
# Terminal 1: Vision model host
# Note: Source the venv of the project first
moondream-station

# Terminal 2: Language orchestrator (only needed for Chat mode)
ollama serve

# Terminal 3: Launch Chat (default)
uv run python main.py

# Or launch Playground instead
uv run python main.py --mode playground
```

By default it binds to `0.0.0.0:7860`, so it's accessible from other devices on your LAN. Override with CLI flags:

```bash
uv run python main.py --port 8080
uv run python main.py --mode playground --port 9000
uv run python main.py --host 127.0.0.1 --port 4030
```

CLI flags take priority over `config.yaml` values. The `--mode` flag accepts `chat` (default) or `playground`.

> [!TIP]
> The Playground doesn't need Ollama at all — it talks directly to Moondream Station. If you just want to explore vision capabilities without setting up a language model, Playground is the way to go.

## Example Conversations (Chat Mode)

| You type | What happens |
|---|---|
| *(just upload, no text)* | Auto-generates a description |
| "What's happening in this photo?" | Visual QA with reasoning enabled |
| "What color is the sky?" | Visual QA with reasoning disabled (fast) |
| "Describe this image in detail" | Long-form caption |
| "Give me a short caption" | Brief one-liner |
| "Find all people" | Bounding boxes + raw coordinate data |
| "How many cars are there?" | Detection with count + collapsible coordinates |
| "Where's the dog?" | Crosshair marker + raw point data |
| "Point to the red button" | Precise location marker |
| "Read all the text in this image" | OCR text extraction |
| "Convert to markdown" | OCR with markdown formatting (tables, headings) |
| "Extract a JSON array with keys: name, color" | Structured JSON output |
| "What does the sign say?" | Quick text reading (no reasoning) |
| "Find the tools and describe the biggest one" | Multi-step: detect + query |
| "What is it?" *(after discussing a hammer)* | Context-aware follow-up |

## Project Structure

```
better_moondream_demo/
├── main.py               Entry point with CLI args (--host, --port, --mode)
├── config.yaml           All configurable settings (endpoints, models, ports)
├── pyproject.toml         Dependencies and project config
├── README.md
├── .gitignore
└── src/
    ├── __init__.py
    ├── config.py          YAML config loader with defaults
    ├── app.py             Gradio chat interface (--mode chat)
    ├── playground.py      Gradio playground interface (--mode playground)
    ├── orchestrator.py    LLM-powered intent parsing via Ollama
    ├── client.py          Moondream Station client wrapper (query, caption, detect, point)
    ├── intent.py          Regex fallback parser (used when Ollama is down)
    ├── renderer.py        Draws bounding boxes and point markers on images
    └── logger.py          Loguru-based colored terminal logging
```

## Dependencies

| Package | Purpose |
|---|---|
| [gradio](https://gradio.app) | Chat interface with multimodal input |
| [moondream](https://pypi.org/project/moondream/) | Python client for the Moondream vision API |
| [moondream-station](https://pypi.org/project/moondream-station/) | Local vision model inference server |
| [requests](https://docs.python-requests.org/) | HTTP client for Ollama API |
| [loguru](https://github.com/Delgan/loguru) | Colored structured logging |
| [Pillow](https://python-pillow.org/) | Image loading and annotation drawing |
| [PyYAML](https://pyyaml.org/) | Configuration file loading |

Ollama is installed separately (not a Python dependency) — see [ollama.com](https://ollama.com).

## Swapping the Orchestrator Model

The orchestrator model is configured in [`config.yaml`](config.yaml):

```yaml
ollama:
  model: "qwen3:4b-instruct-2507-q4_K_M"
```

To use a different model:

1. Pull it via Ollama: `ollama pull <model-name>`
2. Update `ollama.model` in `config.yaml`
3. Restart the app

> [!NOTE]
> Any Ollama model that supports structured JSON output (the `format` parameter) should work. If you choose a model with a "thinking" mode (like the default `qwen3:4b`), it will likely produce empty responses — pick an instruct/non-thinking variant instead.

The system prompt in `SYSTEM_PROMPT` is tuned to be small (~180 tokens) for 4B-class models. If you swap to a larger model (8B+), you could expand the prompt with more examples or rules without degrading performance.

## About the Models

### Moondream 3

A 9B-parameter Mixture-of-Experts vision-language model. It dynamically routes tokens across 64 experts, activating only 8 per inference. This keeps latency low while maintaining high accuracy. It has a 32K context window and uses a SigLIP-based vision encoder with multi-crop channel concatenation for token-efficient high-resolution image processing.

Key features we use: visual QA with toggleable reasoning mode, image captioning, object detection (bounding boxes), object pointing (center coordinates), OCR/text extraction, and native structured output (JSON, markdown, CSV, XML). The `encode_image()` method lets us encode once and reuse across multiple queries on the same image for better performance.

On Apple Silicon, Moondream Station uses MLX for native acceleration with quantized weights. You need at least 16GB of unified memory.

> [!NOTE]
> Moondream 3 also supports [segmentation](https://docs.moondream.ai/skills/segment/) (SVG path masks), but this is currently a cloud-only preview — Station returns `Function 'segment' not available` when called locally.

[Model announcement](https://moondream.ai/blog/moondream-3-preview) · [HuggingFace weights](https://huggingface.co/moondream/moondream3-preview) · [Documentation](https://docs.moondream.ai/)

### Qwen 3 4B Instruct

A 4B-parameter language model from Alibaba's Qwen team. Used here purely as an intent classifier and orchestrator — it never sees the images. We use the **instruct variant** (`qwen3:4b-instruct-2507-q4_K_M`) specifically because it produces direct structured output without chain-of-thought reasoning, which is what we need for fast JSON classification.

> [!NOTE]
> Qwen 3 models were split into "thinking" and "instruct" variants after a July 2025 update. The default tag (`qwen3:4b`) points to the thinking variant where `/no_think` and `think: false` [no longer work](https://github.com/ollama/ollama/issues/12917). The instruct variant avoids this entirely.

[Ollama page](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M) · [Qwen 3 announcement](https://qwenlm.github.io/blog/qwen3/)

## Future: Multi-User on LAN

The app is designed to run on a local network AI server (`0.0.0.0` binding), and multiple people can use it simultaneously without any code changes.

**Gradio** handles concurrent users natively — each browser session gets its own isolated state with separate chat history and image context. Five people opening the app from different machines on the LAN each get their own independent conversation.

**Ollama** queues concurrent requests internally. Orchestrator calls are fast (sub-500ms), so even with several users the queue clears quickly.

**Moondream Station** is the bottleneck. It runs on one GPU/accelerator and processes inference sequentially. If user A sends a detect request and user B sends a query at the same time, B waits until A finishes. For typical requests (1–3 seconds each), this is invisible with a small team (2–8 people) since users naturally stagger their requests — someone's typing while someone else is reading results.

**If it ever becomes a problem:** Run multiple Moondream Station instances on different ports (if you have multiple GPUs or machines) and add round-robin distribution in `client.py`. But for smaller situations, the single-instance setup can handle concurrent usage just fine.

## License

[MIT](LICENSE)
