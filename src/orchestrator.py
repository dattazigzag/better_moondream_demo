"""
LLM-powered intent orchestrator using Qwen 3 4B via Ollama.

Replaces the regex-based intent.py with actual language understanding.
Sends the user message + recent conversation context to a local LLM,
gets back structured JSON telling us which Moondream capability to
call, with what arguments, and in what order.

Falls back to the regex-based intent parser if Ollama is not available.
"""

import json
import requests
from dataclasses import dataclass, field

from src.logger import get_logger

log = get_logger("orchestrator")


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"

# Lean system prompt — under 200 tokens.
# Few-shot examples teach the output format without lengthy explanations.
SYSTEM_PROMPT = """You route user queries about images to vision model actions. Return JSON only, no other text.

Available actions:
- query: Ask a question about the image. Use for any general question.
- caption: Generate image description. Lengths: short, normal, long.
- detect: Find objects with bounding boxes. Needs a subject.
- point: Locate object center points. Needs a subject.
- multi: Multiple sequential actions. Use when user asks for more than one thing.

Examples:
User: "What's happening here?" → {"action":"query","question":"What's happening here?"}
User: "Describe this" → {"action":"caption","length":"normal"}
User: "Give me a detailed description" → {"action":"caption","length":"long"}
User: "Find all cars" → {"action":"detect","subject":"cars"}
User: "How many people are there?" → {"action":"detect","subject":"people"}
User: "Where's the dog?" → {"action":"point","subject":"dog"}
User: "Find the cars and describe the red one" → {"action":"multi","steps":[{"action":"detect","subject":"cars"},{"action":"query","question":"Describe the red car"}]}
User: "What is it?" (previous topic: hammer) → {"action":"query","question":"What is the hammer?"}

Rules:
- If user says "it"/"that"/"this one", resolve from context and put the full question in the output.
- For counting questions, use detect (bounding boxes show the count).
- When unsure, default to query.
- Return ONLY valid JSON. No markdown, no explanation."""


# JSON schema for structured output (Ollama's format parameter)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["query", "caption", "detect", "point", "multi"],
        },
        "question": {"type": "string"},
        "subject": {"type": "string"},
        "length": {"type": "string", "enum": ["short", "normal", "long"]},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "question": {"type": "string"},
                    "subject": {"type": "string"},
                    "length": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
    "required": ["action"],
}


@dataclass
class Action:
    """A single action to perform on the image."""

    action: str  # query, caption, detect, point
    question: str | None = None
    subject: str | None = None
    length: str = "normal"


@dataclass
class OrchestratorResult:
    """Result from the orchestrator — one or more actions to perform."""

    actions: list[Action] = field(default_factory=list)
    error: str | None = None
    used_fallback: bool = False


def _build_context_messages(
    user_message: str, conversation_history: list[dict]
) -> list[dict]:
    """
    Build the messages array for the Ollama chat API.

    Includes the system prompt, recent conversation context (last 6
    exchanges max to stay within small model limits), and the current
    user message.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include recent history for context resolution ("it", "that", etc.)
    # Keep it short — last 6 messages max (3 exchanges)
    recent = conversation_history[-6:] if conversation_history else []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Only include text content, skip file blocks
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = " ".join(text_parts)
        if isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": user_message})
    return messages


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen 3 may emit."""
    import re

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_llm_response(raw: str) -> OrchestratorResult:
    """
    Parse the LLM's JSON response into an OrchestratorResult.
    """
    # Clean any residual thinking tags
    cleaned = _strip_think_tags(raw)
    if not cleaned:
        log.error(f"LLM returned empty content (raw was {len(raw)} chars, likely all <think> tags)")
        return OrchestratorResult(error="LLM returned empty response — thinking mode may be active")

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log.error(f"Failed to parse JSON from LLM: {cleaned[:200]}")
        return OrchestratorResult(error=f"Invalid JSON from LLM: {cleaned[:100]}")

    action_type = data.get("action", "query")

    if action_type == "multi":
        steps = data.get("steps", [])
        actions = []
        for step in steps:
            actions.append(
                Action(
                    action=step.get("action", "query"),
                    question=step.get("question"),
                    subject=step.get("subject"),
                    length=step.get("length", "normal"),
                )
            )
        if not actions:
            return OrchestratorResult(error="Multi action with no steps")
        return OrchestratorResult(actions=actions)
    else:
        action = Action(
            action=action_type,
            question=data.get("question"),
            subject=data.get("subject"),
            length=data.get("length", "normal"),
        )
        return OrchestratorResult(actions=[action])


def orchestrate(
    user_message: str, conversation_history: list[dict] | None = None
) -> OrchestratorResult:
    """
    Parse the user's message into structured action(s) using the LLM.

    Sends the message + recent conversation context to Qwen 3 4B via
    Ollama. Returns an OrchestratorResult with one or more Actions.

    Falls back to regex-based intent parsing if Ollama is unreachable.

    Args:
        user_message: The raw text from the user.
        conversation_history: Recent chat history (Gradio format).

    Returns:
        OrchestratorResult with actions to perform.
    """
    history = conversation_history or []
    messages = _build_context_messages(user_message, history)

    log.info(f"User message: \"{user_message}\"")
    log.debug(f"Context messages: {len(messages)} total ({len(messages) - 2} history)")

    try:
        log.info(f"Sending to {MODEL} via Ollama...")

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "format": OUTPUT_SCHEMA,
                "options": {
                    "temperature": 0.1,  # Low temp for consistent structured output
                    "num_predict": 256,  # Intent JSON is short
                },
            },
            timeout=30,
        )
        response.raise_for_status()

        result_json = response.json()
        content = result_json.get("message", {}).get("content", "")

        log.info(f"LLM response: {content}")

        # Parse timing info if available
        total_ns = result_json.get("total_duration", 0)
        if total_ns:
            total_ms = total_ns / 1_000_000
            log.debug(f"Ollama round-trip: {total_ms:.0f}ms")

        result = _parse_llm_response(content)

        for i, action in enumerate(result.actions):
            log.info(f"Action {i + 1}: {action.action} | subject={action.subject} question={action.question} length={action.length}")

        return result

    except requests.ConnectionError:
        log.warning("Ollama not reachable — falling back to regex intent parser")
        return _fallback_parse(user_message)
    except requests.Timeout:
        log.warning("Ollama timed out — falling back to regex intent parser")
        return _fallback_parse(user_message)
    except Exception as e:
        log.error(f"Orchestrator error: {e}")
        return _fallback_parse(user_message)


def _fallback_parse(user_message: str) -> OrchestratorResult:
    """
    Fall back to the regex-based intent parser when Ollama is unavailable.
    """
    from src.intent import Capability, parse_intent

    log.info("Using regex fallback parser")
    intent = parse_intent(user_message)

    if intent.capability == Capability.CAPTION:
        action = Action(action="caption", length=intent.caption_length)
    elif intent.capability == Capability.DETECT:
        action = Action(action="detect", subject=intent.subject)
    elif intent.capability == Capability.POINT:
        action = Action(action="point", subject=intent.subject)
    else:
        action = Action(action="query", question=intent.question or user_message)

    log.info(f"Fallback action: {action.action} | subject={action.subject} question={action.question}")
    return OrchestratorResult(actions=[action], used_fallback=True)


def is_ollama_available() -> bool:
    """Check if Ollama is running and the model is loaded."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        available = any(MODEL.split(":")[0] in m for m in models)
        if available:
            log.debug(f"Ollama available with {MODEL}")
        else:
            log.warning(f"{MODEL} not found in Ollama. Available: {models}")
        return available
    except Exception:
        return False
