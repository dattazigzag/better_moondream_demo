"""
Logging setup using loguru.

Configures a single shared logger with colored, formatted output.
Each component binds its name for easy filtering in the terminal:

    [ORCHESTRATOR] — LLM intent parsing via Ollama
    [MOONDREAM]    — vision model calls via Station
    [APP]          — Gradio app logic
    [RENDERER]     — image annotation

Usage:
    from src.logger import get_logger
    log = get_logger("orchestrator")
    log.info("Sending prompt to Qwen 3 4B")
"""

import sys

from loguru import logger

# Remove loguru's default handler so we can set our own format
logger.remove()

# Custom format: timestamp, colored component tag, message
logger.add(
    sys.stdout,
    format=(
        "<dim>{time:HH:mm:ss}</dim> "
        "<level><bold>[{extra[component]}]</bold> {message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)


def get_logger(name: str) -> logger.__class__:
    """
    Get a logger bound to a specific component name.

    The component name appears as a colored tag in the output,
    making it easy to trace the flow: orchestrator → moondream → app.
    """
    return logger.bind(component=name.upper())
