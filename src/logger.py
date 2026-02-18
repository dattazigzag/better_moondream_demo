"""
Colored logging for the Moondream Chat app.

Provides a simple logger that color-codes output by component so you
can see the full request flow in the terminal:

    [ORCHESTRATOR] cyan    — LLM intent parsing
    [MOONDREAM]    green   — vision model calls
    [APP]          yellow  — Gradio app logic
    [ERROR]        red     — errors from any component
"""

import sys
import logging


# ANSI color codes
COLORS = {
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "MAGENTA": "\033[95m",
    "DIM": "\033[2m",
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
}


class ColorFormatter(logging.Formatter):
    """
    Custom formatter that adds color based on the logger name.
    """

    COMPONENT_COLORS = {
        "orchestrator": COLORS["CYAN"],
        "moondream": COLORS["GREEN"],
        "app": COLORS["YELLOW"],
        "renderer": COLORS["MAGENTA"],
    }

    LEVEL_COLORS = {
        logging.ERROR: COLORS["RED"],
        logging.WARNING: COLORS["RED"],
    }

    def format(self, record: logging.LogRecord) -> str:
        # Pick color: error/warning overrides component color
        color = self.LEVEL_COLORS.get(record.levelno)
        if color is None:
            color = self.COMPONENT_COLORS.get(record.name, COLORS["DIM"])

        tag = record.name.upper()
        timestamp = self.formatTime(record, "%H:%M:%S")

        dim = COLORS["DIM"]
        reset = COLORS["RESET"]
        bold = COLORS["BOLD"]

        return f"{dim}{timestamp}{reset} {color}{bold}[{tag}]{reset} {color}{record.getMessage()}{reset}"


def get_logger(name: str) -> logging.Logger:
    """
    Get a colored logger for the given component name.

    Usage:
        from src.logger import get_logger
        log = get_logger("orchestrator")
        log.info("Sending prompt to Qwen 3 4B")
        log.error("Ollama not reachable")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    return logger
