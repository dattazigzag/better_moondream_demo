"""
Intent detection for chat messages.

Parses user messages to determine which Moondream capability to invoke:
- "describe" / "caption" / "what's in this" → caption
- "find" / "detect" / "locate" / "where are the" → detect
- "point to" / "point at" / "show me where" → point
- Everything else → query (visual question answering)
"""
