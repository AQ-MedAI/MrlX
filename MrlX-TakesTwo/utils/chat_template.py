"""
Module: chat_template
---------------------
Loads and caches the chat template (Jinja format) for Qwen3 assistant mask.
This template will be used to format doctor-patient conversations for tokenizer input.
"""

import os
from pathlib import Path


def _load_template_once():
    """Load the qwen3 assistant mask template from jinja file only once."""
    template_path = (
        Path(__file__).parent.parent / "chat_template" / "qwen3_assistant_mask.jinja"
    )
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


# Global constant for reuse
TEMPLATE_QWEN3_ASSISTANT_MASK = _load_template_once()
