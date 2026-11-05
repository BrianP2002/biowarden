"""Model aliases and per-token prices (USD per million tokens) for the cost guard.

Prices are from the model catalogue as of 2026-10-28. Prompt-cache writes cost 1.25x and cache
reads 0.1x the input price (5-minute cache). Haiku 5.5 prompts over 100K tokens bill at a higher
rate; trials stay far below that.
"""

from __future__ import annotations

from typing import Any

ALIASES = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5",
    # open-weight models through OpenRouter (ids checked against its catalogue on 2026-10-28)
    "dsflash": "deepseek/deepseek-v4.1-flash",
    "dspro": "deepseek/deepseek-v4-pro",
    "glm": "z-ai/glm-5.3",
    "glmflash": "z-ai/glm-5.3-flash",
    "qwenflash": "qwen/qwen3.8-flash",  # the only Qwen to run: larger ones cost 6-10x more
    "kimi": "moonshotai/kimi-k3",
    "mimo": "xiaomi/mimo-v2.6-flash",
    "gptoss": "openai/gpt-oss-120b",
}
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-5-5": (4.0, 20.0),
}
CACHE_WRITE, CACHE_READ = 1.25, 0.10


def usage_usd(model: str, usage: Any) -> float:
    price_in, price_out = PRICES[model]
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    billed_in = usage.input_tokens + CACHE_WRITE * cache_write + CACHE_READ * cache_read
    return (billed_in * price_in + usage.output_tokens * price_out) / 1e6
