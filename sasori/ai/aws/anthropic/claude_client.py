from __future__ import annotations
from typing import Any, Optional, cast

from sasori.ai.ai_client_config import AIConfig

from sasori.ai.aws.anthropic.claude_client_config import (
    Claude35ClientConfig,
    Claude37ClientConfig,
    ClaudeClientConfig,
)
from sasori.ai.aws.bedrock_client import BedrockClient

import typer


class ClaudeClient(BedrockClient):
    def __init__(self, config: Optional[AIConfig] = None):
        if not config:
            config = Claude37ClientConfig()
        super().__init__(config)

    def _format_request_body(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.config = cast(ClaudeClientConfig, self.config)
        return {
            "anthropic_version": self.config.anthropic_version,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "messages": messages,
        }

    def _parse_response(self, raw: dict[str, Any]) -> str:
        try:
            content = raw["content"][0]["text"]
            return content
        except Exception as e:
            typer.secho(f"❌  Failed to parse Claude response: {e}", fg="red", err=True)
            raise
