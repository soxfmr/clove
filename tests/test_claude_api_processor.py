import asyncio
import unittest

from app.models.claude import MessagesAPIRequest
from app.processors.claude_ai.claude_api_processor import ClaudeAPIProcessor


class ClaudeAPIProcessorRequestPayloadTests(unittest.TestCase):
    def test_build_request_json_strips_unsupported_content_block_metadata(self) -> None:
        request = MessagesAPIRequest.model_validate(
            {
                "model": "claude-opus-4-6",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Use web search"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "Need to search first.",
                                "signature": "sig_123",
                                "start_timestamp": "2026-03-11T14:34:58.684Z",
                            },
                            {
                                "type": "text",
                                "text": "Searching now.",
                                "start_timestamp": "2026-03-11T14:34:59.684Z",
                            }
                        ],
                    },
                ],
            }
        )

        payload = asyncio.run(
            ClaudeAPIProcessor()._build_request_payload(request, original_request=None)
        )

        thinking_block = payload["messages"][1]["content"][0]

        self.assertEqual(thinking_block["type"], "thinking")
        self.assertEqual(thinking_block["thinking"], "Need to search first.")
        self.assertEqual(thinking_block["signature"], "sig_123")
        self.assertNotIn("start_timestamp", thinking_block)

        text_block = payload["messages"][1]["content"][1]

        self.assertEqual(text_block["type"], "text")
        self.assertEqual(text_block["text"], "Searching now.")
        self.assertNotIn("start_timestamp", text_block)

    def test_merge_raw_messages_preserves_signature_sensitive_blocks_verbatim(self) -> None:
        base_messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need to search first.", "signature": "sig_123"},
                    {"type": "text", "text": "Searching now."},
                ],
            },
        ]
        raw_messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Need to search first.",
                        "signature": "sig_123",
                        "start_timestamp": "2026-03-11T14:34:58.684Z",
                    },
                    {
                        "type": "text",
                        "text": "Searching now.",
                        "start_timestamp": "2026-03-11T14:34:59.684Z",
                    },
                ],
            },
        ]

        merged_messages = ClaudeAPIProcessor()._merge_raw_messages_for_preservation(
            base_messages, raw_messages
        )

        self.assertEqual(
            merged_messages[1]["content"][0],
            raw_messages[1]["content"][0],
        )
        self.assertEqual(
            merged_messages[1]["content"][1],
            base_messages[1]["content"][1],
        )

    def test_prepare_headers_preserves_client_anthropic_version_and_headers(self) -> None:
        class FakeRequest:
            headers = {
                "anthropic-version": "2025-02-19",
                "anthropic-beta": "interleaved-thinking-2025-05-14,adaptive-thinking-2026-01-28",
                "anthropic-dangerous-direct-browser-access": "true",
                "x-api-key": "should-not-pass-through",
            }

        headers = ClaudeAPIProcessor()._prepare_headers(
            access_token="oauth-token",
            request=MessagesAPIRequest.model_validate(
                {
                    "model": "claude-opus-4-6",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "hello"}],
                }
            ),
            original_request=FakeRequest(),
        )

        self.assertEqual(headers["Authorization"], "Bearer oauth-token")
        self.assertEqual(headers["anthropic-version"], "2025-02-19")
        self.assertEqual(
            headers["anthropic-dangerous-direct-browser-access"],
            "true",
        )
        self.assertIn("oauth-2025-04-20", headers["anthropic-beta"])
        self.assertIn("interleaved-thinking-2025-05-14", headers["anthropic-beta"])
        self.assertIn("adaptive-thinking-2026-01-28", headers["anthropic-beta"])
        self.assertNotIn("x-api-key", headers)


if __name__ == "__main__":
    unittest.main()
