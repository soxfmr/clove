import json
import copy
from app.core.http_client import (
    Response,
    AsyncSession,
    create_session,
)
from datetime import datetime, timedelta, UTC
from typing import Any, Dict
from loguru import logger
from fastapi.responses import StreamingResponse

from app.models.claude import MessagesAPIRequest, TextContent
from app.processors.base import BaseProcessor
from app.processors.claude_ai import ClaudeAIContext
from app.services.account import account_manager
from app.services.cache import cache_service
from app.core.exceptions import (
    ClaudeHttpError,
    ClaudeRateLimitedError,
    InvalidModelNameError,
    NoAccountsAvailableError,
    OAuthAuthenticationNotAllowedError,
)
from app.core.config import settings


class ClaudeAPIProcessor(BaseProcessor):
    """Processor that calls Claude Messages API directly using OAuth authentication."""

    _ALLOWED_TEXT_BLOCK_KEYS = frozenset(
        {"type", "text", "citations", "cache_control"}
    )
    _ALLOWED_IMAGE_BLOCK_KEYS = frozenset({"type", "source", "cache_control"})
    _ALLOWED_THINKING_BLOCK_KEYS = frozenset(
        {"type", "thinking", "signature", "cache_control"}
    )
    _ALLOWED_REDACTED_THINKING_BLOCK_KEYS = frozenset(
        {"type", "data", "cache_control"}
    )
    _ALLOWED_TOOL_USE_BLOCK_KEYS = frozenset(
        {"type", "id", "name", "input", "cache_control"}
    )
    _ALLOWED_TOOL_RESULT_BLOCK_KEYS = frozenset(
        {"type", "tool_use_id", "content", "is_error", "cache_control"}
    )
    _ALLOWED_IMAGE_SOURCE_KEYS = {
        "base64": frozenset({"type", "media_type", "data"}),
        "url": frozenset({"type", "url"}),
        "file": frozenset({"type", "file_uuid"}),
    }
    _ALLOWED_WEB_SEARCH_RESULT_KEYS = frozenset(
        {"type", "title", "url", "encrypted_content", "page_age"}
    )

    def __init__(self):
        self.messages_api_url = (
            settings.claude_api_baseurl.encoded_string().rstrip("/") + "/v1/messages"
        )

    async def _request_messages_api(
        self, session: AsyncSession, request_json: str, headers: Dict[str, str]
    ) -> Response:
        """Make HTTP request with retry mechanism for curl_cffi exceptions."""
        response: Response = await session.request(
            "POST",
            self.messages_api_url,
            data=request_json,
            headers=headers,
            stream=True,
        )
        return response

    async def process(self, context: ClaudeAIContext) -> ClaudeAIContext:
        """
        Process Claude API request using OAuth authentication.

        Requires:
            - messages_api_request in context

        Produces:
            - response in context (StreamingResponse)
        """
        if context.response:
            logger.debug("Skipping ClaudeAPIProcessor due to existing response")
            return context

        if not context.messages_api_request:
            logger.warning(
                "Skipping ClaudeAPIProcessor due to missing messages_api_request"
            )
            return context

        self._insert_system_message(context)

        try:
            # First try to get account from cache service
            cached_account_id, checkpoints = cache_service.process_messages(
                context.messages_api_request.model,
                context.messages_api_request.messages,
                context.messages_api_request.system,
            )

            account = None
            if cached_account_id:
                account = await account_manager.get_account_by_id(cached_account_id)
                if account:
                    logger.info(f"Using cached account: {cached_account_id[:8]}...")

            # If no cached account or account not available, get a new one
            if not account:
                account = await account_manager.get_account_for_oauth(
                    is_max=True
                    if (context.messages_api_request.model in settings.max_models)
                    else None
                )

            with account:
                request_payload = await self._build_request_payload(
                    context.messages_api_request,
                    context.original_request,
                )
                request_json = json.dumps(request_payload)
                headers = self._prepare_headers(
                    account.oauth_token.access_token,
                    context.messages_api_request,
                    context.original_request,
                )

                session = create_session(
                    proxy=settings.proxy_url,
                    timeout=settings.request_timeout,
                    impersonate="chrome",
                    follow_redirects=False,
                )

                response = await self._request_messages_api(
                    session, request_json, headers
                )

                error_data = None
                resets_at = response.headers.get("anthropic-ratelimit-unified-reset")
                if resets_at:
                    try:
                        resets_at = int(resets_at)
                        account.resets_at = datetime.fromtimestamp(resets_at, tz=UTC)
                    except ValueError:
                        logger.error(
                            f"Invalid resets_at format from Claude API: {resets_at}"
                        )
                        account.resets_at = None

                # Handle rate limiting
                if response.status_code == 429:
                    next_hour = datetime.now(UTC).replace(
                        minute=0, second=0, microsecond=0
                    ) + timedelta(hours=1)
                    raise ClaudeRateLimitedError(
                        resets_at=account.resets_at or next_hour
                    )

                if response.status_code >= 400:
                    error_data = await response.json()

                if response.status_code >= 400:
                    if (
                        response.status_code == 400
                        and error_data.get("error", {}).get("message")
                        == "system: Invalid model name"
                    ):
                        raise InvalidModelNameError(context.messages_api_request.model)

                    if (
                        response.status_code == 401
                        and error_data.get("error", {}).get("message")
                        == "OAuth authentication is currently not allowed for this organization."
                    ):
                        raise OAuthAuthenticationNotAllowedError()

                    logger.error(
                        f"Claude API error: {response.status_code} - {error_data}"
                    )
                    raise ClaudeHttpError(
                        url=self.messages_api_url,
                        status_code=response.status_code,
                        error_type=error_data.get("error", {}).get("type", "unknown"),
                        error_message=error_data.get("error", {}).get(
                            "message", "Unknown error"
                        ),
                    )

                async def stream_response():
                    async for chunk in response.aiter_bytes():
                        yield chunk

                    await session.close()

                filtered_headers = {}
                for key, value in response.headers.items():
                    if key.lower() in ["content-encoding", "content-length"]:
                        logger.debug(f"Filtering out header: {key}: {value}")
                        continue
                    filtered_headers[key] = value

                context.response = StreamingResponse(
                    stream_response(),
                    status_code=response.status_code,
                    headers=filtered_headers,
                )

                # Stop pipeline on success
                context.metadata["stop_pipeline"] = True
                logger.info("Successfully processed request via Claude API")

                # Store checkpoints in cache service after successful request
                if checkpoints and account:
                    cache_service.add_checkpoints(
                        checkpoints, account.organization_uuid
                    )

        except (NoAccountsAvailableError, InvalidModelNameError):
            logger.debug("No accounts available for Claude API, continuing pipeline")

        return context

    async def _build_request_payload(
        self, request: MessagesAPIRequest, original_request=None
    ) -> Dict[str, Any]:
        """Build a Claude API payload with selective raw-message preservation."""
        payload = request.model_dump(exclude_none=True, by_alias=True)
        removed_fields = self._sanitize_content_blocks(payload.get("messages", []))

        system = payload.get("system")
        if isinstance(system, list):
            removed_fields.extend(self._sanitize_blocks(system, "system"))

        if removed_fields:
            logger.debug(
                "Stripped unsupported content metadata from Claude API payload: "
                + ", ".join(removed_fields)
            )

        raw_payload = await self._load_original_request_payload(original_request)
        raw_messages = raw_payload.get("messages") if isinstance(raw_payload, dict) else None
        base_messages = payload.get("messages")
        if isinstance(base_messages, list) and isinstance(raw_messages, list):
            merged_messages = self._merge_raw_messages_for_preservation(
                base_messages, raw_messages
            )
            if merged_messages != base_messages:
                logger.debug(
                    "Preserved raw assistant signature-sensitive blocks from original request payload"
                )
                payload["messages"] = merged_messages

        return payload

    async def _load_original_request_payload(self, original_request) -> Dict[str, Any] | None:
        """Load and parse the original request JSON body if available."""
        if not original_request:
            return None

        try:
            body = await original_request.body()
        except Exception as exc:
            logger.debug(f"Failed to read original request body: {type(exc).__name__}: {exc}")
            return None

        if not body:
            return None

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.debug(f"Failed to parse original request JSON: {exc}")
            return None

        return payload if isinstance(payload, dict) else None

    def _sanitize_content_blocks(self, messages: list[dict]) -> list[str]:
        """Remove transport metadata that Claude API rejects on content blocks."""
        removed_fields: list[str] = []

        for message_index, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue

            removed_fields.extend(
                self._sanitize_blocks(content, f"messages[{message_index}].content")
            )

        return removed_fields

    def _merge_raw_messages_for_preservation(
        self, base_messages: list[dict], raw_messages: list[dict]
    ) -> list[dict]:
        """Preserve raw assistant blocks that are sensitive to signature validation."""
        if len(base_messages) != len(raw_messages):
            if any(self._message_needs_raw_preservation(message) for message in raw_messages):
                logger.debug(
                    "Using raw messages for preservation because sanitized and raw message counts differ"
                )
                return copy.deepcopy(raw_messages)
            return base_messages

        merged_messages = copy.deepcopy(base_messages)

        for index, raw_message in enumerate(raw_messages):
            if not self._message_needs_raw_preservation(raw_message):
                continue

            merged_messages[index] = self._merge_single_message_for_preservation(
                base_messages[index], raw_message
            )

        return merged_messages

    def _merge_single_message_for_preservation(
        self, base_message: dict, raw_message: dict
    ) -> dict:
        """Merge a sanitized message with raw signature-sensitive blocks preserved verbatim."""
        if not isinstance(base_message, dict) or not isinstance(raw_message, dict):
            return copy.deepcopy(raw_message)

        base_content = base_message.get("content")
        raw_content = raw_message.get("content")
        if not isinstance(base_content, list) or not isinstance(raw_content, list):
            return copy.deepcopy(raw_message)

        if len(base_content) != len(raw_content):
            logger.debug(
                "Using raw assistant message for preservation because content block counts differ"
            )
            return copy.deepcopy(raw_message)

        merged_message = copy.deepcopy(base_message)
        merged_content = []
        for block_index, raw_block in enumerate(raw_content):
            if self._block_needs_raw_preservation(raw_block):
                merged_content.append(copy.deepcopy(raw_block))
            else:
                merged_content.append(copy.deepcopy(base_content[block_index]))

        merged_message["content"] = merged_content
        return merged_message

    def _message_needs_raw_preservation(self, message: Any) -> bool:
        """Determine whether a message contains blocks that must stay byte-faithful."""
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return False

        content = message.get("content")
        if not isinstance(content, list):
            return False

        return any(self._block_needs_raw_preservation(block) for block in content)

    def _block_needs_raw_preservation(self, block: Any) -> bool:
        """Detect blocks whose raw structure should be preserved for upstream validation."""
        if not isinstance(block, dict):
            return False

        block_type = block.get("type")
        return (
            block_type in {"thinking", "redacted_thinking"}
            or "signature" in block
        )

    def _sanitize_blocks(self, blocks: list[dict], location: str) -> list[str]:
        """Sanitize a list of content blocks in-place and report stripped keys."""
        removed_fields: list[str] = []

        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            allowed_keys = self._allowed_keys_for_block(block_type)
            if not allowed_keys:
                continue

            extra_keys = sorted(set(block) - allowed_keys)
            if extra_keys:
                for key in extra_keys:
                    block.pop(key, None)
                removed_fields.append(f"{location}[{block_index}]={extra_keys}")

            if block_type == "image":
                source = block.get("source")
                if isinstance(source, dict):
                    extra_source_keys = self._sanitize_image_source(
                        source, f"{location}[{block_index}].source"
                    )
                    if extra_source_keys:
                        removed_fields.append(
                            f"{location}[{block_index}].source={extra_source_keys}"
                        )
            elif block_type == "tool_result":
                tool_content = block.get("content")
                if isinstance(tool_content, list):
                    removed_fields.extend(
                        self._sanitize_blocks(
                            tool_content, f"{location}[{block_index}].content"
                        )
                    )
            elif block_type == "web_search_tool_result":
                results = block.get("content")
                if isinstance(results, list):
                    removed_fields.extend(
                        self._sanitize_web_search_results(
                            results, f"{location}[{block_index}].content"
                        )
                    )

        return removed_fields

    def _allowed_keys_for_block(self, block_type: str | None) -> frozenset[str] | None:
        if block_type == "text":
            return self._ALLOWED_TEXT_BLOCK_KEYS
        if block_type == "image":
            return self._ALLOWED_IMAGE_BLOCK_KEYS
        if block_type == "thinking":
            return self._ALLOWED_THINKING_BLOCK_KEYS
        if block_type == "redacted_thinking":
            return self._ALLOWED_REDACTED_THINKING_BLOCK_KEYS
        if block_type in {"tool_use", "server_tool_use"}:
            return self._ALLOWED_TOOL_USE_BLOCK_KEYS
        if block_type in {"tool_result", "web_search_tool_result"}:
            return self._ALLOWED_TOOL_RESULT_BLOCK_KEYS
        return None

    def _sanitize_image_source(self, source: dict, location: str) -> list[str]:
        source_type = source.get("type")
        allowed_keys = self._ALLOWED_IMAGE_SOURCE_KEYS.get(source_type)
        if not allowed_keys:
            return []

        extra_keys = sorted(set(source) - allowed_keys)
        for key in extra_keys:
            source.pop(key, None)

        return extra_keys

    def _sanitize_web_search_results(
        self, results: list[dict], location: str
    ) -> list[str]:
        removed_fields: list[str] = []

        for result_index, result in enumerate(results):
            if not isinstance(result, dict):
                continue

            extra_keys = sorted(set(result) - self._ALLOWED_WEB_SEARCH_RESULT_KEYS)
            if not extra_keys:
                continue

            for key in extra_keys:
                result.pop(key, None)

            removed_fields.append(f"{location}[{result_index}]={extra_keys}")

        return removed_fields

    def _insert_system_message(self, context: ClaudeAIContext) -> None:
        """Insert system message into the request."""

        request = context.messages_api_request

        # Handle system field
        system_message_text = (
            "You are Claude Code, Anthropic's official CLI for Claude."
        )
        system_message = TextContent(type="text", text=system_message_text)

        if isinstance(request.system, str) and request.system:
            request.system = [
                system_message,
                TextContent(type="text", text=request.system),
            ]
        elif isinstance(request.system, list) and request.system:
            if request.system[0].text == system_message_text:
                logger.debug("System message already exists, skipping injection.")
            else:
                request.system = [system_message] + request.system
        else:
            request.system = [system_message]

    def _prepare_headers(
        self,
        access_token: str,
        request: MessagesAPIRequest,
        original_request=None,
    ) -> Dict[str, str]:
        """Prepare headers for Claude API request.

        Keep relay-owned auth/content headers authoritative while preserving
        client-provided anthropic headers needed for feature/version compatibility.
        """
        beta_features = ["oauth-2025-04-20"]
        passthrough_headers: Dict[str, str] = {}

        if original_request:
            for header_name, header_value in original_request.headers.items():
                header_name_lower = header_name.lower()
                if header_name_lower.startswith("anthropic-") and header_name_lower not in {
                    "anthropic-beta",
                    "anthropic-version",
                }:
                    passthrough_headers[header_name] = header_value

            client_beta = original_request.headers.get("anthropic-beta", "")
            if client_beta:
                for beta in client_beta.split(","):
                    beta = beta.strip()
                    if beta and beta not in beta_features:
                        beta_features.append(beta)

        headers = {
            "Authorization": f"Bearer {access_token}",
            "anthropic-beta": ",".join(beta_features),
            "anthropic-version": (
                original_request.headers.get("anthropic-version", "2023-06-01")
                if original_request
                else "2023-06-01"
            ),
            "Content-Type": "application/json",
        }
        headers.update(passthrough_headers)
        return headers
