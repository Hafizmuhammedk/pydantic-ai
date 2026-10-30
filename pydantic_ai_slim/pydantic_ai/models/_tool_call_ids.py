"""Canonicalize repeated function-tool IDs only for adapters that reject them on replay."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import replace

from ..exceptions import UserError
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponsePart,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]
    ToolCallPart,
    ToolReturnPart,
)
from . import (
    _hashed_tool_call_id,  # pyright: ignore[reportPrivateUsage]
    _legacy_retry_as_sendable,  # pyright: ignore[reportPrivateUsage]
)


def deduplicate_tool_call_ids(
    messages: Sequence[ModelMessage], *, across_responses: bool = False, id_length: int = 32
) -> list[ModelMessage]:
    """Rewrite repeated IDs and their FIFO results without changing stored history.

    Most affected APIs require uniqueness within a response; Cohere requires it across
    the conversation. Assign in history order so extending a history never changes its
    already-rendered prefix. Native tool IDs belong to the server and are left alone.
    """
    taken: set[str] = set()
    pending: dict[str, deque[str]] = {}
    mapped: list[ModelMessage] = []
    changed = False
    for message in messages:
        if isinstance(message, ModelRequest):
            request_parts: list[ModelRequestPart] = []
            for part in message.parts:
                # Direct model calls render deprecated retries in the adapter, after this pass.
                # Lower tool-bound retries first so they consume the matching call's FIFO ID.
                if isinstance(part, RetryPromptPart) and part.tool_name is not None:  # pyright: ignore[reportDeprecated]
                    part = _legacy_retry_as_sendable(part)
                if isinstance(part, ToolReturnPart) and (ids := pending.get(part.tool_call_id)):
                    replacement = ids.popleft()
                    if replacement != part.tool_call_id:
                        part = replace(part, tool_call_id=replacement)
                request_parts.append(part)
            message_changed = any(new is not old for new, old in zip(request_parts, message.parts))
            mapped.append(replace(message, parts=request_parts) if message_changed else message)
        else:
            if not across_responses:
                taken.clear()
            response_ids: set[str] = set()
            response_parts: list[ModelResponsePart] = []
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    original = part.tool_call_id
                    if original not in response_ids:
                        # Reuse in a later response shadows any unanswered earlier call, matching
                        # message repair's pairing contract. Repeated calls in one response are FIFO.
                        pending[original] = deque()
                        response_ids.add(original)
                    replacement = original
                    attempt = 0
                    while replacement in taken:
                        if len(taken) >= 62**id_length:
                            raise UserError('Too many tool calls to assign distinct IDs in the required format.')
                        attempt += 1
                        replacement = _hashed_tool_call_id(f'{original}#{attempt}', id_length)
                    taken.add(replacement)
                    pending[original].append(replacement)
                    if replacement != original:
                        part = replace(part, tool_call_id=replacement)
                response_parts.append(part)
            message_changed = any(new is not old for new, old in zip(response_parts, message.parts))
            mapped.append(replace(message, parts=response_parts) if message_changed else message)
        changed |= message_changed
    return mapped if changed or not isinstance(messages, list) else messages
