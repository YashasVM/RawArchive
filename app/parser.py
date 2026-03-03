from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from app.models import CanonicalMessage


@dataclass
class ParseResult:
    messages: list[CanonicalMessage]
    warnings: list[str]
    participants: list[str]


def _slugify(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return token.strip("_") or "unknown"


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_conversations(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("messages"), list):
            return [payload]
        if isinstance(payload.get("conversations"), list):
            return [item for item in payload["conversations"] if isinstance(item, dict)]
        if isinstance(payload.get("inbox"), list):
            return [item for item in payload["inbox"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Unsupported Instagram export structure")


def parse_instagram_export(payload: Any, source_tag: str = "") -> ParseResult:
    warnings: list[str] = []
    messages: list[CanonicalMessage] = []
    participant_ids: set[str] = set()

    conversations = _iter_conversations(payload)
    if not conversations:
        raise ValueError("No conversations found in JSON payload")

    for index, conversation in enumerate(conversations):
        title = str(conversation.get("title") or f"conversation_{index}")
        raw_conversation_id = (
            conversation.get("conversation_id")
            or conversation.get("thread_path")
            or hashlib.sha1(f"{source_tag}:{title}:{index}".encode("utf-8")).hexdigest()[:12]
        )
        conversation_id = _slugify(str(raw_conversation_id))

        for participant in conversation.get("participants", []):
            if isinstance(participant, dict):
                name = str(participant.get("name") or "").strip()
                if name:
                    participant_ids.add(_slugify(name))

        for item in conversation.get("messages", []):
            if not isinstance(item, dict):
                warnings.append("non_object_message_skipped")
                continue

            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                warnings.append("non_text_messages_skipped")
                continue

            sender_display = str(item.get("sender_name") or item.get("sender") or "unknown")
            sender_id = _slugify(sender_display)
            timestamp_ms = _to_int(item.get("timestamp_ms") or item.get("timestamp") or item.get("created_at"))
            if timestamp_ms is None:
                warnings.append("timestamp_missing_skipped")
                continue

            participant_ids.add(sender_id)
            messages.append(
                CanonicalMessage(
                    conversation_id=conversation_id,
                    timestamp_ms=timestamp_ms,
                    sender_id=sender_id,
                    sender_display=sender_display,
                    text=content.strip(),
                )
            )

    messages.sort(key=lambda msg: (msg.conversation_id, msg.timestamp_ms))
    deduped_warnings = sorted(set(warnings))
    participants = sorted(participant_ids)
    return ParseResult(messages=messages, warnings=deduped_warnings, participants=participants)
