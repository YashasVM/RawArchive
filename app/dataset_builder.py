from __future__ import annotations

import random
from collections import defaultdict, deque
from statistics import mean

from app.models import CanonicalMessage, TrainingExample


def compute_dataset_stats(messages: list[CanonicalMessage], target_user_id: str | None = None) -> dict:
    conversation_ids = {msg.conversation_id for msg in messages}
    target_messages = [msg for msg in messages if target_user_id and msg.sender_id == target_user_id]

    gaps: list[float] = []
    grouped: dict[str, list[CanonicalMessage]] = defaultdict(list)
    for message in messages:
        grouped[message.conversation_id].append(message)

    for conversation_messages in grouped.values():
        conversation_messages.sort(key=lambda msg: msg.timestamp_ms)
        for previous, current in zip(conversation_messages, conversation_messages[1:]):
            delta = (current.timestamp_ms - previous.timestamp_ms) / 1000.0
            if delta >= 0:
                gaps.append(delta)

    return {
        "messages_total": len(messages),
        "target_user_messages": len(target_messages),
        "avg_reply_gap_sec": round(mean(gaps), 2) if gaps else None,
        "conversations_total": len(conversation_ids),
    }


def _format_context(context: list[CanonicalMessage], target_user_id: str) -> str:
    lines: list[str] = []
    for item in context:
        speaker = "TARGET" if item.sender_id == target_user_id else item.sender_display
        lines.append(f"{speaker}: {item.text}")
    context_text = "\n".join(lines)
    return (
        "You are generating the next message in the target user's texting style.\n"
        "Only produce one realistic next reply from the target user.\n"
        f"Context:\n{context_text}\n"
        "Reply:"
    )


def _balanced_sample(examples: list[TrainingExample], max_samples: int) -> list[TrainingExample]:
    by_conversation: dict[str, deque[TrainingExample]] = defaultdict(deque)
    for example in examples:
        by_conversation[example.conversation_id].append(example)

    sampled: list[TrainingExample] = []
    buckets = list(by_conversation.values())
    while buckets and len(sampled) < max_samples:
        next_buckets: list[deque[TrainingExample]] = []
        for bucket in buckets:
            if bucket and len(sampled) < max_samples:
                sampled.append(bucket.popleft())
            if bucket:
                next_buckets.append(bucket)
        buckets = next_buckets
    return sampled


def build_training_examples(
    messages: list[CanonicalMessage],
    target_user_id: str,
    context_turns: int,
    min_reply_chars: int,
    max_samples: int,
    val_ratio: float,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample], dict]:
    by_conversation: dict[str, list[CanonicalMessage]] = defaultdict(list)
    for message in messages:
        by_conversation[message.conversation_id].append(message)

    examples: list[TrainingExample] = []
    dedup_targets: set[str] = set()

    for conversation_id, conversation_messages in by_conversation.items():
        conversation_messages.sort(key=lambda msg: msg.timestamp_ms)
        for index, message in enumerate(conversation_messages):
            if message.sender_id != target_user_id:
                continue
            if len(message.text.strip()) < min_reply_chars:
                continue

            context_slice = conversation_messages[max(0, index - context_turns) : index]
            if not context_slice:
                continue

            normalized_target = " ".join(message.text.lower().split())
            if normalized_target in dedup_targets:
                continue
            dedup_targets.add(normalized_target)

            examples.append(
                TrainingExample(
                    conversation_id=conversation_id,
                    input_text=_format_context(context_slice, target_user_id),
                    target_text=message.text.strip(),
                )
            )

    balanced = _balanced_sample(examples, max_samples=max_samples)
    rng = random.Random(seed)
    rng.shuffle(balanced)

    val_count = max(1, int(len(balanced) * val_ratio)) if len(balanced) > 1 else 0
    val_examples = balanced[:val_count]
    train_examples = balanced[val_count:]

    meta = {
        "target_user_id": target_user_id,
        "total_examples": len(balanced),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "context_turns": context_turns,
        "min_reply_chars": min_reply_chars,
        "max_samples": max_samples,
        "val_ratio": val_ratio,
    }
    return train_examples, val_examples, meta
