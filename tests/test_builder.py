from app.dataset_builder import build_training_examples
from app.models import CanonicalMessage


def test_build_training_examples_creates_train_and_val() -> None:
    messages = [
        CanonicalMessage(conversation_id="c1", timestamp_ms=1000, sender_id="bob", sender_display="Bob", text="hello"),
        CanonicalMessage(conversation_id="c1", timestamp_ms=2000, sender_id="alice", sender_display="Alice", text="hey there"),
        CanonicalMessage(conversation_id="c1", timestamp_ms=3000, sender_id="bob", sender_display="Bob", text="how are you"),
        CanonicalMessage(conversation_id="c1", timestamp_ms=4000, sender_id="alice", sender_display="Alice", text="doing well"),
        CanonicalMessage(conversation_id="c2", timestamp_ms=5000, sender_id="bob", sender_display="Bob", text="new convo"),
        CanonicalMessage(conversation_id="c2", timestamp_ms=6000, sender_id="alice", sender_display="Alice", text="nice"),
    ]

    train, val, meta = build_training_examples(
        messages=messages,
        target_user_id="alice",
        context_turns=3,
        min_reply_chars=2,
        max_samples=100,
        val_ratio=0.2,
    )

    assert len(train) >= 1
    assert len(train) + len(val) == meta["total_examples"]
    assert meta["target_user_id"] == "alice"
