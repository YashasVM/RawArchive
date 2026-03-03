from app.parser import parse_instagram_export


def test_parse_instagram_export_single_conversation() -> None:
    payload = {
        "title": "Friends",
        "participants": [{"name": "Alice"}, {"name": "Bob"}],
        "messages": [
            {"sender_name": "Alice", "timestamp_ms": 1000, "content": "Hey"},
            {"sender_name": "Bob", "timestamp_ms": 2000, "content": "Hi"},
            {"sender_name": "Bob", "timestamp_ms": 2200},
        ],
    }

    result = parse_instagram_export(payload)

    assert len(result.messages) == 2
    assert "alice" in result.participants
    assert "bob" in result.participants
    assert "non_text_messages_skipped" in result.warnings
