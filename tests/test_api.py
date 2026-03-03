import json
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _sample_payload() -> dict:
    return {
        "title": "Team",
        "participants": [{"name": "Yash"}, {"name": "Riya"}],
        "messages": [
            {"sender_name": "Yash", "timestamp_ms": 1000, "content": "Hi"},
            {"sender_name": "Riya", "timestamp_ms": 1200, "content": "Hello"},
            {"sender_name": "Yash", "timestamp_ms": 1300, "content": "How are you?"},
            {"sender_name": "Riya", "timestamp_ms": 1400, "content": "I am good"},
            {"sender_name": "Yash", "timestamp_ms": 1500, "content": "Great"},
            {"sender_name": "Riya", "timestamp_ms": 1600, "content": "See you"},
        ],
    }


def _sample_payload_two() -> dict:
    return {
        "title": "Friends",
        "participants": [{"name": "Yash"}, {"name": "Nina"}],
        "messages": [
            {"sender_name": "Nina", "timestamp_ms": 1700, "content": "Movie tonight?"},
            {"sender_name": "Yash", "timestamp_ms": 1800, "content": "Yes, 8 pm works"},
        ],
    }


def test_upload_build_launch_and_register() -> None:
    upload = client.post(
        "/v1/datasets/instagram/upload",
        files=[
            ("files", ("chat_1.json", json.dumps(_sample_payload()), "application/json")),
            ("files", ("chat_2.json", json.dumps(_sample_payload_two()), "application/json")),
        ],
    )
    assert upload.status_code == 200
    upload_data = upload.json()
    dataset_id = upload_data["dataset_id"]
    assert "yash" in upload_data["participants"]
    assert "nina" in upload_data["participants"]

    build = client.post(
        f"/v1/datasets/{dataset_id}/build",
        json={
            "target_user_id": "yash",
            "context_turns": 4,
            "min_reply_chars": 2,
            "max_samples": 500,
            "val_ratio": 0.2,
        },
    )
    assert build.status_code == 200
    build_data = build.json()
    assert build_data["train_examples"] >= 1

    launch = client.get(f"/v1/colab/launch?bundle_id={build_data['bundle_id']}")
    assert launch.status_code == 200
    assert launch.json()["env"]["BUNDLE_ID"] == build_data["bundle_id"]

    train_script = client.get("/v1/colab/train-script")
    assert train_script.status_code == 200
    assert "def main()" in train_script.text

    parsed = urlparse(build_data["download_url"])
    token = parse_qs(parsed.query)["token"][0]
    download = client.get(f"/v1/bundles/{build_data['bundle_id']}/download?token={token}")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/zip")

    register = client.post(
        "/v1/models/register",
        json={
            "bundle_id": build_data["bundle_id"],
            "adapter_uri": "hf://my-user/my-adapter",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "metrics": {"val_loss": 1.5, "style_score": 0.81},
        },
    )
    assert register.status_code == 200
    assert register.json()["status"] == "registered"
