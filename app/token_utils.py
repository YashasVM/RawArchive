from __future__ import annotations

import hmac
import time
from hashlib import sha256


def create_download_token(bundle_id: str, secret: str, ttl_seconds: int = 3600) -> str:
    expires_at = int(time.time()) + ttl_seconds
    payload = f"{bundle_id}:{expires_at}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), payload, sha256).hexdigest()
    return f"{expires_at}.{signature}"


def verify_download_token(bundle_id: str, token: str, secret: str) -> bool:
    try:
        expires_str, signature = token.split(".", 1)
        expires_at = int(expires_str)
    except (ValueError, AttributeError):
        return False

    if expires_at < int(time.time()):
        return False

    expected = hmac.new(
        secret.encode("utf-8"),
        f"{bundle_id}:{expires_at}".encode("utf-8"),
        sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
