"""Allas S3 fallback accessor for the pilot.

The .env OpenStack credentials both projects carry have expired (keystone
answers 401 Unauthorized for every request), so common/allas_utils'
swiftclient path is currently unusable. This module instead reads the S3
credentials rclone stores in ~/.config/rclone/rclone.conf (remote
`s3allas-project_2006782`, endpoint a3s.fi) and serves shard bytes through
boto3 -- same in-memory contract as keyframe_io.download_shard, nothing is
written to disk and no credential value is ever printed.

If the .env credentials are ever refreshed, common/allas_utils works again
and this fallback becomes unnecessary (common_pilot tries swift first).
"""
from __future__ import annotations

import configparser
import threading
from pathlib import Path

_LOCK = threading.Lock()
_CLIENT = None
_CONTAINER = "2006782_youtube_videos_keyframes"
_RCLONE_CONF = Path.home() / ".config" / "rclone" / "rclone.conf"


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        with _LOCK:
            if _CLIENT is None:
                import boto3
                cfg = configparser.ConfigParser()
                cfg.read(_RCLONE_CONF)
                sec = cfg["s3allas-project_2006782"]
                endpoint = sec["endpoint"]
                if not endpoint.startswith("http"):
                    endpoint = "https://" + endpoint
                _CLIENT = boto3.client(
                    "s3",
                    endpoint_url=endpoint,
                    aws_access_key_id=sec["access_key_id"],
                    aws_secret_access_key=sec["secret_access_key"],
                )
    return _CLIENT


def get_object_bytes(key: str, bucket: str = _CONTAINER) -> bytes | None:
    try:
        return _get_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as e:
        print(f"[allas-s3] failed to download {bucket}/{key}: {e}")
        return None


def object_exists(key: str, bucket: str = _CONTAINER) -> bool:
    try:
        _get_client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False
