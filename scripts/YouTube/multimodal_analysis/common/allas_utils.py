"""Shared Allas (CSC Swift object storage) helpers, used by every pipeline
stage that needs to pull source video / upload keyframes."""
from __future__ import annotations

import os

from dotenv import load_dotenv
import swiftclient

load_dotenv()


def connect_to_allas() -> swiftclient.Connection:
    os_options = {
        "user_domain_name": os.getenv("OS_USER_DOMAIN_NAME"),
        "project_domain_name": os.getenv("OS_USER_DOMAIN_NAME"),
        "project_name": os.getenv("OS_PROJECT_NAME"),
    }
    return swiftclient.Connection(
        authurl=os.getenv("OS_AUTH_URL"),
        user=os.getenv("OS_USERNAME"),
        key=os.getenv("OS_PASSWORD"),
        os_options=os_options,
        auth_version=os.getenv("OS_IDENTITY_API_VERSION"),
    )


def list_container_video_ids(conn: swiftclient.Connection, container: str) -> set[str]:
    """A single get_container() call is silently capped at Swift's default
    listing page size (10000) -- a container past that (ALLAS_VIDEO_CONTAINER
    already is, at 13000+) would otherwise come back truncated with no
    error, undercounting what's actually there. Page through with `marker`
    until a response comes back short of the page size."""
    video_ids = set()
    marker = None
    while True:
        _, objects = conn.get_container(container, marker=marker)
        if not objects:
            break
        video_ids.update(info["name"].split(".")[0] for info in objects)
        marker = objects[-1]["name"]
    return video_ids


def download_bytes(conn: swiftclient.Connection, container: str, object_name: str) -> bytes | None:
    try:
        return conn.get_object(container, object_name)[1]
    except Exception as e:
        print(f"[allas] swift download failed for {container}/{object_name} ({e}); trying S3 fallback")
    try:
        return _fallback_s3().get_object(Bucket=container, Key=object_name)["Body"].read()
    except Exception as e:
        print(f"[allas] failed to download {container}/{object_name}: {e}")
        return None


def upload_bytes(conn: swiftclient.Connection, container: str, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
    try:
        conn.put_object(container, object_name, contents=data, content_type=content_type)
        return True
    except Exception as e:
        print(f"[allas] swift upload failed for {container}/{object_name} ({e}); trying S3 fallback")
    try:
        _fallback_s3().put_object(Bucket=container, Key=object_name, Body=data, ContentType=content_type)
        return True
    except Exception as e:
        print(f"[allas] failed to upload {len(data)} bytes -> {container}/{object_name}: {e}")
        return False


_S3_CLIENT = None


def _fallback_s3():
    """boto3 client built from the rclone S3 credentials
    (~/.config/rclone/rclone.conf, remote s3allas-project_2006782). Used when
    the OpenStack/swift credentials in .env are expired (keystone 401)."""
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import configparser
        from pathlib import Path
        import boto3
        cfg = configparser.ConfigParser()
        cfg.read(Path.home() / ".config" / "rclone" / "rclone.conf")
        sec = cfg["s3allas-project_2006782"]
        endpoint = sec["endpoint"]
        if not endpoint.startswith("http"):
            endpoint = "https://" + endpoint
        _S3_CLIENT = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=sec["access_key_id"],
            aws_secret_access_key=sec["secret_access_key"],
        )
    return _S3_CLIENT


def download_object(conn: swiftclient.Connection, container: str, object_name: str, save_path: str) -> bool:
    """Downloads to a local file -- for objects a stage genuinely needs on
    disk (e.g. source .mp4s, which OpenCV/Whisper read by path). Keyframe
    shards don't need this: see common/keyframe_io.py, which reads/writes
    them with download_bytes/upload_bytes directly, never touching disk."""
    data = download_bytes(conn, container, object_name)
    if data is None:
        return False
    with open(save_path, "wb") as f:
        f.write(data)
    return True


def upload_file(conn: swiftclient.Connection, container: str, object_name: str, local_path: str, content_type: str = "application/octet-stream") -> bool:
    try:
        with open(local_path, "rb") as f:
            conn.put_object(container, object_name, contents=f, content_type=content_type)
        return True
    except Exception as e:
        print(f"[allas] failed to upload {local_path} -> {container}/{object_name}: {e}")
        return False
