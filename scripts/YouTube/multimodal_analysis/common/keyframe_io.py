"""Shared helper for stages that consume keyframes.

Each video's keyframes live in exactly one Allas object,
`{video_id}.tar.gz`, containing `{video_id}/shots.json` plus that video's
`frame_*.jpg` files (WebDataset-style tar shard -- see
02_detect_shots_and_extract_keyframes.py, which builds and uploads it
directly, and 03_clip_dedup.py, which overwrites it in place with the
deduped subset).

Everything here operates on in-memory bytes -- download_shard/upload_shard
talk to Allas directly (common/allas_utils.download_bytes/upload_bytes),
and read_shard/build_shard_bytes read and write tar archives via
io.BytesIO. No stage ever writes a keyframe (shard or individual frame) to
local disk: CSC project scratch has a limited file-count/storage quota,
Allas is the purpose-built bulk store for this data, and a shard is only
ever a few MB -- decoding it fully in memory is cheap and avoids touching
the filesystem at all.
"""
from __future__ import annotations

import io
import json
import os
import tarfile

from PIL import Image

from . import allas_utils, config


def download_shard(conn, video_id: str) -> bytes | None:
    """Downloads {video_id}.tar.gz straight into memory. Returns None if no
    shard exists yet for this video (e.g. stage 1 hasn't run for it)."""
    return allas_utils.download_bytes(conn, config.ALLAS_KEYFRAME_CONTAINER, f"{video_id}.tar.gz")


def upload_shard(conn, video_id: str, data: bytes) -> bool:
    return allas_utils.upload_bytes(
        conn, config.ALLAS_KEYFRAME_CONTAINER, f"{video_id}.tar.gz", data, content_type="application/gzip"
    )


def read_shard(shard_bytes: bytes, video_id: str) -> tuple[list[dict], dict[str, bytes]]:
    """Reads shots.json and every frame's raw jpg bytes out of in-memory tar
    bytes in one pass. Returns (shot_records, {frame_name: jpg_bytes})."""
    frame_bytes: dict[str, bytes] = {}
    shot_records: list[dict] = []
    with tarfile.open(fileobj=io.BytesIO(shard_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            data = tar.extractfile(member).read()
            if name == "shots.json":
                shot_records = json.loads(data)
            elif name.endswith(".jpg"):
                frame_bytes[name] = data
    return shot_records, frame_bytes


def decode_images(frame_bytes: dict[str, bytes], frame_names: list[str]) -> dict[str, Image.Image]:
    """Decodes just the requested frames into PIL Images."""
    return {name: Image.open(io.BytesIO(frame_bytes[name])).convert("RGB") for name in frame_names if name in frame_bytes}


def build_shard_bytes(video_id: str, shot_records: list[dict], frame_bytes: dict[str, bytes]) -> bytes:
    """Builds the `{video_id}/shots.json` + `{video_id}/frame_*.jpg` tar.gz
    layout in memory and returns its bytes, ready to upload directly (used
    by stage 1 to produce the shard, and by stage 3 to persist the deduped
    subset)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        payload = json.dumps(shot_records).encode()
        info = tarfile.TarInfo(name=f"{video_id}/shots.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

        for rec in shot_records:
            data = frame_bytes[rec["frame_name"]]
            info = tarfile.TarInfo(name=f"{video_id}/{rec['frame_name']}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()
