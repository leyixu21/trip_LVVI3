"""Download Flickr images (z-size, 640px) listed in flickr_finland_state_lands.csv.

Saves to data/Flickr/images_z/<id>.jpg and writes a failures list.
Idempotent: skips files that already exist.

Usage: python download_flickr_images.py
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data" / "Flickr" / "flickr_finland_state_lands.csv"
OUT_DIR = ROOT / "data" / "Flickr" / "images_z"
FAILURES = OUT_DIR / "download_failures.json"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def fetch(row):
    pid, url = row
    dest = OUT_DIR / f"{pid}.jpg"
    if dest.exists() and dest.stat().st_size > 1000:
        return pid, True, "cached"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-download/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
                f.write(r.read())
            return pid, True, "ok"
        except Exception as e:
            time.sleep(2 * (attempt + 1))
            err = str(e)
    return pid, False, err


def main():
    df = pd.read_csv(CSV)
    df = df.drop_duplicates("id")
    rows = [(int(r.id), r.photo_url) for r in df.itertuples()
            if isinstance(r.photo_url, str) and r.photo_url.startswith("http")]
    print(f"{len(rows)} photos to fetch into {OUT_DIR}", flush=True)
    fails, done = [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch, r) for r in rows]
        for fut in as_completed(futs):
            pid, ok, msg = fut.result()
            done += 1
            if not ok:
                fails.append({"id": pid, "error": msg})
            if done % 500 == 0:
                rate = done / (time.time() - t0)
                print(f"{done}/{len(rows)} ({rate:.1f}/s), fails={len(fails)}", flush=True)
    FAILURES.write_text(json.dumps(fails, indent=1))
    print(f"DONE: {done - len(fails)} ok, {len(fails)} failed -> {FAILURES}", flush=True)


if __name__ == "__main__":
    main()
