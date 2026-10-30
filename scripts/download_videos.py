"""Download full source lectures once per YouTube ID, with resumable downloads."""
import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from evaluation.common import (read_config, resolve, selected_splits, load_rows,
                               video_path, json_write)


def youtube_id(row):
    value = row.get("youtube_id")
    if value:
        return str(value)
    url = row.get("source_url") or ""
    parsed = urlparse(url)
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/")
    if parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        return parse_qs(parsed.query).get("v", [None])[0]
    return None


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def video_groups(rows):
    groups = {}
    for row in rows:
        key = youtube_id(row) or row.get("source_url") or row["source_video_id"]
        groups.setdefault(key, []).append(row)
    return groups


def download_group(config, key, rows):
    # One network download for each distinct lecture; hard-link aliases locally.
    needed_end = max(float(r["end_seconds"]) for r in rows)
    targets = sorted({video_path(config, r) for r in rows})
    for target in targets:
        if target.exists() and probe(target) + 0.1 < needed_end:
            raise ValueError(f"Existing video is shorter than required windows: {target}")
    existing = next((p for p in targets if p.exists()), None)
    if existing is None:
        existing = targets[0]
        existing.parent.mkdir(parents=True, exist_ok=True)
        row = rows[0]
        yid = youtube_id(row)
        url = f"https://www.youtube.com/watch?v={yid}" if yid else row.get("source_url")
        if not url:
            raise ValueError("Source has no downloadable URL")
        from yt_dlp import YoutubeDL
        options = {
            "format": config["videos"].get("format", "best"),
            "outtmpl": str(existing.with_suffix(".%(ext)s")),
            "merge_output_format": "mp4",
            "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
            "continuedl": True,
            "keepvideo": True,
            "overwrites": False,
            "noplaylist": True,
            "retries": 3,
            "fragment_retries": 3,
            "legacyserverconnect": config["videos"].get("legacy_server_connect", False),
        }
        cookies = config["videos"].get("cookies_file")
        with YoutubeDL(options) as downloader:
            if cookies:
                from yt_dlp.cookies import YoutubeDLCookieJar
                # Keep each worker's cookies in memory. Passing cookiefile makes
                # yt-dlp rewrite the shared file on close, racing other readers.
                jar = YoutubeDLCookieJar(str(resolve(config, cookies)))
                jar.load()
                jar.filename = None
                downloader.cookiejar = jar
            downloader.download([url])
        if not existing.exists() or probe(existing) + 0.1 < needed_end:
            raise ValueError("Downloaded video missing or shorter than its annotated windows")
    for target in targets:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.hardlink_to(existing)
            except OSError:
                shutil.copyfile(existing, target)
    return {"source": key, "status": "available", "source_video_ids": sorted({r["source_video_id"] for r in rows})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--dry-run", action="store_true", help="List source counts without network access")
    parser.add_argument("--legacy-server-connect", action="store_true",
                        help="Allow legacy TLS compatibility for servers with handshake failures")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.legacy_server_connect:
        config["videos"]["legacy_server_connect"] = True
    rows = [r for split in selected_splits(args.split) for r in load_rows(config, split)]
    groups = video_groups(rows)
    print(f"{len(rows)} questions; {len({r['source_video_id'] for r in rows})} source IDs; {len(groups)} distinct downloads")
    if args.dry_run:
        return
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        raise RuntimeError("Install ffmpeg and ffprobe before downloading")
    results = []
    with ThreadPoolExecutor(max_workers=int(config["videos"].get("workers", 2))) as pool:
        jobs = {pool.submit(download_group, config, key, rows): key for key, rows in groups.items()}
        for job in as_completed(jobs):
            try:
                result = job.result()
            except Exception as exc:
                result = {"source": jobs[job], "status": "failed", "error": str(exc)}
            results.append(result)
            print(result["source"], result["status"], result.get("error", ""))
    root = resolve(config, config["videos"]["base_dir"])
    root.mkdir(parents=True, exist_ok=True)
    json_write(root / f"download_{args.split}.json", results)
    if any(r["status"] == "failed" for r in results):
        raise SystemExit("Some lectures were unavailable; see the download manifest. Annotations were not removed.")


if __name__ == "__main__":
    main()
