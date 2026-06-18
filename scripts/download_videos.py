"""Download Clash Royale gameplay videos with yt-dlp.

Usage:
    .venv/bin/python -m scripts.download_videos URL [URL ...] [--out data/youtube]

Requires yt-dlp (pip install yt-dlp). Downloads at <=1080p mp4 — the frames
are resized to 84x84 anyway, so higher resolutions only waste bandwidth.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def download(urls: list[str], out_dir: Path) -> list[str]:
    """Download each URL; returns the URLs that failed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for url in urls:
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--format",
            "bestvideo[height<=1080][ext=mp4]/best[height<=1080]",
            "--output",
            str(out_dir / "%(id)s.%(ext)s"),
            url,
        ]
        print(f"Downloading {url}")
        if subprocess.run(command).returncode != 0:
            failed.append(url)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="+", help="YouTube video URLs")
    parser.add_argument("--out", default="data/youtube", help="output directory")
    args = parser.parse_args()
    failed = download(args.urls, Path(args.out))
    if failed:
        raise SystemExit("Failed to download: " + ", ".join(failed))


if __name__ == "__main__":
    main()
