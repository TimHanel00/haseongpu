#!/usr/bin/env python3
"""Record local Git revisions and dirty state before a targeted source transfer."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True, stdout=subprocess.PIPE
    ).stdout.rstrip("\n")


def repository(root: Path) -> dict[str, object]:
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD"], check=True, stdout=subprocess.PIPE
    ).stdout
    return {
        "path": str(root.resolve()),
        "head": git(root, "rev-parse", "HEAD"),
        "branch": git(root, "branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dependency", action="append", default=[], type=Path)
    args = parser.parse_args()
    document = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": repository(args.source_root),
        "dependencies": [repository(path) for path in args.dependency],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
