#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _root_inventory() -> dict:
    dirs = []
    files = []
    for entry in sorted(ROOT.iterdir(), key=lambda p: p.name.lower()):
        if entry.name in {".git", "__pycache__"}:
            continue
        if entry.is_dir():
            dirs.append(entry.name)
        else:
            files.append(entry.name)
    return {"dirs": dirs, "files": files}


def _large_files(threshold_mb: int = 100) -> list[dict]:
    findings = []
    for entry in ROOT.iterdir():
        if not entry.is_file():
            continue
        size_mb = entry.stat().st_size / (1024 * 1024)
        if size_mb >= threshold_mb:
            findings.append({"file": entry.name, "size_mb": round(size_mb, 1)})
    return sorted(findings, key=lambda x: x["size_mb"], reverse=True)


def _exists(path: str) -> bool:
    return (ROOT / path).exists()


def build_report() -> dict:
    inventory = _root_inventory()
    checks = {
        "new_platform_src": _exists("src/uris_platform"),
        "scene_templates": _exists("configs/scenes"),
        "legacy_app": _exists("legacy/legacy_video_reasoning_app.py"),
        "makefile": _exists("Makefile"),
        "streamlit_config": _exists(".streamlit/config.toml"),
        "tests_dir": _exists("tests"),
    }
    return {
        "repo_root": str(ROOT),
        "cwd": os.getcwd(),
        "checks": checks,
        "inventory_summary": {
            "root_dir_count": len(inventory["dirs"]),
            "root_file_count": len(inventory["files"]),
        },
        "large_root_files_100mb_plus": _large_files(),
        "notable_root_dirs": inventory["dirs"][:30],
    }


def main() -> int:
    report = build_report()
    print("URIS Project Doctor")
    print("=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    failed = [name for name, ok in report["checks"].items() if not ok]
    if failed:
        print("\nMissing expected paths:")
        for item in failed:
            print(f"- {item}")
        return 1
    print("\nAll core platform structure checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
