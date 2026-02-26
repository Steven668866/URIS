#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from uris_platform.prompts.qwen_interaction_prompt import (  # type: ignore
        REQUIRED_JSON_FIELDS,
        build_qwen_interaction_prompt,
    )
except Exception:
    REQUIRED_JSON_FIELDS = [
        "intent",
        "user_goal",
        "observed_objects",
        "spatial_relations",
        "scene_summary",
        "recommendation_steps",
        "clarification_needed",
        "clarifying_question",
        "confidence",
        "evidence_basis",
        "limitations",
    ]

    def build_qwen_interaction_prompt(**kwargs: Any) -> dict[str, Any]:
        user_query = str(kwargs.get("user_query") or "")
        scene_summary = str(kwargs.get("scene_summary") or "")
        return {
            "schema_version": "uris-qwen-live-v1",
            "system_prompt": "You are URIS, a multimodal interaction simulation assistant.",
            "user_prompt": f"user_query: {user_query}\nscene_summary: {scene_summary}",
            "required_json_fields": REQUIRED_JSON_FIELDS,
        }


def _json_load_any(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno} is not a JSON object")
            rows.append(obj)
        return rows

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON array or JSONL file")
    rows = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{i}] is not a JSON object")
        rows.append(item)
    return rows


def _extract_target(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    # Preferred container format
    for key in ("target", "assistant_target"):
        cand = row.get(key)
        if isinstance(cand, dict):
            user_response = str(cand.get("user_response") or "").strip()
            analysis_json = cand.get("analysis_json") or {}
            if user_response and isinstance(analysis_json, dict):
                return user_response, dict(analysis_json)

    # Flat fallback format
    if isinstance(row.get("analysis_json"), dict):
        return str(row.get("user_response") or "").strip(), dict(row.get("analysis_json") or {})

    raise ValueError("Missing target/assistant_target with user_response + analysis_json")


def _normalize_analysis_json(analysis_json: dict[str, Any], *, user_query: str, scene_summary: str) -> dict[str, Any]:
    out = {
        "intent": "unknown",
        "user_goal": user_query,
        "observed_objects": [],
        "spatial_relations": [],
        "scene_summary": scene_summary,
        "recommendation_steps": [],
        "clarification_needed": False,
        "clarifying_question": "",
        "confidence": 0.5,
        "evidence_basis": [],
        "limitations": [],
    }
    out.update(analysis_json or {})
    missing = [k for k in REQUIRED_JSON_FIELDS if k not in out]
    if missing:
        raise ValueError(f"analysis_json missing required fields: {missing}")
    return out


def _coerce_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _to_conversations(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for m in messages:
        role = str(m.get("role") or "")
        content = str(m.get("content") or "")
        if role == "user":
            out.append({"from": "user", "value": content})
        elif role == "assistant":
            out.append({"from": "assistant", "value": content})
        else:
            out.append({"from": role, "value": content})
    return out


def _stable_bucket(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def _resolve_image_path(raw_path: str, *, images_root: Path | None, allow_missing: bool) -> str:
    path = Path(raw_path)
    if not path.is_absolute() and images_root is not None:
        path = (images_root / path).resolve()
    if not allow_missing and not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return str(path)


def _parse_source_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SOURCE_NAME=/path/to/file.jsonl")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Source name cannot be empty")
    return name, Path(raw_path).expanduser().resolve()


def _build_example(
    row: dict[str, Any],
    *,
    source_name: str,
    images_root: Path | None,
    allow_missing_images: bool,
    compact_context: bool,
    language: str,
) -> dict[str, Any]:
    user_query = str(row.get("user_query") or row.get("query") or row.get("instruction") or "").strip()
    if not user_query:
        raise ValueError("Missing user_query")

    scene_summary = str(row.get("scene_summary") or "").strip()
    image_raw = str(row.get("image") or row.get("image_path") or "").strip()
    if not image_raw:
        raise ValueError("Missing image/image_path")
    image_path = _resolve_image_path(image_raw, images_root=images_root, allow_missing=allow_missing_images)

    user_response, analysis_json = _extract_target(row)
    normalized_analysis = _normalize_analysis_json(
        analysis_json,
        user_query=user_query,
        scene_summary=scene_summary,
    )

    prompt = build_qwen_interaction_prompt(
        user_query=user_query,
        scene_summary=scene_summary,
        detections=_coerce_list(row.get("detections")),
        preferences=_coerce_list(row.get("preferences")),
        recent_turns=_coerce_list(row.get("recent_turns")),
        object_registry=_coerce_list(row.get("object_registry")),
        recent_scene_events=_coerce_list(row.get("recent_scene_events")),
        reference_resolution=_as_dict(row.get("reference_resolution")),
        compact_context=compact_context,
        language=language,
    )

    assistant_payload = {
        "user_response": user_response,
        "analysis_json": normalized_analysis,
    }
    assistant_text = (
        f"{user_response}\n\n```json\n"
        + json.dumps(assistant_payload, ensure_ascii=False, indent=2)
        + "\n```"
    )

    user_text = "<image>\n" + str(prompt.get("user_prompt") or "")
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    row_id = str(row.get("id") or row.get("sample_id") or "").strip()
    if not row_id:
        row_id = hashlib.sha1((source_name + "|" + image_path + "|" + user_query).encode("utf-8")).hexdigest()[:16]

    meta = {
        "id": row_id,
        "source": source_name,
        "task_type": str(row.get("task_type") or row.get("task") or "unknown"),
        "room_id": str(row.get("room_id") or row.get("scene_id") or ""),
        "split_key": str(row.get("split_key") or row.get("room_id") or row.get("scene_id") or image_path),
        "prompt_version": str(prompt.get("schema_version") or "uris-qwen-live-v1"),
        "reference_resolution": _as_dict(row.get("reference_resolution")),
        "json_required_fields": list(REQUIRED_JSON_FIELDS),
    }

    return {
        "id": row_id,
        "messages": messages,
        "conversations": _to_conversations(messages),  # compatibility for older LF configs
        "images": [image_path],
        "system": str(prompt.get("system_prompt") or ""),
        "meta": meta,
    }


def _split_examples(
    examples: list[dict[str, Any]],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_key_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        meta = ex.get("meta") or {}
        key = str(meta.get(split_key_field) or meta.get("split_key") or ex.get("id"))
        grouped[key].append(ex)

    # Deterministic group-level split to reduce scene leakage.
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    keys = sorted(grouped.keys())
    rnd = random.Random(seed)
    # Mix equal-hash clusters slightly to avoid pathological ordering in tiny datasets.
    rnd.shuffle(keys)
    keys.sort(key=lambda k: _stable_bucket(k))

    for key in keys:
        bucket = _stable_bucket(key)
        if bucket < test_ratio:
            test.extend(grouped[key])
        elif bucket < (test_ratio + val_ratio):
            val.extend(grouped[key])
        else:
            train.extend(grouped[key])

    return train, val, test


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare URIS VLM SFT dataset (Qwen2.5-VL) in LLaMA-Factory-compatible ShareGPT format."
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        type=_parse_source_arg,
        help="Repeatable. Format: source_name=/abs/or/relative/path.(json|jsonl)",
    )
    parser.add_argument("--images-root", type=Path, default=None, help="Optional root dir to resolve relative image paths")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for prepared dataset files")
    parser.add_argument("--dataset-name-prefix", default="uris_vlm", help="Prefix for dataset_info.json entries")
    parser.add_argument("--max-per-source", type=int, default=0, help="Cap samples per source (0 = no cap)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-key-field",
        default="split_key",
        help="meta field used for leak-resistant grouping (e.g., split_key / room_id)",
    )
    parser.add_argument("--language", default="zh", choices=["zh", "en"], help="Prompt language")
    parser.add_argument("--compact-context", action="store_true", help="Use compact prompt context when building training prompts")
    parser.add_argument("--allow-missing-images", action="store_true", help="Skip image existence check")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any malformed record (default: skip malformed records and continue)",
    )
    args = parser.parse_args()

    if not args.source:
        parser.error("At least one --source is required")

    if args.images_root is not None:
        args.images_root = args.images_root.expanduser().resolve()
    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    prepared: list[dict[str, Any]] = []
    stats = {
        "loaded_per_source": Counter(),
        "prepared_per_source": Counter(),
        "skipped_per_source": Counter(),
        "skip_reasons": Counter(),
        "task_types": Counter(),
    }

    for source_name, path in args.source:
        rows = _json_load_any(path)
        stats["loaded_per_source"][source_name] += len(rows)
        if args.max_per_source and len(rows) > args.max_per_source:
            rows = list(rows)
            rng.shuffle(rows)
            rows = rows[: args.max_per_source]

        for row in rows:
            try:
                ex = _build_example(
                    row,
                    source_name=source_name,
                    images_root=args.images_root,
                    allow_missing_images=args.allow_missing_images,
                    compact_context=bool(args.compact_context),
                    language=args.language,
                )
            except Exception as exc:
                stats["skipped_per_source"][source_name] += 1
                stats["skip_reasons"][type(exc).__name__] += 1
                if args.strict:
                    raise
                continue

            prepared.append(ex)
            stats["prepared_per_source"][source_name] += 1
            stats["task_types"][str((ex.get("meta") or {}).get("task_type") or "unknown")] += 1

    if not prepared:
        print("No valid samples were prepared.")
        return 1

    train, val, test = _split_examples(
        prepared,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_key_field=args.split_key_field,
    )

    train_name = f"{args.dataset_name_prefix}_train"
    val_name = f"{args.dataset_name_prefix}_val"
    test_name = f"{args.dataset_name_prefix}_test"

    train_file = "train_sharegpt.json"
    val_file = "val_sharegpt.json"
    test_file = "test_sharegpt.json"

    _write_json(out_dir / train_file, train)
    _write_json(out_dir / val_file, val)
    _write_json(out_dir / test_file, test)

    dataset_info = {
        train_name: {
            "file_name": train_file,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images", "system": "system"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
        val_name: {
            "file_name": val_file,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images", "system": "system"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
        test_name: {
            "file_name": test_file,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images", "system": "system"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
    }
    _write_json(out_dir / "dataset_info.json", dataset_info)

    summary = {
        "out_dir": str(out_dir),
        "dataset_names": {"train": train_name, "val": val_name, "test": test_name},
        "counts": {"all": len(prepared), "train": len(train), "val": len(val), "test": len(test)},
        "source_files": {name: str(path) for name, path in args.source},
        "options": {
            "images_root": str(args.images_root) if args.images_root else None,
            "max_per_source": args.max_per_source,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "split_key_field": args.split_key_field,
            "language": args.language,
            "compact_context": bool(args.compact_context),
            "allow_missing_images": bool(args.allow_missing_images),
            "seed": args.seed,
        },
        "stats": {
            "loaded_per_source": dict(stats["loaded_per_source"]),
            "prepared_per_source": dict(stats["prepared_per_source"]),
            "skipped_per_source": dict(stats["skipped_per_source"]),
            "skip_reasons": dict(stats["skip_reasons"]),
            "task_types": dict(stats["task_types"]),
        },
        "required_json_fields": list(REQUIRED_JSON_FIELDS),
    }
    _write_json(out_dir / "prepare_summary.json", summary)

    print("URIS VLM dataset prepared successfully")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
