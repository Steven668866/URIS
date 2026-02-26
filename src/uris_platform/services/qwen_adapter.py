from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from uris_platform.prompts.qwen_interaction_prompt import build_qwen_interaction_prompt


JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
GENERIC_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _default_analysis_json(
    *,
    user_goal: str = "",
    confidence: float = 0.4,
    evidence_basis: list[str] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "intent": "unknown",
        "user_goal": user_goal,
        "observed_objects": [],
        "spatial_relations": [],
        "scene_summary": "",
        "recommendation_steps": [],
        "predicted_next_action": "",
        "proactive_suggestion": "",
        "clarification_needed": False,
        "clarifying_question": "",
        "confidence": confidence,
        "evidence_basis": evidence_basis or [],
        "limitations": limitations or [],
    }


def parse_qwen_structured_response(text: str) -> dict[str, Any]:
    candidate = None
    match = JSON_FENCE_RE.search(text or "")
    if match:
        candidate = match.group(1)
    else:
        generic = GENERIC_JSON_RE.search(text or "")
        if generic:
            candidate = generic.group(1)

    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "user_response" in parsed and "analysis_json" in parsed:
                analysis = parsed.get("analysis_json")
                if not isinstance(analysis, dict):
                    analysis = _default_analysis_json()
                merged_analysis = _default_analysis_json()
                merged_analysis.update(analysis)
                return {
                    "user_response": str(parsed.get("user_response", "")).strip() or str(text).strip(),
                    "analysis_json": merged_analysis,
                    "json_valid": True,
                    "raw_text": text,
                    "parse_error": None,
                }
        except Exception as exc:
            parse_error = str(exc)
        else:
            parse_error = "Invalid schema"
    else:
        parse_error = "No JSON block found"

    fallback = _default_analysis_json(
        confidence=0.35,
        evidence_basis=["natural_language_only"],
        limitations=["No valid structured JSON was parsed from model output."],
    )
    return {
        "user_response": (text or "").strip(),
        "analysis_json": fallback,
        "json_valid": False,
        "raw_text": text,
        "parse_error": parse_error,
    }


def build_fallback_qwen_response(*, user_query: str, detection_summary: str) -> dict[str, Any]:
    user_response = (
        f"我目前使用降级交互模式。基于检测摘要：{detection_summary}。"
        f"针对你的问题“{user_query}”，我可以先给出场景级建议；如果需要更细节，请重新拍摄更清晰画面。"
    )
    return {
        "user_response": user_response,
        "analysis_json": {
            **_default_analysis_json(
                user_goal=user_query,
                confidence=0.45,
                evidence_basis=["yolo_detection_summary"],
                limitations=["Qwen model unavailable or response parsing failed."],
            ),
            "scene_summary": detection_summary,
            "recommendation_steps": [
                "根据检测结果确认关键物体是否在画面内。",
                "如需精细判断，调整摄像头角度或距离后重试。",
            ],
        },
        "json_valid": False,
        "raw_text": user_response,
        "parse_error": "fallback_mode",
    }


def build_reference_clarification_response(
    *,
    user_query: str,
    scene_summary: str,
    reference_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    rr = dict(reference_resolution or {})
    clarifying_question = str(
        rr.get("clarifying_question")
        or "需要澄清：你指的是哪一个对象？请说明位置（如左边/右边）或类别。"
    )
    candidate_ids = list(rr.get("candidates") or [])
    candidate_count = int(rr.get("candidate_count") or len(candidate_ids) or 0)
    user_response = (
        "我可以继续给出建议，但当前指代对象不够明确。"
        f"{clarifying_question}"
    )
    analysis = _default_analysis_json(
        user_goal=user_query,
        confidence=0.42,
        evidence_basis=["reference_resolution", "yolo_detection_summary"],
        limitations=["Reference is ambiguous across multiple visible candidates."],
    )
    analysis.update(
        {
            "intent": "clarify_reference",
            "scene_summary": scene_summary,
            "clarification_needed": True,
            "clarifying_question": clarifying_question,
            "spatial_relations": [f"candidate_ids={candidate_ids}"] if candidate_ids else [],
            "recommendation_steps": [
                "先澄清目标对象（例如左边/右边或更具体属性）。",
                "确认目标后再给出针对性的整理或交互建议。",
            ],
        }
    )
    if candidate_count > 0:
        analysis["limitations"].append(f"Detected {candidate_count} candidate objects matching the reference.")
    return {
        "user_response": user_response,
        "analysis_json": analysis,
        "json_valid": True,
        "raw_text": user_response,
        "parse_error": None,
    }


@dataclass
class QwenAdapterStatus:
    available: bool
    mode: str
    reason: str | None = None


class QwenLiveAdapter:
    """Lightweight runtime wrapper. Uses fallback mode if dependencies/models are unavailable."""

    def __init__(self, *, adapter_path: str | None = None) -> None:
        self.adapter_path = adapter_path
        self._status = self._detect_status()
        self._response_cache: dict[str, dict[str, Any]] = {}
        self._cache_order: list[str] = []
        self._cache_limit = 64
        self._cache_stats = {"hits": 0, "misses": 0}

    def _detect_status(self) -> QwenAdapterStatus:
        try:
            import transformers  # noqa: F401
            import peft  # noqa: F401

            return QwenAdapterStatus(available=True, mode="lazy")
        except Exception as exc:  # pragma: no cover - environment specific
            return QwenAdapterStatus(available=False, mode="fallback", reason=str(exc))

    @property
    def status(self) -> QwenAdapterStatus:
        return self._status

    @property
    def cache_stats(self) -> dict[str, int]:
        return dict(self._cache_stats)

    def _build_cache_key(
        self,
        *,
        user_query: str,
        scene_summary: str,
        detections: Sequence[dict[str, Any]],
        preferences: Sequence[str],
        recent_turns: Sequence[dict[str, Any]],
        object_registry: Sequence[dict[str, Any]] | None,
        recent_scene_events: Sequence[dict[str, Any]] | None,
        reference_resolution: dict[str, Any] | None,
        compact_prompt_context: bool,
    ) -> str:
        # Research-driven tradeoff:
        # For online interaction latency, cache should key on scene + query + grounding context.
        # Excluding recent_turns avoids cache misses caused by self-history drift on repeated queries.
        payload = {
            "q": str(user_query).strip(),
            "scene": str(scene_summary).strip(),
            "detections": list(detections)[:12],
            "prefs": list(preferences)[:8],
            "object_registry": list(object_registry or [])[:12],
            "recent_scene_events": list(recent_scene_events or [])[-6:],
            "reference_resolution": dict(reference_resolution or {}),
            "compact_prompt_context": bool(compact_prompt_context),
            "adapter_path": self.adapter_path or "",
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        cached = self._response_cache.get(key)
        if cached is None:
            self._cache_stats["misses"] += 1
            return None
        self._cache_stats["hits"] += 1
        return copy.deepcopy(cached)

    def _cache_set(self, key: str, value: dict[str, Any]) -> None:
        self._response_cache[key] = copy.deepcopy(value)
        if key in self._cache_order:
            self._cache_order.remove(key)
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_limit:
            oldest = self._cache_order.pop(0)
            self._response_cache.pop(oldest, None)

    def generate_live_response(
        self,
        *,
        user_query: str,
        scene_summary: str,
        detections: Sequence[dict[str, Any]],
        preferences: Sequence[str],
        recent_turns: Sequence[dict[str, Any]],
        object_registry: Sequence[dict[str, Any]] | None = None,
        recent_scene_events: Sequence[dict[str, Any]] | None = None,
        reference_resolution: dict[str, Any] | None = None,
        enable_cache: bool = True,
        include_prompt_bundle: bool = True,
        compact_prompt_context: bool = False,
        prefer_fast_clarification: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        cache_key = self._build_cache_key(
            user_query=user_query,
            scene_summary=scene_summary,
            detections=detections,
            preferences=preferences,
            recent_turns=recent_turns,
            object_registry=object_registry,
            recent_scene_events=recent_scene_events,
            reference_resolution=reference_resolution,
            compact_prompt_context=compact_prompt_context,
        )
        if enable_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                if include_prompt_bundle:
                    cached["prompt_bundle"] = build_qwen_interaction_prompt(
                        user_query=user_query,
                        scene_summary=scene_summary,
                        detections=detections,
                        preferences=preferences,
                        recent_turns=recent_turns,
                        object_registry=object_registry or [],
                        recent_scene_events=recent_scene_events or [],
                        reference_resolution=reference_resolution or {},
                        compact_context=compact_prompt_context,
                    )
                else:
                    cached["prompt_bundle"] = None
                cached["cache_hit"] = True
                cached["latency_ms"] = (time.perf_counter() - started) * 1000
                return cached

        should_fast_clarify = bool(
            prefer_fast_clarification
            and reference_resolution
            and reference_resolution.get("clarification_needed")
            and not reference_resolution.get("resolved")
        )
        if should_fast_clarify:
            parsed = build_reference_clarification_response(
                user_query=user_query,
                scene_summary=scene_summary or "No detection summary available",
                reference_resolution=reference_resolution,
            )
            if include_prompt_bundle:
                parsed["prompt_bundle"] = build_qwen_interaction_prompt(
                    user_query=user_query,
                    scene_summary=scene_summary,
                    detections=detections,
                    preferences=preferences,
                    recent_turns=recent_turns,
                    object_registry=object_registry or [],
                    recent_scene_events=recent_scene_events or [],
                    reference_resolution=reference_resolution or {},
                    compact_context=compact_prompt_context,
                )
            else:
                parsed["prompt_bundle"] = None
            parsed["cache_hit"] = False
            parsed["latency_ms"] = (time.perf_counter() - started) * 1000
            if enable_cache:
                self._cache_set(cache_key, parsed)
            return parsed

        prompt_bundle = build_qwen_interaction_prompt(
            user_query=user_query,
            scene_summary=scene_summary,
            detections=detections,
            preferences=preferences,
            recent_turns=recent_turns,
            object_registry=object_registry or [],
            recent_scene_events=recent_scene_events or [],
            reference_resolution=reference_resolution or {},
            compact_context=compact_prompt_context,
        )

        if not self._status.available:
            parsed = build_fallback_qwen_response(
                user_query=user_query,
                detection_summary=scene_summary or "no detection summary",
            )
            parsed["prompt_bundle"] = prompt_bundle if include_prompt_bundle else None
            parsed["cache_hit"] = False
            parsed["latency_ms"] = (time.perf_counter() - started) * 1000
            if enable_cache:
                self._cache_set(cache_key, parsed)
            return parsed

        # ── 真实 Qwen2.5-VL + LoRA 推理 ──────────────────────────────
        raw_text = self._run_vlm_inference(
            user_query=user_query,
            scene_summary=scene_summary,
            detections=list(detections),
            object_registry=list(object_registry or []),
            recent_turns=list(recent_turns),
            reference_resolution=reference_resolution or {},
        )
        parsed = parse_qwen_structured_response(raw_text)
        parsed["prompt_bundle"] = prompt_bundle if include_prompt_bundle else None
        parsed["cache_hit"] = False
        parsed["latency_ms"] = (time.perf_counter() - started) * 1000
        if enable_cache:
            self._cache_set(cache_key, parsed)
        return parsed

    # ── 懒加载模型（首次调用时初始化）─────────────────────────────────
    def _ensure_model_loaded(self) -> None:
        if getattr(self, "_model", None) is not None:
            return
        import torch
        from transformers import AutoProcessor
        from transformers import Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel

        adapter_path = self.adapter_path or ""
        base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

        # 从 adapter_config 读取 base model
        import json as _json
        cfg_path = f"{adapter_path}/adapter_config.json"
        try:
            with open(cfg_path) as f:
                adapter_cfg = _json.load(f)
            base_model_id = adapter_cfg.get("base_model_name_or_path", base_model_id)
        except Exception:
            pass

        # 自动检测精度
        use_4bit = False
        dtype = torch.bfloat16
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 50:
                use_4bit = True  # 40GB 及以下用 4-bit

        bnb_config = None
        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=dtype,
        )

        if adapter_path:
            self._model = PeftModel.from_pretrained(base, adapter_path)
            self._processor = AutoProcessor.from_pretrained(
                adapter_path, trust_remote_code=True
            )
        else:
            self._model = base
            self._processor = AutoProcessor.from_pretrained(
                base_model_id, trust_remote_code=True
            )

        self._model.eval()

    def _run_vlm_inference(
        self,
        *,
        user_query: str,
        scene_summary: str,
        detections: list[dict[str, Any]],
        object_registry: list[dict[str, Any]],
        recent_turns: list[dict[str, Any]],
        reference_resolution: dict[str, Any],
    ) -> str:
        import json as _json
        import torch

        self._ensure_model_loaded()

        from uris_platform.prompts.qwen_interaction_prompt import URIS_SYSTEM_PROMPT

        context = _json.dumps(
            {
                "scene_summary": scene_summary,
                "detections": detections[:8],
                "object_registry": object_registry[:12],
                "reference_resolution": reference_resolution,
            },
            ensure_ascii=False,
            indent=2,
        )
        user_content = f"user_query: {user_query}\ncontext_json:\n{context}"

        messages = [
            {"role": "system", "content": URIS_SYSTEM_PROMPT},
        ]
        for turn in recent_turns[-4:]:
            messages.append(turn)
        messages.append({"role": "user", "content": user_content})

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=text, return_tensors="pt").to(
            next(self._model.parameters()).device
        )

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

        response = self._processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response

