# Live Camera + YOLO + Qwen Interaction Design

**Date:** 2026-02-24
**Scope:** Real-time camera ingestion, object detection, on-demand Qwen interaction, academic prompt redesign
**Positioning:** Interaction simulation and recommendation platform (no robot motion control)

---

## 1. Goal

Extend the new URIS platform so users can open a camera directly in the platform, see object detection results, and immediately interact with the system through natural-language questions/commands. The system returns recommendation-oriented responses with academic rigor and evaluation-friendly structured outputs.

## 2. Core Flow

1. Browser camera feeds frames into a `Live Camera` tab.
2. YOLO performs continuous object detection (high frequency, lightweight loop).
3. Platform maintains a live scene state (objects, counts, confidence, timestamps, optional frame snapshot).
4. Qwen + LoRA is triggered on demand (user prompt) rather than every frame.
5. Qwen receives:
   - latest frame snapshot (or fallback scene summary)
   - YOLO detections summary
   - user query
   - user preferences + recent context
   - prompt template optimized for UX + academic output quality
6. Qwen returns:
   - natural language response for user-facing UI
   - structured JSON for metrics/evaluation logging

## 3. Runtime Strategy (Performance)

- YOLO high frequency (target 5-10 FPS in Phase 1, depending on hardware)
- Qwen low frequency, only on user submit by default
- optional cooldown and scene-signature cache to avoid duplicate Qwen calls
- graceful degradation if YOLO/Qwen dependencies are unavailable

## 4. Prompt Redesign (Qwen)

### 4.1 UX Principles
- answer-first
- concise but informative
- no exposed chain-of-thought
- one clarifying question maximum when needed

### 4.2 Academic Principles
- explicit distinction between observation and inference
- confidence reporting
- limitations and uncertainty notes
- structured JSON for evaluation and reproducibility

### 4.3 Output Contract
Qwen response must include:
- `user_response` (natural language)
- `analysis_json` (structured schema)

Suggested JSON fields:
- `intent`
- `user_goal`
- `observed_objects`
- `spatial_relations`
- `scene_summary`
- `recommendation_steps`
- `clarification_needed`
- `clarifying_question`
- `confidence`
- `evidence_basis`
- `limitations`

## 5. Module Plan

Create new modules:
- `/Users/shihaochen/github/URIS/src/uris_platform/prompts/qwen_interaction_prompt.py`
- `/Users/shihaochen/github/URIS/src/uris_platform/services/vision_yolo.py`
- `/Users/shihaochen/github/URIS/src/uris_platform/services/live_camera.py`
- `/Users/shihaochen/github/URIS/src/uris_platform/services/qwen_adapter.py`

Modify:
- `/Users/shihaochen/github/URIS/src/uris_platform/state.py`
- `/Users/shihaochen/github/URIS/src/uris_platform/streamlit_app.py`
- `/Users/shihaochen/github/URIS/requirements.txt` (optional dependencies hints)

## 6. Live Camera UI (Phase 1)

New `Live Camera` tab includes:
- camera source panel (WebRTC if installed, fallback `st.camera_input` snapshot mode)
- detection panel (current objects, confidence, counts)
- scene summary panel
- interaction panel (user query -> Qwen)
- response panel (natural language + collapsible structured JSON)
- status metrics (YOLO availability, Qwen availability, last update time, latency)

## 7. Error Handling / Degradation

- No `streamlit-webrtc`: show fallback snapshot mode and install instructions
- No YOLO dependency/weights: keep camera + text interaction, detection disabled
- No Qwen dependencies or model load failure: use heuristic fallback responder with same JSON schema
- JSON parse failure from model: show natural response and log schema parse error

## 8. Evaluation Hooks

Integrate live interactions into existing Evaluation Lab by logging:
- `vision_to_response_latency_ms`
- `json_schema_valid` flag
- `clarification_needed`
- live interaction source (`camera_live` / `camera_snapshot`)

## 9. Testing Strategy

Unit tests for:
- prompt builder content and schema instructions
- YOLO result normalization
- live Qwen trigger policy (cooldown / user-trigger)
- Qwen response parser (JSON extraction + fallback)
- live state defaults

Manual verification:
- camera opens (or fallback shown)
- detection list updates (real detector or mock detector)
- Qwen interaction returns natural text + JSON (or fallback)
- evaluation metrics continue updating

## 10. Out of Scope (This Phase)

- audio streaming / STT integration
- automatic Qwen trigger on scene-change (optional later)
- robot control or ROS2 actuation
- custom YOLO fine-tuning workflow
