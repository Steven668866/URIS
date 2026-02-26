# Live Camera + YOLO + Qwen Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a live camera interaction tab with YOLO-based object detection and on-demand Qwen+LoRA interaction using a redesigned academic/UX-oriented prompt, with graceful fallbacks.

**Architecture:** Introduce modular services for camera trigger policy, YOLO result normalization, and Qwen prompt/response handling; integrate them into the Streamlit platform as a new `Live Camera` tab. Use optional dependencies and fallback logic so the app remains runnable without GPU/model setup.

**Tech Stack:** Python, Streamlit, dataclasses, pytest, optional `ultralytics`, optional `streamlit-webrtc`, Transformers/PEFT (optional runtime), JSON

---

### Task 1: Add failing tests for prompt, detection normalization, trigger policy, and parser

**Files:**
- Create: `/Users/shihaochen/github/URIS/tests/test_qwen_prompt_builder.py`
- Create: `/Users/shihaochen/github/URIS/tests/test_vision_result_normalization.py`
- Create: `/Users/shihaochen/github/URIS/tests/test_live_trigger_policy.py`
- Create: `/Users/shihaochen/github/URIS/tests/test_qwen_response_parser.py`
- Modify: `/Users/shihaochen/github/URIS/tests/test_state_init.py`

**Step 1: Write failing tests**
- Prompt builder includes answer-first, observation/inference separation, and schema requirements
- YOLO normalization aggregates labels/counts/summary correctly
- Trigger policy respects user-trigger and cooldown
- Parser extracts JSON block or falls back safely
- State initialization adds live camera defaults

**Step 2: Run tests to verify failure**
Run: `pytest tests/test_qwen_prompt_builder.py tests/test_vision_result_normalization.py tests/test_live_trigger_policy.py tests/test_qwen_response_parser.py tests/test_state_init.py -q`
Expected: FAIL (missing modules / fields)

### Task 2: Implement prompt and service modules (minimal to pass)

**Files:**
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/prompts/__init__.py`
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/prompts/qwen_interaction_prompt.py`
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/services/vision_yolo.py`
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/services/live_camera.py`
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/services/qwen_adapter.py`
- Modify: `/Users/shihaochen/github/URIS/src/uris_platform/state.py`

**Step 1: Write minimal implementations**
- Prompt templates + builder
- Detection normalization and scene summary
- Trigger policy dataclass/helper
- Qwen response parser + fallback response generator
- Live state defaults

**Step 2: Run tests to verify pass**
Run: `pytest tests/test_qwen_prompt_builder.py tests/test_vision_result_normalization.py tests/test_live_trigger_policy.py tests/test_qwen_response_parser.py tests/test_state_init.py -q`
Expected: PASS

### Task 3: Integrate Live Camera tab into Streamlit app

**Files:**
- Modify: `/Users/shihaochen/github/URIS/src/uris_platform/streamlit_app.py`
- Modify: `/Users/shihaochen/github/URIS/src/uris_platform/ui/components.py` (if reusable render helpers needed)

**Step 1: Add UI and live interaction flow**
- New `Live Camera` tab
- WebRTC optional path + snapshot fallback path
- Detection summary display and scene summary
- Qwen interaction form and response panel
- Latency / availability status indicators

**Step 2: Manual smoke test**
Run: `streamlit run app.py`
Expected: App starts, `Live Camera` tab visible, graceful fallback shown if optional deps are missing

### Task 4: Wire evaluation logging and optional dependency hints

**Files:**
- Modify: `/Users/shihaochen/github/URIS/src/uris_platform/streamlit_app.py`
- Modify: `/Users/shihaochen/github/URIS/requirements.txt`
- Modify: `/Users/shihaochen/github/URIS/docs/plans/2026-02-24-live-camera-yolo-qwen-design.md` (if implementation notes differ)

**Step 1: Log live interaction metadata**
- source mode, JSON-valid flag, live latency
- preserve Evaluation Lab compatibility

**Step 2: Add dependency hints**
- optional `ultralytics`
- optional `streamlit-webrtc`
- optional `av`

### Task 5: Verification

**Files:**
- None required (verification only)

**Step 1: Run tests and compile checks**
Run:
- `pytest tests/test_platform_config.py tests/test_scenario_engine.py tests/test_state_init.py tests/test_evaluation_metrics.py tests/test_qwen_prompt_builder.py tests/test_vision_result_normalization.py tests/test_live_trigger_policy.py tests/test_qwen_response_parser.py -q`
- `python -m py_compile src/uris_platform/streamlit_app.py src/uris_platform/services/*.py src/uris_platform/prompts/qwen_interaction_prompt.py`

**Step 2: Streamlit smoke check**
Run: `streamlit run app.py --server.headless true`
Expected: startup success (camera/detector may degrade gracefully depending on environment)
