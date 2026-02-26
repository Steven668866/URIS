# Live Memory + Reference Grounding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add object registry, temporal memory/event summaries, and lightweight reference resolution to the Live Camera interaction path, and pass the new grounding/memory context into the Qwen prompt.

**Architecture:** Introduce a small `live_scene_memory` service that updates tracked objects and emits scene events from normalized detections. Integrate it into the snapshot detection flow and Qwen interaction flow in `streamlit_app.py`, while expanding the prompt builder and adapter interface to include registry/event/reference context.

**Tech Stack:** Python, Streamlit, dataclasses, pytest, existing YOLO/Qwen adapter modules, JSON

---

### Task 1: Add failing tests for registry, reference resolution, temporal memory, and prompt context

**Files:**
- Create: `/Users/shihaochen/github/URIS/tests/test_live_scene_memory.py`
- Modify: `/Users/shihaochen/github/URIS/tests/test_qwen_prompt_builder.py`
- Modify: `/Users/shihaochen/github/URIS/tests/test_state_init.py`

**Step 1: Write failing tests**
- stable object IDs are reused across nearby detections
- event summaries are emitted when object counts change
- directional/deictic references resolve to best candidate or request clarification
- prompt includes object registry / recent events / reference-resolution context
- state initialization includes new live memory keys

**Step 2: Run targeted tests to confirm RED**
Run: `pytest tests/test_live_scene_memory.py tests/test_qwen_prompt_builder.py tests/test_state_init.py -q`
Expected: FAIL on missing module/functions/fields

### Task 2: Implement minimal live scene memory service and state defaults

**Files:**
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/services/live_scene_memory.py`
- Modify: `/Users/shihaochen/github/URIS/src/uris_platform/state.py`

**Step 1: Add minimal registry + event logic**
- registry upsert with same-label nearest-center matching
- bounded detection history/event log helpers
- reference resolver using labels + left/right + deictic heuristics

**Step 2: Run targeted tests to confirm GREEN**
Run: `pytest tests/test_live_scene_memory.py tests/test_state_init.py -q`
Expected: PASS

### Task 3: Expand Qwen prompt builder + adapter interface

**Files:**
- Modify: `/Users/shihaochen/github/URIS/src/uris_platform/prompts/qwen_interaction_prompt.py`
- Modify: `/Users/shihaochen/github/URIS/src/uris_platform/services/qwen_adapter.py`
- Test: `/Users/shihaochen/github/URIS/tests/test_qwen_prompt_builder.py`

**Step 1: Add new prompt context inputs**
- `object_registry`
- `recent_scene_events`
- `reference_resolution`

**Step 2: Pass through adapter and keep output compatibility**
- adapter accepts new arguments, defaults safely
- no schema-breaking changes

**Step 3: Run prompt tests**
Run: `pytest tests/test_qwen_prompt_builder.py tests/test_qwen_response_parser.py -q`
Expected: PASS

### Task 4: Integrate memory/grounding into Live Camera UI flow

**Files:**
- Modify: `/Users/shihaochen/github/URIS/src/uris_platform/streamlit_app.py`
- (Optional) Modify: `/Users/shihaochen/github/URIS/src/uris_platform/ui/components.py`

**Step 1: Update state on snapshot detection**
- call registry/memory updater after normalized detections
- persist registry, event log, event summary, detection history

**Step 2: Resolve references before Qwen call**
- compute `reference_resolution` from query + registry
- pass registry/events/reference block to adapter prompt builder
- record grounding metadata into live response history

**Step 3: Add compact UI surfaces**
- registry preview table
- recent event log
- latest reference resolution block

**Step 4: Manual smoke test**
Run: `streamlit run app.py`
Expected: Live Camera tab shows registry/event sections and keeps existing interaction flow intact

### Task 5: Verification

**Files:**
- None (verification only)

**Step 1: Run tests and compile checks**
Run:
- `pytest tests -q`
- `python -m py_compile src/uris_platform/streamlit_app.py src/uris_platform/services/*.py src/uris_platform/prompts/qwen_interaction_prompt.py`

**Step 2: Streamlit smoke startup**
Run: `streamlit run app.py --server.headless true`
Expected: startup success; live tab degrades gracefully if optional deps missing
