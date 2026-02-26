# URIS Platform Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modular, premium Streamlit home-robot interaction simulation platform (no robot motion control) while preserving the existing VL demo as a legacy app.

**Architecture:** Introduce a new `src/uris_platform` package with domain/service/UI modules and a thin root launcher. Keep the current monolithic app under `legacy/` for compatibility. Add automation scripts and unit tests for core logic before implementation (TDD).

**Tech Stack:** Python, Streamlit, dataclasses, pytest, JSON configs, Makefile

---

### Task 1: Create test scaffolding for core platform logic

**Files:**
- Create: `/Users/shihaochen/github/URIS/tests/test_platform_config.py`
- Create: `/Users/shihaochen/github/URIS/tests/test_scenario_engine.py`
- Create: `/Users/shihaochen/github/URIS/tests/test_state_init.py`

**Step 1: Write the failing tests**
- Cover env-based config defaults/overrides
- Cover video sampling policy for short/medium/long durations
- Cover basic command planning (`move`, `clean`, fallback)
- Cover session state initialization keys

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_platform_config.py tests/test_scenario_engine.py tests/test_state_init.py -q`
Expected: FAIL (missing `uris_platform` modules)

### Task 2: Implement core package and configuration/domain/service modules

**Files:**
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/__init__.py`
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/config.py`
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/state.py`
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/domain.py`
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/services/scenario_engine.py`
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/services/perf.py`

**Step 1: Write minimal code to pass tests**
- Implement dataclasses and helper functions only required by tests

**Step 2: Run tests to verify they pass**
Run: `pytest tests/test_platform_config.py tests/test_scenario_engine.py tests/test_state_init.py -q`
Expected: PASS

### Task 3: Build premium Streamlit UI package and scene templates

**Files:**
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/ui/theme.py`
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/ui/components.py`
- Create: `/Users/shihaochen/github/URIS/src/uris_platform/streamlit_app.py`
- Create: `/Users/shihaochen/github/URIS/configs/scenes/living_room.json`
- Create: `/Users/shihaochen/github/URIS/configs/scenes/kitchen.json`
- Create: `/Users/shihaochen/github/URIS/configs/scenes/bedroom.json`
- Create: `/Users/shihaochen/github/URIS/.streamlit/config.toml`

**Step 1: Implement premium UI shell**
- Mission Control / Scenario Studio / Interaction Console / Operations / Automation tabs
- Form-driven interactions and metrics cards
- Heuristic planner backend integration

**Step 2: Manual smoke test**
Run: `streamlit run app.py`
Expected: New platform UI renders and simulates recommendation outputs for commands

### Task 4: Reorganize entrypoints while preserving legacy app

**Files:**
- Move: `/Users/shihaochen/github/URIS/app.py` -> `/Users/shihaochen/github/URIS/legacy/legacy_video_reasoning_app.py`
- Create: `/Users/shihaochen/github/URIS/app.py`
- Create: `/Users/shihaochen/github/URIS/legacy/README.md`

**Step 1: Replace root launcher**
- New `app.py` imports `src/uris_platform/streamlit_app.py`
- `legacy/README.md` documents how to run old app

**Step 2: Manual smoke test**
Run: `python app.py` (no-op launcher import check) and `streamlit run app.py`
Expected: New launcher works; legacy file remains intact

### Task 5: Add automation and project-health tooling

**Files:**
- Create: `/Users/shihaochen/github/URIS/scripts/project_doctor.py`
- Create: `/Users/shihaochen/github/URIS/scripts/benchmark_interaction.py`
- Create: `/Users/shihaochen/github/URIS/Makefile`
- Create: `/Users/shihaochen/github/URIS/.github/workflows/platform-checks.yml`

**Step 1: Implement scripts**
- `project_doctor.py`: structure + large file + template presence checks
- `benchmark_interaction.py`: batch run scenario planner and timing stats

**Step 2: Verify scripts**
Run: `python scripts/project_doctor.py` and `python scripts/benchmark_interaction.py --iterations 50`
Expected: Summary reports print without exceptions

### Task 6: Documentation updates and verification pass

**Files:**
- Modify: `/Users/shihaochen/github/URIS/README.md` (minimal addendum if needed)

**Step 1: Run verification**
Run:
- `pytest tests/test_platform_config.py tests/test_scenario_engine.py tests/test_state_init.py -q`
- `python scripts/project_doctor.py`
- `python scripts/benchmark_interaction.py --iterations 20`

**Step 2: Capture notes**
- Record what remains legacy-only (VL) and what remains adapter placeholders (YOLO / optional future integrations)
- Record next integration tasks
