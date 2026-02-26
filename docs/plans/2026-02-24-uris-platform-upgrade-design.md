# URIS Platform Upgrade Design

**Date:** 2026-02-24
**Scope:** UI premiumization, performance improvements, project restructuring, automation upgrades
**Approach:** Phase-based migration (recommended mixed route)

---

## 1. Problem Statement

The current `/Users/shihaochen/github/URIS/app.py` is a monolithic Streamlit application that mixes:
- model loading and inference runtime
- video and camera processing
- voice/STT handling
- UI layout and interaction logic
- session state management
- cache and memory cleanup logic

This makes the system hard to evolve into the target product direction: an online interactive platform that simulates home-robot interaction in domestic environments.

## 2. Product Direction (Aligned to URIS Proposal)

Upgrade the project from a video reasoning demo to a **home robot interaction simulation platform** that can:
- accept multimodal inputs (text/voice/video/camera; voice/video can degrade gracefully)
- represent a home scene and user intent
- produce interaction/task recommendations and natural-language feedback
- track adaptation/personalization signals
- surface latency/resource metrics for system evaluation
- remain compatible with future model iterations and YOLO integration (without requiring robot motion control)

## 3. Architecture (Phase 1 in Streamlit, future-ready)

### 3.1 Layers

- **UI / Presentation**: Streamlit pages, premium visual style, interaction forms, metrics panels
- **Application**: interaction orchestration, command execution flow, pipeline timing, session actions
- **Domain**: scene state, robot state, action planning objects, adaptation memory
- **Infrastructure**: configuration, file paths, caches, logging, template loading
- **Adapters**: heuristic planner (now), future LLM/YOLO connectors (later)

### 3.2 Migration Strategy

- Preserve existing heavy video reasoning app as a legacy entry under `legacy/`
- Replace root `app.py` with a new platform launcher
- Build new modular package under `src/uris_platform/`
- Keep compatibility with `streamlit run app.py`

## 4. UI Design (Premium + Research Demo Friendly)

### 4.1 Information Architecture

Tabs/pages inside one Streamlit app:
- **Mission Control**: system overview, status cards, quick simulation actions, current scene summary
- **Scenario Studio**: create/edit room scene and user profile, choose templates
- **Interaction Console**: command input, simulated task recommendation planning, simulation timeline, adaptation notes
- **Operations**: latency and cache metrics, performance history, runtime diagnostics
- **Evaluation Lab**: response latency, recommendation consistency, satisfaction records, simulated task completion metrics
- **Automation**: project scripts, maintenance commands, benchmark shortcuts, folder organization guidance

### 4.2 Visual Direction

- warm neutral + teal industrial palette (avoid default Streamlit look)
- gradient hero header and panel chrome
- card-based metrics and timeline components
- compact but high-information layout for research demos
- consistent typography via custom CSS variables

## 5. Performance Plan (System-first, model-agnostic)

### 5.1 Immediate Improvements

- modular state initialization (reduce accidental rerun churn)
- form-based interactions (avoid unnecessary re-execution on every widget update)
- cached scene/template loading (`st.cache_data`)
- lightweight planner caching (`functools.lru_cache`)
- stage-level timing instrumentation for profiling
- batch benchmark script for simulation latency regression tracking

### 5.2 Future Adapter Performance Hooks

- pluggable backend interface (`heuristic`, `llm`, `llm+yolo`)
- pipeline stage timing (perception / intent / plan / response)
- optional async queue and background workers when heavier model adapters are added

## 6. Project Structure Reorganization

Target additions / migration:
- `/Users/shihaochen/github/URIS/src/uris_platform/` (new modular app package)
- `/Users/shihaochen/github/URIS/tests/` (unit tests for core platform logic)
- `/Users/shihaochen/github/URIS/scripts/` (automation and diagnostics)
- `/Users/shihaochen/github/URIS/configs/scenes/` (scene templates)
- `/Users/shihaochen/github/URIS/legacy/` (old Streamlit video reasoning app)
- `/Users/shihaochen/github/URIS/.streamlit/config.toml` (theme/runtime defaults)

No large model files or datasets are moved in this phase.

## 7. Automation Upgrades

- `Makefile` shortcuts (`run`, `run-legacy`, `test`, `doctor`, `benchmark`)
- `scripts/project_doctor.py` for repo audit (large files, structure, quick health checks)
- `scripts/benchmark_interaction.py` for latency benchmarking
- optional CI workflow for core unit tests and diagnostics

## 8. Error Handling & Graceful Degradation

- New platform should run without GPU/model dependencies
- Legacy app remains available for full VL inference workflows
- Missing optional dependencies are reported in UI with clear recovery guidance
- Invalid scene JSON or command input returns structured validation errors instead of crashing

## 9. Testing Strategy

Phase 1 unit tests cover:
- configuration loading / env overrides
- video sampling policy helper
- heuristic planning behavior (core command types)
- session state initialization defaults

Manual verification covers:
- Streamlit UI rendering
- scenario template switching
- interaction timeline updates
- evaluation feedback recording and metric updates
- benchmark/doctor scripts execution

## 10. Out-of-Scope (This Phase)

- replacing the VL model backend itself
- physical robot motion control integration
- real YOLO detector integration in the new platform path
- reinforcement learning adaptation loop implementation

These remain future adapter milestones (only if later research scope expands beyond simulation).
