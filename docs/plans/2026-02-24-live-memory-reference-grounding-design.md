# Live Memory + Reference Grounding Design

**Date:** 2026-02-24
**Scope:** Object registry, reference disambiguation, temporal memory, event summaries for Live Camera interaction
**Positioning:** Interaction simulation and recommendation platform (no robot motion control)

---

## 1. Goal

Upgrade the Live Camera path from single-snapshot interaction to continuous, stateful interaction by introducing an object registry and temporal memory. The platform should better handle user references such as “那个杯子 / 左边那个” and provide scene-change summaries grounded in recent detections.

## 2. Why This Matters

- Improves user experience in multi-object scenes (less ambiguous responses)
- Improves academic rigor (explicit grounding and memory layers)
- Supports evaluation of reference resolution and scene-change handling
- Preserves the existing modular detector + Qwen architecture

## 3. Core Additions

### 3.1 Object Registry
Maintain a live registry of tracked objects derived from YOLO detections:
- stable `obj_id`
- `label`
- `bbox`, `center_norm`
- `confidence`
- `first_seen_ts`, `last_seen_ts`
- `seen_count`
- `mention_count`
- `status` (`visible` / `stale`)

Matching strategy (Phase 1): same-label nearest-center matching with normalized distance threshold.

### 3.2 Temporal Memory
Store recent snapshots of live detections and a bounded event log:
- detection summaries over time
- scene signature history
- count deltas (appear/disappear/increase/decrease)
- latest event summary string for prompt/UI

### 3.3 Reference Resolution
Before Qwen call, run a lightweight heuristic resolver on user query using registry state:
- explicit labels (cup/chair/table)
- directional references (`左边`, `右边`, `left`, `right`)
- deictic references (`那个`, `that one`, `this one`)

Outputs a structured `reference_resolution` block with selected object and confidence, or a clarification flag.

## 4. Data Flow Changes

1. Snapshot detection completes.
2. Normalize detections (existing)
3. Update object registry + temporal memory (new)
4. Generate event summary and store in session state (new)
5. User submits query
6. Resolve references using registry (new)
7. Build Qwen prompt with scene summary + registry + recent events + reference resolution (expanded)
8. Log interaction and evaluation metadata including grounding info

## 5. Prompt Context Upgrade

Add prompt context fields:
- `object_registry`
- `recent_scene_events`
- `reference_resolution`

Keep output contract stable (`user_response` + `analysis_json`) to preserve evaluation compatibility.

## 6. UI Changes (Live Camera)

Add a grounding/memory panel:
- current object registry table (ID, label, location, conf, seen count)
- recent scene events (last N)
- latest reference resolution result for submitted query

This should be compact and not disrupt the existing live interaction UX.

## 7. Error Handling / Degradation

- No detections: empty registry, emit “scene stable / no confident detections” memory event
- Resolver cannot disambiguate: return `clarification_needed=True` and candidate list
- Missing center coordinates: fallback to label-only resolution
- Registry matching failure: create new object entries rather than forcing incorrect matches

## 8. Evaluation Hooks

Log additional metadata for future metrics:
- `reference_resolution` result and confidence
- `object_registry_size`
- `recent_event_count`
- `event_summary`

Future metrics (not required in this phase): reference-resolution accuracy, detection stability, clarification rate.

## 9. Testing Strategy

Unit tests for:
- registry stable ID reuse across nearby detections
- event generation on count changes
- directional reference resolution (`左边` / `right`)
- prompt builder includes memory/grounding context
- state defaults include new memory/registry keys

## 10. Out of Scope (This Phase)

- true multi-object tracking (ByteTrack/DeepSORT)
- segmentation-based grounding
- audio/gaze fusion
- learned reference resolver
