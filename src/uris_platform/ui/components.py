from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from html import escape
from typing import Iterable, Sequence

import streamlit as st

from uris_platform.domain import ActionPlan


def metric_cards_html(cards: Sequence[dict[str, str]]) -> str:
    """Build compact HTML (no markdown-indentation) to avoid raw-tag leakage in Streamlit markdown."""
    items: list[str] = []
    for card in cards:
        items.append(
            "<div class=\"uris-card\">"
            f"<div class=\"label\">{escape(str(card.get('label', '')))}</div>"
            f"<div class=\"value\">{escape(str(card.get('value', '')))}</div>"
            f"<div class=\"sub\">{escape(str(card.get('sub', '')))}</div>"
            "</div>"
        )
    return "<div class=\"uris-grid\">" + "".join(items) + "</div>"


def render_hero(title: str, subtitle: str, pills: Sequence[str]) -> None:
    pill_html = "".join(f'<span class="uris-pill">{escape(p)}</span>' for p in pills)
    st.markdown(
        f"""
        <section class="uris-hero">
          <h1>{escape(title)}</h1>
          <p>{escape(subtitle)}</p>
          <div class="uris-pill-row">{pill_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(cards: Sequence[dict[str, str]]) -> None:
    st.markdown(metric_cards_html(cards), unsafe_allow_html=True)


def render_panel(title: str, body_html: str) -> None:
    st.markdown(
        f'<section class="uris-panel"><h3>{escape(title)}</h3>{body_html}</section>',
        unsafe_allow_html=True,
    )


def render_status_badges(badges: Sequence[dict[str, str]]) -> None:
    html_parts = []
    for badge in badges:
        tone = escape(str(badge.get("tone", "info")))
        cls = f"uris-badge uris-badge--{tone}" if tone in {"ok", "warn", "err"} else "uris-badge"
        html_parts.append(
            f"<span class='{cls}'>"
            f"<span class='k'>{escape(str(badge.get('label', '')))}</span>"
            f"<span class='v'>{escape(str(badge.get('value', '')))}</span>"
            "</span>"
        )
    st.markdown(f"<div class='uris-badge-row'>{''.join(html_parts)}</div>", unsafe_allow_html=True)


def render_surface(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="uris-surface">
          <div class="title">{escape(title)}</div>
          <div class="body">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_response_card(title: str, body: str, meta: Sequence[str] | None = None) -> None:
    meta_html = ""
    if meta:
        meta_html = "<div class='meta'>" + "".join(f"<span>{escape(str(m))}</span>" for m in meta) + "</div>"
    st.markdown(
        f"""
        <section class="uris-response">
          <div class="head">
            <div class="title">{escape(title)}</div>
          </div>
          <div class="body">{escape(body)}</div>
          {meta_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_scene_objects(objects: Iterable[dict]) -> None:
    rows = []
    for obj in objects:
        rows.append(
            "<div class='uris-step'>"
            f"<div class='uris-step-title'>{escape(str(obj.get('name', 'object')))}</div>"
            f"<div class='uris-step-meta'>zone={escape(str(obj.get('zone', 'unknown')))} | "
            f"state={escape(str(obj.get('state', 'idle')))}</div>"
            "</div>"
        )
    if not rows:
        rows.append("<div class='uris-step'><div class='uris-step-title'>No objects</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_action_plan(plan: ActionPlan | None) -> None:
    if plan is None:
        render_panel(
            "Interaction Recommendation",
            "<p style='color:#60706f;margin:0;'>No recommendation yet. Submit a command in Interaction Console.</p>",
        )
        return
    metadata_html = (
        "<div class='uris-kv'>"
        f"<div class='k'>Action</div><div class='v'>{escape(plan.action)}</div>"
        f"<div class='k'>Target</div><div class='v'>{escape(plan.target or 'N/A')}</div>"
        f"<div class='k'>Confidence</div><div class='v'>{plan.confidence:.2f}</div>"
        f"<div class='k'>Explanation</div><div class='v'>{escape(plan.explanation)}</div>"
        "</div>"
    )
    render_panel("Interaction Recommendation", metadata_html)

    steps_html = []
    for idx, step in enumerate(plan.steps, start=1):
        steps_html.append(
            "<div class='uris-step'>"
            f"<div class='uris-step-title'>Step {idx}</div>"
            f"<div class='uris-step-meta'>{escape(step)}</div>"
            "</div>"
        )
    if plan.adaptation_note:
        steps_html.append(
            "<div class='uris-step'>"
            "<div class='uris-step-title'>Adaptation Note</div>"
            f"<div class='uris-step-meta'>{escape(plan.adaptation_note)}</div>"
            "</div>"
        )
    st.markdown(
        f"<div class='uris-panel'><h3>Simulation Timeline</h3><div class='uris-timeline'>{''.join(steps_html)}</div></div>",
        unsafe_allow_html=True,
    )


def render_interaction_history(history: Sequence[dict]) -> None:
    if not history:
        render_panel(
            "Interaction History",
            "<p style='color:#60706f;margin:0;'>No interactions yet.</p>",
        )
        return

    for item in reversed(history[-10:]):
        ts = item.get("timestamp")
        display_ts = ts
        if isinstance(ts, (int, float)):
            display_ts = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        with st.container(border=True):
            st.markdown(
                f"**User Command**  \n`{item.get('command', '')}`  \n"
                f"_Room: {item.get('room', '')} | Time: {display_ts}_"
            )
            plan = item.get("plan")
            if plan:
                st.json(plan, expanded=False)


def render_perf_table(perf_entries: Sequence[dict]) -> None:
    if not perf_entries:
        render_panel(
            "Recent Pipeline Timings",
            "<p style='color:#60706f;margin:0;'>No timing entries recorded yet.</p>",
        )
        return
    rows = []
    for e in reversed(perf_entries[-8:]):
        rows.append(
            {
                "time": e.get("time"),
                "total_ms": round(float(e.get("total_ms", 0.0)), 2),
                "planning_ms": round(float(e.get("stages", {}).get("planning_ms", 0.0)), 2),
                "render_ms": round(float(e.get("stages", {}).get("render_ms", 0.0)), 2),
                "command": e.get("command", "")[:60],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def action_plan_as_dict(plan: ActionPlan) -> dict:
    return asdict(plan)
