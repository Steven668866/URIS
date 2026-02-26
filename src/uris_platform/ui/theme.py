from __future__ import annotations

import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --uris-bg: #f5f1e8;
          --uris-panel: rgba(255,255,255,0.78);
          --uris-panel-strong: rgba(255,255,255,0.92);
          --uris-ink: #142022;
          --uris-muted: #60706f;
          --uris-line: rgba(20, 32, 34, 0.08);
          --uris-teal: #0d7f7a;
          --uris-teal-deep: #0a5956;
          --uris-accent: #c97f2a;
          --uris-danger: #a63f3f;
          --uris-shadow: 0 20px 60px rgba(17, 32, 34, 0.08);
        }

        .stApp {
          background:
            radial-gradient(circle at 12% 12%, rgba(13,127,122,0.14), transparent 45%),
            radial-gradient(circle at 90% 8%, rgba(201,127,42,0.12), transparent 40%),
            radial-gradient(circle at 75% 78%, rgba(10,89,86,0.09), transparent 35%),
            linear-gradient(180deg, #f7f3eb 0%, #efe9dd 100%);
          color: var(--uris-ink);
        }

        .block-container {
          max-width: 1280px;
          padding-top: 1.25rem;
          padding-bottom: 2rem;
        }

        .uris-hero {
          position: relative;
          border: 1px solid var(--uris-line);
          border-radius: 22px;
          padding: 1.1rem 1.25rem;
          background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(245,251,250,0.88));
          box-shadow: var(--uris-shadow);
          overflow: hidden;
          margin-bottom: 0.9rem;
        }

        .uris-hero::after {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(100deg, rgba(13,127,122,0.07), transparent 35%, rgba(201,127,42,0.06));
          pointer-events: none;
        }

        .uris-hero h1 {
          margin: 0;
          font-size: 1.7rem;
          line-height: 1.15;
          letter-spacing: -0.02em;
          color: #0f1d1f;
        }

        .uris-hero p {
          margin: 0.35rem 0 0;
          color: var(--uris-muted);
          font-size: 0.95rem;
        }

        .uris-pill-row {
          display: flex;
          gap: 0.45rem;
          flex-wrap: wrap;
          margin-top: 0.8rem;
        }

        .uris-pill {
          display: inline-flex;
          align-items: center;
          border-radius: 999px;
          border: 1px solid rgba(13,127,122,0.14);
          background: rgba(13,127,122,0.06);
          color: var(--uris-teal-deep);
          padding: 0.28rem 0.62rem;
          font-size: 0.77rem;
          font-weight: 600;
          letter-spacing: 0.01em;
        }

        .uris-badge-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem;
          margin: 0.35rem 0 0.75rem;
        }

        .uris-badge {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          border-radius: 12px;
          border: 1px solid var(--uris-line);
          padding: 0.34rem 0.58rem;
          background: rgba(255,255,255,0.74);
          box-shadow: 0 6px 18px rgba(17, 32, 34, 0.04);
        }

        .uris-badge .k {
          color: var(--uris-muted);
          font-size: 0.72rem;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .uris-badge .v {
          font-weight: 700;
          font-size: 0.8rem;
          color: var(--uris-ink);
        }

        .uris-badge--ok {
          border-color: rgba(13,127,122,0.22);
          background: rgba(13,127,122,0.07);
        }

        .uris-badge--warn {
          border-color: rgba(201,127,42,0.23);
          background: rgba(201,127,42,0.07);
        }

        .uris-badge--err {
          border-color: rgba(166,63,63,0.23);
          background: rgba(166,63,63,0.06);
        }

        .uris-surface {
          border-radius: 16px;
          border: 1px solid var(--uris-line);
          background: rgba(255,255,255,0.86);
          box-shadow: 0 10px 26px rgba(17,32,34,0.04);
          padding: 0.8rem 0.9rem;
          margin: 0.4rem 0 0.8rem;
        }

        .uris-surface .title {
          margin: 0 0 0.3rem;
          color: var(--uris-ink);
          font-size: 0.9rem;
          font-weight: 700;
          letter-spacing: -0.01em;
        }

        .uris-surface .body {
          color: var(--uris-muted);
          font-size: 0.82rem;
          line-height: 1.45;
        }

        .uris-response {
          border-radius: 18px;
          border: 1px solid var(--uris-line);
          background:
            linear-gradient(180deg, rgba(255,255,255,0.93), rgba(247,251,250,0.85));
          box-shadow: 0 12px 30px rgba(17,32,34,0.05);
          padding: 0.95rem 1rem;
          margin-bottom: 0.7rem;
        }

        .uris-response .head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 0.6rem;
          margin-bottom: 0.45rem;
        }

        .uris-response .head .title {
          margin: 0;
          font-size: 0.95rem;
          color: #143135;
          font-weight: 700;
        }

        .uris-response .body {
          color: #18282a;
          line-height: 1.5;
          font-size: 0.88rem;
        }

        .uris-response .meta {
          margin-top: 0.55rem;
          color: var(--uris-muted);
          font-size: 0.75rem;
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem;
        }

        .uris-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 0.7rem;
          margin: 0.5rem 0 0.9rem;
        }

        .uris-card {
          border-radius: 18px;
          border: 1px solid var(--uris-line);
          background: var(--uris-panel);
          backdrop-filter: blur(10px);
          box-shadow: 0 10px 28px rgba(17, 32, 34, 0.05);
          padding: 0.85rem 0.9rem;
        }

        .uris-card .label {
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--uris-muted);
          margin-bottom: 0.35rem;
        }

        .uris-card .value {
          font-size: 1.15rem;
          font-weight: 700;
          color: var(--uris-ink);
          line-height: 1.1;
        }

        .uris-card .sub {
          margin-top: 0.25rem;
          color: var(--uris-muted);
          font-size: 0.78rem;
        }

        .uris-panel {
          border-radius: 18px;
          border: 1px solid var(--uris-line);
          background: var(--uris-panel-strong);
          box-shadow: 0 12px 32px rgba(17, 32, 34, 0.05);
          padding: 0.95rem 1rem;
          margin-bottom: 0.8rem;
        }

        .uris-panel h3 {
          margin: 0 0 0.45rem;
          font-size: 1rem;
          color: #132325;
          letter-spacing: -0.01em;
        }

        .uris-mono {
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: 0.84rem;
        }

        .uris-timeline {
          border-left: 2px solid rgba(13,127,122,0.18);
          margin-left: 0.4rem;
          padding-left: 0.9rem;
        }

        .uris-step {
          margin-bottom: 0.55rem;
          padding: 0.55rem 0.7rem;
          border: 1px solid var(--uris-line);
          border-radius: 12px;
          background: rgba(255,255,255,0.75);
        }

        .uris-step-title {
          font-weight: 600;
          color: #153033;
          margin-bottom: 0.15rem;
          font-size: 0.88rem;
        }

        .uris-step-meta {
          color: var(--uris-muted);
          font-size: 0.76rem;
        }

        .uris-kv {
          display: grid;
          grid-template-columns: 120px 1fr;
          gap: 0.35rem 0.55rem;
          align-items: start;
        }
        .uris-kv .k { color: var(--uris-muted); font-size: 0.8rem; }
        .uris-kv .v { color: var(--uris-ink); font-size: 0.84rem; }

        @media (max-width: 900px) {
          .uris-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .uris-kv { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
