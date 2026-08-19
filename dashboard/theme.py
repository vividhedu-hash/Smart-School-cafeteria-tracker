"""
Shared visual language for every dashboard page.

One import gives a page the fonts, card styles, badges, and the status strip
so pages stop re-declaring slightly different CSS.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

STATUS_CLASSES = {
    "READY": "status-ok",
    "ONLINE": "status-ok",
    "ACTIVE": "status-ok",
    "PROXY": "status-warn",
    "PENDING": "status-warn",
    "NONE": "status-warn",
    "UNKNOWN": "status-warn",
    "TRAINING": "status-info",
    "MISSING": "status-error",
    "ERROR": "status-error",
    "OFFLINE": "status-error",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220 0%, #16233a 100%);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #38bdf8; }

.state-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em;
}
.status-ok    { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.status-warn  { background: #451a03; color: #fb923c; border: 1px solid #ea580c; }
.status-info  { background: #0c4a6e; color: #7dd3fc; border: 1px solid #38bdf8; }
.status-error { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }

/* Status strip shown at the top of every page */
.status-strip {
    display: flex; flex-wrap: wrap; gap: 10px;
    background: linear-gradient(135deg, #131c2e, #0b1220);
    border: 1px solid #24314b; border-radius: 14px;
    padding: 10px 14px; margin: 0 0 18px;
}
.status-chip {
    display: flex; flex-direction: column; gap: 2px;
    padding: 4px 14px 4px 0; min-width: 96px;
    border-right: 1px solid #1f2b42;
}
.status-chip:last-child { border-right: none; }
.status-chip-label {
    font-size: 0.62rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: #64748b;
}
.status-chip-value { font-size: 0.94rem; font-weight: 650; color: #e2e8f0; }
.status-chip-value.ok    { color: #34d399; }
.status-chip-value.warn  { color: #fb923c; }
.status-chip-value.error { color: #f87171; }
.status-chip-value.info  { color: #7dd3fc; }

/* Generic glass card */
.glass-card {
    background: linear-gradient(135deg, #1a2438 0%, #0f172a 100%);
    border: 1px solid #2b3a55; border-radius: 16px;
    padding: 18px 22px; margin: 8px 0;
}
.card-title { font-size: 1.02rem; font-weight: 650; color: #e2e8f0; margin-bottom: 4px; }
.card-body  { font-size: 0.88rem; color: #94a3b8; line-height: 1.55; }

.step-card {
    background: linear-gradient(135deg, #1a2438 0%, #0f172a 100%);
    border: 1px solid #2b3a55; border-radius: 16px;
    padding: 20px 24px; margin: 8px 0;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.step-card:hover { border-color: #3b82f6; box-shadow: 0 0 24px rgba(59,130,246,0.15); }
.step-card-num {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
}
.step-card-title { font-size: 1.05rem; font-weight: 600; color: #e2e8f0; margin: 6px 0 4px; }
.step-card-desc  { font-size: 0.85rem; color: #94a3b8; line-height: 1.5; }

.nav-card {
    background: linear-gradient(135deg, #1a2438, #0f172a);
    border: 1px solid #2b3a55; border-radius: 12px;
    padding: 16px; text-align: center; transition: all 0.2s;
}
.nav-card:hover { border-color: #6366f1; box-shadow: 0 0 20px rgba(99,102,241,0.2); }
.nav-card-icon  { font-size: 2rem; margin-bottom: 8px; }
.nav-card-title { font-weight: 600; color: #e2e8f0; font-size: 0.9rem; }
.nav-card-desc  { color: #64748b; font-size: 0.78rem; margin-top: 4px; }

/* Coaching panel used by the face scanner */
.coach-panel {
    background: linear-gradient(135deg, #101b2d, #0a1120);
    border: 1px solid #24314b; border-radius: 18px;
    padding: 16px 20px; text-align: center;
}
.coach-headline { font-size: 1.35rem; font-weight: 750; color: #f1f5f9; letter-spacing: -0.02em; }
.coach-sub { font-size: 0.86rem; color: #94a3b8; margin-top: 4px; }

.scan-pulse {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    background: #3b82f6; margin-right: 6px;
    animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.7); }
}
</style>
"""


def inject_css() -> None:
    # [AI-CoLab: Verified by Antigravity] Injects unified enterprise theme CSS
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(status: str) -> str:
    """Return HTML for a coloured status pill."""
    label = (status or "UNKNOWN").upper()
    return (
        f'<span class="state-badge {STATUS_CLASSES.get(label, "status-warn")}">'
        f"{label}</span>"
    )


def status_strip(state_data: Optional[dict], engine_alive: bool, enrolled: int = 0) -> None:
    """
    Render the persistent system strip: engine, camera, FPS, models, people.

    ``state_data`` is the engine snapshot (may be empty when it is stopped).
    """
    data = state_data or {}
    components = data.get("components", {}) or {}
    active = data.get("active_models", {}) or {}

    if not engine_alive:
        engine_value, engine_tone = "Stopped", "error"
    elif data:
        engine_value, engine_tone = "Online", "ok"
    else:
        engine_value, engine_tone = "Starting…", "info"

    camera_ok = components.get("camera", "") == "OK" or data.get("camera_connected")
    if not engine_alive:
        camera_value, camera_tone = "Off", "warn"
    else:
        camera_value = "Connected" if camera_ok else "Waiting"
        camera_tone = "ok" if camera_ok else "warn"

    fps = data.get("fps") or 0.0
    people = data.get("enrolled_persons", enrolled) or enrolled

    def _model_label(task: str) -> str:
        # [AI-CoLab: Verified by Antigravity] Explicit fallback label for visual OpenCV detectors vs YOLO models
        value = active.get(task)
        if value in (None, "", "none", "None"):
            return "visual (built-in)"
        return str(value)

    chips = [
        ("Engine", engine_value, engine_tone),
        ("Camera", camera_value, camera_tone),
        ("FPS", f"{float(fps):.1f}" if engine_alive else "—", ""),
        ("Pipeline", str(data.get("state", "—")) if engine_alive else "—", ""),
        ("Plate model", _model_label("plate"), ""),
        ("Waste model", _model_label("waste"), ""),
        ("Enrolled", f"{people} people", "ok" if people else "warn"),
    ]

    html = ['<div class="status-strip">']
    for label, value, tone in chips:
        tone_cls = f" {tone}" if tone else ""
        html.append(
            f'<div class="status-chip">'
            f'<div class="status-chip-label">{label}</div>'
            f'<div class="status-chip-value{tone_cls}">{value}</div>'
            f"</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)
