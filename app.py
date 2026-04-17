"""
AI Student Performance Analyzer — Streamlit App
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import altair as alt

from model.train import PerformanceModel, engineer_features, READINESS_WEIGHTS, FEATURE_COLS
from utils.suggestions import (
    rank_features, get_suggestions, get_grade, get_trend_label, FEATURE_LABELS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & full-page background ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp, [data-testid="stAppViewContainer"] { background: #030303 !important; }
[data-testid="stMain"], .block-container { background: transparent !important; }
.block-container { padding-top: 1.6rem !important; padding-bottom: 2rem !important; }

/* ── Ambient page radial glow (indigo left, rose right) ── */
.stApp::before {
  content: ''; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse at 15% 50%, rgba(99,102,241,.055) 0%, transparent 55%),
    radial-gradient(ellipse at 85% 45%, rgba(244,63,94,.055) 0%, transparent 55%);
}

/* ── Floating shape keyframes (whole-page background) ── */
@keyframes sf1{0%,100%{transform:rotate(12deg) translateY(0)}50%{transform:rotate(12deg) translateY(18px)}}
@keyframes sf2{0%,100%{transform:rotate(-15deg) translateY(0)}50%{transform:rotate(-15deg) translateY(18px)}}
@keyframes sf3{0%,100%{transform:rotate(-8deg) translateY(0)}50%{transform:rotate(-8deg) translateY(18px)}}
@keyframes sf4{0%,100%{transform:rotate(20deg) translateY(0)}50%{transform:rotate(20deg) translateY(18px)}}
@keyframes sf5{0%,100%{transform:rotate(-25deg) translateY(0)}50%{transform:rotate(-25deg) translateY(18px)}}


/* ── KPI cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 24px; }
.kpi-card {
  background: linear-gradient(145deg, #0d0f1c 0%, #080910 100%);
  border: 1px solid rgba(99,102,241,.18);
  border-radius: 16px; padding: 22px 18px 18px;
  text-align: center; position: relative; overflow: hidden;
  box-shadow: 0 4px 28px rgba(0,0,0,.55);
}
.kpi-card::after {
  content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, rgba(165,180,252,.4), rgba(253,164,175,.4), transparent);
}
.kpi-icon  { font-size: 1.6rem; margin-bottom: 8px; display: block; }
.kpi-label { font-size: .68rem; color: #475569; letter-spacing: .12em;
             text-transform: uppercase; margin-bottom: 8px; }
.kpi-val   { font-size: 2.3rem; font-weight: 800; line-height: 1.1; color: #f1f5f9; }
.kpi-denom { font-size: 1rem; font-weight: 500; color: #64748b; }
.kpi-sub   { font-size: .72rem; color: #475569; margin-top: 6px; }

/* ── Section header ── */
.sec-hd {
  font-size: .82rem; font-weight: 700; color: #a5b4fc;
  letter-spacing: .12em; text-transform: uppercase;
  border-bottom: 1px solid rgba(165,180,252,.15);
  padding-bottom: 8px; margin: 20px 0 16px;
}

/* ── Custom progress bars ── */
.prog-wrap  { margin-bottom: 14px; }
.prog-top   { display: flex; justify-content: space-between; align-items: center;
              margin-bottom: 5px; }
.prog-name  { font-size: .84rem; font-weight: 600; color: #cbd5e1; }
.prog-right { display: flex; align-items: center; gap: 8px; }
.prog-pct   { font-size: .82rem; font-weight: 700; }
.prog-track { background: #0d0f1a; border-radius: 100px; height: 7px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 100px; transition: width .4s ease; }

.badge-good { background: rgba(34,197,94,.15);  color: #4ade80;
              font-size:.68rem; padding:2px 9px; border-radius:100px;
              border: 1px solid rgba(34,197,94,.3); font-weight:600; }
.badge-fair { background: rgba(245,158,11,.15); color: #fbbf24;
              font-size:.68rem; padding:2px 9px; border-radius:100px;
              border: 1px solid rgba(245,158,11,.3); font-weight:600; }
.badge-poor { background: rgba(239,68,68,.15);  color: #f87171;
              font-size:.68rem; padding:2px 9px; border-radius:100px;
              border: 1px solid rgba(239,68,68,.3); font-weight:600; }

/* ── Insight cards (weakness / strength) ── */
.insight-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 16px 0; }
.insight-card {
  border-radius: 14px; padding: 20px 18px;
  position: relative; overflow: hidden;
}
.insight-card-red   { background: linear-gradient(135deg, #1a0608, #0f0305);
                      border: 1px solid rgba(244,63,94,.3); }
.insight-card-green { background: linear-gradient(135deg, #050e18, #030810);
                      border: 1px solid rgba(165,180,252,.25); }
.ins-tag  { font-size:.65rem; font-weight:700; letter-spacing:.12em;
            text-transform:uppercase; margin-bottom:6px; }
.ins-name { font-size:1.25rem; font-weight:800; margin: 2px 0; }
.ins-score{ font-size:2rem; font-weight:900; line-height:1; margin: 8px 0 4px; }
.ins-desc { font-size:.78rem; color:#94a3b8; }

/* ── Tip cards ── */
.tip-card {
  border-radius: 12px; padding: 16px 18px; margin-bottom: 12px;
  position: relative; overflow: hidden;
}
.tip-high   { background: linear-gradient(135deg,#1f0808,#180808);
              border-left: 4px solid #ef4444; }
.tip-medium { background: linear-gradient(135deg,#1f1408,#160f00);
              border-left: 4px solid #f59e0b; }
.tip-low    { background: linear-gradient(135deg,#081a0e,#06140a);
              border-left: 4px solid #22c55e; }
.tip-hd     { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.tip-title  { font-size: .9rem; font-weight: 700; color: #e2e8f0; }
.tip-pri    { font-size: .63rem; font-weight: 700; letter-spacing: .1em;
              padding: 2px 8px; border-radius: 100px; text-transform: uppercase; }
.pri-high   { background: rgba(239,68,68,.2);  color: #f87171; }
.pri-medium { background: rgba(245,158,11,.2); color: #fbbf24; }
.pri-low    { background: rgba(34,197,94,.2);  color: #4ade80; }
.tip-body   { font-size: .82rem; color: #94a3b8; line-height: 1.55; }

/* ── Readiness formula row ── */
.formula-row {
  display: grid; grid-template-columns: repeat(6,1fr); gap: 8px;
  margin: 12px 0;
}
.formula-cell {
  background: #0d0f1c; border: 1px solid rgba(165,180,252,.15);
  border-radius: 10px; padding: 12px 8px; text-align: center;
}
.fc-name   { font-size:.66rem; color:#64748b; margin-bottom:4px; }
.fc-weight { font-size:.75rem; font-weight:700; color:#818cf8; }
.fc-score  { font-size:1.05rem; font-weight:800; color:#e2e8f0; margin:3px 0; }
.fc-contrib{ font-size:.72rem; }

/* ── Altair chart wrapper ── */
.chart-box {
  background: #06080f; border: 1px solid rgba(165,180,252,.1);
  border-radius: 14px; padding: 16px; margin: 4px 0;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid rgba(165,180,252,.1); margin: 22px 0; }

/* ── Animated header (AnimatedText port) ── */
@keyframes anim-hd-fade {
  from { opacity:0; transform:translateX(-10px); }
  to   { opacity:1; transform:translateX(0); }
}
@keyframes anim-hd-draw {
  from { stroke-dashoffset: 420; }
  to   { stroke-dashoffset: 0; }
}
.anim-hd-wrap {
  position:relative; display:inline-block;
  padding-bottom:14px; margin-bottom:18px; width:100%;
}
.anim-hd-text {
  font-size:.82rem; font-weight:800; letter-spacing:.12em;
  text-transform:uppercase; color:#a5b4fc;
  animation: anim-hd-fade 0.55s ease forwards;
}
.anim-hd-svg {
  position:absolute; bottom:0; left:0;
  width:100%; height:12px; overflow:visible;
}
.anim-hd-path {
  stroke-dasharray: 420; stroke-dashoffset: 420;
  animation: anim-hd-draw 1.4s ease-in-out 0.25s forwards;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #030303 !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.sb-section {
  background: rgba(99,102,241,.06); border: 1px solid rgba(165,180,252,.12);
  border-radius: 12px; padding: 14px 12px; margin-bottom: 12px;
}
.sb-title { font-size:.72rem; font-weight:700; color:#a5b4fc; letter-spacing:.1em;
            text-transform:uppercase; margin-bottom:10px; }

/* ── Hide Streamlit chrome ── */
#MainMenu { display: none !important; }
footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
/* Hide header bar but keep sidebar toggle visible on mobile */
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stHeader"]::before { display: none !important; }
/* Always show the sidebar open/close arrow */
[data-testid="stSidebarCollapsedControl"] { display: flex !important; visibility: visible !important; }
[data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; }

/* ── Tab styling override ── */
button[data-baseweb="tab"] { font-size:.84rem !important; font-weight:600 !important; }

/* ═══════════════════════════════════
   RESPONSIVE — tablet / mobile
   ═══════════════════════════════════ */

/* Tablet (≤ 960px) */
@media (max-width: 960px) {
  .kpi-grid    { grid-template-columns: repeat(2, 1fr); }
  .insight-row { grid-template-columns: 1fr; }
  .formula-row { grid-template-columns: repeat(3, 1fr); }
  .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}

/* Phone landscape / small tablet (≤ 680px) */
@media (max-width: 680px) {
  .kpi-grid    { grid-template-columns: repeat(2, 1fr); gap: 10px; }
  .kpi-card    { padding: 16px 12px 12px; }
  .kpi-val     { font-size: 1.75rem; }
  .formula-row { grid-template-columns: repeat(2, 1fr); }
  .tip-card    { padding: 12px 14px; }
  .insight-card { padding: 16px 14px; }
  .ins-score   { font-size: 1.55rem; }
}

/* Phone portrait (≤ 480px) */
@media (max-width: 480px) {
  .kpi-grid    { grid-template-columns: 1fr 1fr; gap: 8px; }
  .kpi-val     { font-size: 1.55rem; }
  .kpi-icon    { font-size: 1.25rem; }
  .kpi-label   { font-size: .6rem; }
  .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
  .formula-row { grid-template-columns: repeat(2, 1fr); gap: 6px; }
  .formula-cell { padding: 8px 6px; }
  .fc-name     { font-size: .58rem; }
  .fc-weight   { font-size: .65rem; }
  .fc-score    { font-size: .88rem; }
  .anim-hd-text { font-size: .72rem; }
  button[data-baseweb="tab"] { font-size:.72rem !important; padding: 6px 8px !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Full-page floating shape background (vw-relative so it scales on all devices) ──
st.markdown("""
<div style="position:fixed;width:min(700px,90vw);height:min(160px,18vw);left:-5%;top:18%;z-index:0;
  border-radius:50%;border:2px solid rgba(255,255,255,.1);
  background:linear-gradient(to right,rgba(99,102,241,.13),transparent);
  box-shadow:0 8px 32px rgba(255,255,255,.05);
  backdrop-filter:blur(2px);pointer-events:none;
  animation:sf1 14s ease-in-out infinite;"></div>

<div style="position:fixed;width:min(560px,80vw);height:min(130px,15vw);right:-2%;top:72%;z-index:0;
  border-radius:50%;border:2px solid rgba(255,255,255,.09);
  background:linear-gradient(to right,rgba(244,63,94,.13),transparent);
  box-shadow:0 8px 32px rgba(255,255,255,.05);
  backdrop-filter:blur(2px);pointer-events:none;
  animation:sf2 16s ease-in-out 1s infinite;"></div>

<div style="position:fixed;width:min(320px,65vw);height:min(85px,12vw);left:8%;bottom:12%;z-index:0;
  border-radius:50%;border:2px solid rgba(255,255,255,.08);
  background:linear-gradient(to right,rgba(139,92,246,.13),transparent);
  box-shadow:0 8px 32px rgba(255,255,255,.05);
  backdrop-filter:blur(2px);pointer-events:none;
  animation:sf3 18s ease-in-out 2s infinite;"></div>

<div style="position:fixed;width:min(220px,50vw);height:min(65px,10vw);right:18%;top:12%;z-index:0;
  border-radius:50%;border:2px solid rgba(255,255,255,.08);
  background:linear-gradient(to right,rgba(245,158,11,.13),transparent);
  box-shadow:0 8px 32px rgba(255,255,255,.05);
  backdrop-filter:blur(2px);pointer-events:none;
  animation:sf4 20s ease-in-out 0.5s infinite;"></div>

<div style="position:fixed;width:min(160px,40vw);height:min(45px,8vw);left:28%;top:8%;z-index:0;
  border-radius:50%;border:2px solid rgba(255,255,255,.07);
  background:linear-gradient(to right,rgba(6,182,212,.13),transparent);
  box-shadow:0 8px 32px rgba(255,255,255,.05);
  backdrop-filter:blur(2px);pointer-events:none;
  animation:sf5 22s ease-in-out 1.5s infinite;"></div>
""", unsafe_allow_html=True)


# ── Altair chart helpers ──────────────────────────────────────────────────────
CHART_CONFIG = dict(background="transparent")

def _base():
    """Shared axis/view config for all charts."""
    return {
        "axis": {"gridColor": "#1e2540", "domainColor": "#2d3561",
                 "labelColor": "#64748b", "titleColor": "#94a3b8",
                 "tickColor": "#2d3561"},
        "view": {"stroke": "transparent", "fill": "transparent"},
    }

def bar_chart(df, x, y, color="#6366f1", h=220, h_orient=False):
    """Styled vertical or horizontal bar chart."""
    if h_orient:
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=color)
            .encode(
                y=alt.Y(y, sort="-x", axis=alt.Axis(labelColor="#94a3b8", domainColor="#2d3561")),
                x=alt.X(x, axis=alt.Axis(labelColor="#64748b", gridColor="#1e2540",
                                          domainColor="#2d3561")),
                tooltip=[y, x],
            )
            .properties(height=h, background="transparent")
            .configure_view(stroke="transparent", fill="transparent")
            .configure_axis(gridColor="#1e2540", domainColor="#2d3561")
        )
    else:
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=color)
            .encode(
                x=alt.X(x, sort=None, axis=alt.Axis(labelColor="#94a3b8", domainColor="#2d3561")),
                y=alt.Y(y, axis=alt.Axis(labelColor="#64748b", gridColor="#1e2540",
                                          domainColor="#2d3561")),
                tooltip=[x, y],
            )
            .properties(height=h, background="transparent")
            .configure_view(stroke="transparent", fill="transparent")
            .configure_axis(gridColor="#1e2540", domainColor="#2d3561")
        )
    return chart


def line_chart(df, x, y, color="#22c55e", h=220):
    return (
        alt.Chart(df)
        .mark_line(color=color, strokeWidth=3, interpolate="monotone")
        .encode(
            x=alt.X(x, sort=None, axis=alt.Axis(labelColor="#94a3b8", domainColor="#2d3561")),
            y=alt.Y(y, scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(labelColor="#64748b", gridColor="#1e2540",
                                   domainColor="#2d3561")),
            tooltip=[x, y],
        )
        .mark_line(color=color, strokeWidth=3, interpolate="monotone")
        + alt.Chart(df).mark_point(color=color, size=80, filled=True).encode(
            x=alt.X(x, sort=None),
            y=alt.Y(y, scale=alt.Scale(domain=[0, 100])),
            tooltip=[x, y],
        )
    ).properties(height=h, background="transparent"
    ).configure_view(stroke="transparent", fill="transparent"
    ).configure_axis(gridColor="#1e2540", domainColor="#2d3561")


def color_bar_chart(df, x, y, h=260):
    """Bar chart where bar color = green/yellow/red based on score value."""
    df = df.copy()
    df["_cat"] = df[x].apply(
        lambda s: "Good" if s >= 65 else ("Fair" if s >= 40 else "Poor")
    )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y(y, sort="-x", axis=alt.Axis(labelColor="#cbd5e1", domainColor="#2d3561")),
            x=alt.X(x, scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(labelColor="#64748b", gridColor="#1e2540", domainColor="#2d3561")),
            color=alt.Color("_cat:N",
                scale=alt.Scale(domain=["Good","Fair","Poor"],
                                range=["#22c55e","#f59e0b","#ef4444"]),
                legend=None,
            ),
            tooltip=[y, x],
        )
        .properties(height=h, background="transparent")
        .configure_view(stroke="transparent", fill="transparent")
        .configure_axis(gridColor="#1e2540", domainColor="#2d3561")
    )


def grouped_bar_chart(df, x, ys, colors, h=220):
    """Multi-series grouped bar chart for model comparison."""
    layers = []
    for y, color in zip(ys, colors):
        layers.append(
            alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=color)
            .encode(
                x=alt.X("Model:N", axis=alt.Axis(labelColor="#94a3b8", domainColor="#2d3561"), title=""),
                y=alt.Y(f"{y}:Q", axis=alt.Axis(labelColor="#64748b", gridColor="#1e2540",
                                                   domainColor="#2d3561"), title=y),
                tooltip=["Model:N", f"{y}:Q"],
            )
        )
    return (
        alt.layer(*layers)
        .properties(height=h, background="transparent")
        .configure_view(stroke="transparent", fill="transparent")
        .configure_axis(gridColor="#1e2540", domainColor="#2d3561")
    )


# ── Progress bar HTML ─────────────────────────────────────────────────────────
def progress_bar(label, score, show_badge=True):
    if score >= 65:
        fill  = "linear-gradient(90deg,#22c55e,#4ade80)"
        pct_c = "#4ade80"
        badge = '<span class="badge-good">Good</span>'
    elif score >= 40:
        fill  = "linear-gradient(90deg,#f59e0b,#fbbf24)"
        pct_c = "#fbbf24"
        badge = '<span class="badge-fair">Fair</span>'
    else:
        fill  = "linear-gradient(90deg,#ef4444,#f87171)"
        pct_c = "#f87171"
        badge = '<span class="badge-poor">Needs Work</span>'

    badge_html = badge if show_badge else ""
    return f"""
    <div class="prog-wrap">
      <div class="prog-top">
        <span class="prog-name">{label}</span>
        <div class="prog-right">
          {badge_html}
          <span class="prog-pct" style="color:{pct_c}">{score}%</span>
        </div>
      </div>
      <div class="prog-track">
        <div class="prog-fill" style="width:{min(score,100)}%; background:{fill}"></div>
      </div>
    </div>"""


# ── Animated section header (port of AnimatedText component) ─────────────────
def animated_header(text: str, color: str = "#818cf8") -> str:
    """
    Renders a section title with a wavy SVG underline that draws itself in.
    Ported from the animated-underline-text-one React/framer-motion component.
    Uses CSS @keyframes defined in the global style block.
    """
    return f"""
    <div class="anim-hd-wrap">
      <div class="anim-hd-text" style="color:{color}">{text}</div>
      <svg class="anim-hd-svg" viewBox="0 0 300 12" preserveAspectRatio="none">
        <path class="anim-hd-path"
              d="M 0,6 Q 75,0 150,6 Q 225,12 300,6"
              stroke="{color}" stroke-width="1.8" fill="none" opacity="0.55"/>
      </svg>
    </div>"""


# ════════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training AI models…")
def load_model():
    m = PerformanceModel()
    m.train()
    return m

model = load_model()


# ════════════════════════════════════════════════════════════════════════════════
# INPUT PANEL — inline expander (works on desktop + mobile, no sidebar needed)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Style the expander to match dark theme */
[data-testid="stExpander"] {
  background: linear-gradient(145deg,#0d0f1c,#080910) !important;
  border: 1px solid rgba(165,180,252,.2) !important;
  border-radius: 16px !important;
  margin-bottom: 18px;
}
[data-testid="stExpander"] summary {
  font-size: .9rem !important; font-weight: 700 !important;
  color: #a5b4fc !important; padding: 14px 18px !important;
}
[data-testid="stExpanderDetails"] { padding: 0 18px 16px !important; }
/* Hide sidebar entirely since inputs are inline */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

with st.expander("🎓  Enter Your Study Data — tap to open", expanded=False):
    st.caption("All predictions update live as you adjust the sliders.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="sb-title" style="color:#a5b4fc;font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">Previous Test Marks</div>', unsafe_allow_html=True)
        m1 = st.slider("Test 1 — oldest", 0, 100, 55)
        m2 = st.slider("Test 2 — middle", 0, 100, 62)
        m3 = st.slider("Test 3 — latest", 0, 100, 68)

    with c2:
        st.markdown('<div class="sb-title" style="color:#a5b4fc;font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">Daily Study Hours (last 7 days)</div>', unsafe_allow_html=True)
        days     = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        defaults = [3.0, 3.5, 2.5, 4.0, 3.0, 5.0, 2.0]
        daily_hours = [st.slider(d, 0.0, 10.0, dfl, 0.5, key=f"h_{d}") for d,dfl in zip(days,defaults)]

    with c3:
        st.markdown('<div class="sb-title" style="color:#a5b4fc;font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">Study Quality</div>', unsafe_allow_html=True)
        focused_time     = st.slider("Focused Study Time (hrs/day)", 0.0, 8.0, 2.0, 0.5)
        revision_freq    = st.slider("Revision Sessions / Week",     0,   10,  3)
        distraction      = st.slider("Distraction Level (0–10)",     0.0, 10.0,4.0, 0.5)
        subject_strength = st.slider("Subject Strength (1–10)",      1.0, 10.0,6.0, 0.5)

    st.markdown(
        f"<div style='text-align:center;padding:8px 0 0'>"
        f"<span style='font-size:.7rem;color:#475569'>Model: <b style='color:#818cf8'>{model.best_model_name}</b> · "
        f"R²={model.r2:.3f} · MAE≈{model.mae:.1f}</span></div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# COMPUTE
# ════════════════════════════════════════════════════════════════════════════════
features    = engineer_features([m1,m2,m3], daily_hours, focused_time,
                                 revision_freq, distraction, subject_strength)
result      = model.predict(features)
ranked      = rank_features(features)
weakest     = ranked[0]
strongest   = ranked[-1]
suggestions = get_suggestions(features)
trend_label = get_trend_label(features["improvement"])
grade, grade_label = get_grade(result["predicted_marks"])

score       = result["readiness_score"]
pred        = result["predicted_marks"]
r_color     = "#22c55e" if score>=70 else "#f59e0b" if score>=45 else "#ef4444"
grade_color = "#22c55e" if grade in("A+","A") else "#f59e0b" if grade in("B","C") else "#ef4444"
trend_color = "#22c55e" if features["improvement"]>=0.55 else "#f59e0b" if features["improvement"]>=0.45 else "#ef4444"


# ════════════════════════════════════════════════════════════════════════════════
# HERO HEADER — simple starfield + title (shapes are now full-page background)
# ════════════════════════════════════════════════════════════════════════════════
_best_model = model.best_model_name
HERO_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
html,body{width:100%;height:100%;background:transparent;overflow:hidden;
          font-family:'Inter','Segoe UI',sans-serif;color:#fff;}

/* Stars */
@keyframes twinkle{0%,100%{opacity:.15}50%{opacity:.9}}
.star{position:absolute;border-radius:50%;background:#fff;}

/* Content fade-up */
@keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}

.content{
  position:absolute;inset:0;z-index:10;
  display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center;padding:0 20px;
}
.badge{
  display:inline-flex;align-items:center;gap:8px;
  padding:5px 14px;border-radius:100px;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);
  margin-bottom:18px;animation:fadeUp .8s ease .3s both;
}
.bdot{width:7px;height:7px;border-radius:50%;background:rgba(244,63,94,.9);}
.btxt{font-size:.7rem;color:rgba(255,255,255,.5);letter-spacing:.09em;}

.t1{
  font-size:2rem;font-weight:800;line-height:1.15;
  background:linear-gradient(to bottom,#fff,rgba(255,255,255,.75));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:fadeUp .8s ease .5s both;
}
.t2{
  font-size:2rem;font-weight:800;line-height:1.15;margin-bottom:14px;
  background:linear-gradient(to right,#a5b4fc,rgba(255,255,255,.9),#fda4af);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:fadeUp .8s ease .7s both;
}
.sub{
  font-size:.78rem;color:rgba(255,255,255,.35);max-width:420px;
  line-height:1.7;font-weight:300;letter-spacing:.03em;
  animation:fadeUp .8s ease .9s both;
}
.pills{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:16px;animation:fadeUp .8s ease 1.1s both;}
.pill{font-size:.62rem;font-weight:700;padding:4px 12px;border-radius:100px;letter-spacing:.07em;}
.p1{background:rgba(99,102,241,.18);color:#a5b4fc;border:1px solid rgba(99,102,241,.3);}
.p2{background:rgba(244,63,94,.15);color:#fda4af;border:1px solid rgba(244,63,94,.28);}
.p3{background:rgba(139,92,246,.15);color:#c4b5fd;border:1px solid rgba(139,92,246,.28);}
.p4{background:rgba(6,182,212,.12);color:#67e8f9;border:1px solid rgba(6,182,212,.25);}

@media(max-width:600px){
  .t1,.t2{font-size:1.45rem;}
  .sub{font-size:.72rem;}
  .badge{padding:4px 10px;}
  .pill{font-size:.58rem;padding:3px 9px;}
}
@media(max-width:380px){
  .t1,.t2{font-size:1.2rem;}
  .pills{gap:5px;}
}
</style></head>
<body>
<div id="starfield"></div>
<div class="content">
  <div class="badge"><div class="bdot"></div><span class="btxt">AI Performance Analytics</span></div>
  <div class="t1">Predict Your Exam Score</div>
  <div class="t2">Powered by Machine Learning</div>
  <div class="sub">Six behavioral signals. Two competing models.<br>One transparent, explainable prediction.</div>
  <div class="pills">
    <span class="pill p1">ML-Powered</span>
    <span class="pill p2">Live Predictions</span>
    <span class="pill p3">2,000 Samples</span>
    <span class="pill p4">""" + _best_model + """ Active</span>
  </div>
</div>
<script>
(function(){
  var sf = document.getElementById('starfield');
  var W = window.innerWidth, H = window.innerHeight;
  for(var i=0;i<120;i++){
    var s=document.createElement('div');
    s.className='star';
    var sz=Math.random()*2+0.5;
    var dur=Math.random()*4+2;
    var del=Math.random()*5;
    s.style.cssText='width:'+sz+'px;height:'+sz+'px;left:'+
      (Math.random()*100)+'%;top:'+(Math.random()*100)+'%;'+
      'opacity:'+(Math.random()*0.5+0.1)+';'+
      'animation:twinkle '+dur+'s ease-in-out '+del+'s infinite;';
    sf.appendChild(s);
  }
})();
</script>
</body></html>"""

components.html(HERO_HTML, height=260)


# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Prediction",
    "🔍  Breakdown & Insights",
    "⚙️  Model Comparison",
    "🔄  What-If Simulator",
    "📖  How It Works",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  ·  PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # KPI cards — neo-brutalist dark style (inspired by pricing card design)
    _dot_cards = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
html,body{{background:transparent;font-family:'Inter','Segoe UI',sans-serif;overflow:hidden;height:100%;}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;padding:20px 4px 8px;}}

/* Card base */
.card{{
  background:linear-gradient(145deg,#0d0f1c 0%,#090b15 100%);
  border-radius:16px;padding:20px 16px 14px;
  position:relative;overflow:visible;
  transition:transform .18s ease,box-shadow .18s ease;
  cursor:default;
}}
.card:hover{{transform:translate(-2px,-3px);}}

/* Animated floating badge — top-right corner */
.badge-ball{{
  position:absolute;top:-18px;right:-14px;
  width:58px;height:58px;border-radius:50%;
  border:2.5px solid rgba(0,0,0,.55);
  display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center;
  z-index:20;
  box-shadow:3px 3px 0px 0px rgba(0,0,0,.75);
}}
@keyframes ballFloat{{
  0%,100%{{transform:rotate(0deg) translateY(0) scale(1)}}
  20%{{transform:rotate(8deg) translateY(-5px) scale(1.07)}}
  50%{{transform:rotate(-6deg) translateY(-8px) scale(1.04)}}
  80%{{transform:rotate(4deg) translateY(-3px) scale(1.02)}}
}}

/* Feature rows */
.feat{{
  display:flex;align-items:center;gap:7px;
  padding:5px 8px;margin-bottom:5px;
  background:rgba(255,255,255,.025);
  border:1px solid rgba(255,255,255,.055);
  border-radius:8px;font-size:.67rem;color:#64748b;font-weight:500;
}}
.fdot{{width:5px;height:5px;border-radius:50%;flex-shrink:0;}}
.fcheck{{
  width:16px;height:16px;border-radius:4px;
  display:flex;align-items:center;justify-content:center;
  font-size:.55rem;font-weight:900;color:#fff;flex-shrink:0;
  border:1px solid rgba(0,0,0,.4);
  box-shadow:1px 1px 0 rgba(0,0,0,.6);
}}

.card-icon{{font-size:1.25rem;margin-bottom:6px;display:block;}}
.card-lbl{{font-size:.58rem;font-weight:700;color:#475569;
           letter-spacing:.13em;text-transform:uppercase;margin-bottom:5px;}}
.card-val{{font-size:1.85rem;font-weight:900;line-height:1.1;margin-bottom:10px;}}
.card-denom{{font-size:.82rem;font-weight:600;color:#475569;}}

/* Per-card accent border + hard shadow */
.c1{{border:2px solid rgba(99,102,241,.4); box-shadow:5px 5px 0 0 rgba(99,102,241,.5);}}
.c2{{border:2px solid {r_color}55;         box-shadow:5px 5px 0 0 {r_color}77;}}
.c3{{border:2px solid {grade_color}55;     box-shadow:5px 5px 0 0 {grade_color}77;}}
.c4{{border:2px solid {trend_color}55;     box-shadow:5px 5px 0 0 {trend_color}77;}}

.c1:hover{{box-shadow:7px 7px 0 0 rgba(99,102,241,.55);}}
.c2:hover{{box-shadow:7px 7px 0 0 {r_color}88;}}
.c3:hover{{box-shadow:7px 7px 0 0 {grade_color}88;}}
.c4:hover{{box-shadow:7px 7px 0 0 {trend_color}88;}}

/* ── Tablet: 2-column grid ── */
@media (max-width: 720px) {{
  .grid {{ grid-template-columns: repeat(2,1fr); gap:12px; }}
  .card {{ padding:16px 14px 12px; }}
  .card-val {{ font-size:1.6rem; margin-bottom:8px; }}
  .badge-ball {{ width:48px;height:48px;top:-14px;right:-10px; }}
}}

/* ── Phone portrait: still 2-col but tighter ── */
@media (max-width: 420px) {{
  .grid {{ grid-template-columns: repeat(2,1fr); gap:9px; padding:16px 2px 6px; }}
  .card {{ padding:14px 10px 10px; border-radius:12px; }}
  .card-icon {{ font-size:1rem; margin-bottom:4px; }}
  .card-lbl {{ font-size:.52rem; letter-spacing:.08em; }}
  .card-val {{ font-size:1.35rem; margin-bottom:6px; }}
  .card-denom {{ font-size:.68rem; }}
  .badge-ball {{ width:40px;height:40px;top:-12px;right:-8px; }}
  .feat {{ padding:4px 6px; font-size:.6rem; gap:5px; }}
  .fcheck {{ width:13px;height:13px;font-size:.48rem; }}
}}
</style></head><body>
<div class="grid">

  <!-- Card 1 · Predicted Marks (indigo) -->
  <div class="card c1">
    <div class="badge-ball" style="background:linear-gradient(135deg,#6366f1,#818cf8);
         animation:ballFloat 4.8s ease-in-out infinite;">
      <div style="font-size:.72rem;font-weight:900;color:#fff;line-height:1.1;">
        <span id="c1b">0</span>
      </div>
      <div style="font-size:.48rem;font-weight:700;color:rgba(255,255,255,.7)">/100</div>
    </div>
    <div class="card-icon">🎯</div>
    <div class="card-lbl">Predicted Marks</div>
    <div class="card-val" style="color:#a5b4fc;"><span id="c1v">0</span><span class="card-denom">/100</span></div>
    <div class="feat">
      <div class="fcheck" style="background:#6366f1;">✓</div>
      Range: {result['confidence_low']}–{result['confidence_high']}
    </div>
    <div class="feat">
      <div class="fdot" style="background:#818cf8;"></div>
      ML model prediction
    </div>
  </div>

  <!-- Card 2 · Readiness Score -->
  <div class="card c2">
    <div class="badge-ball" style="background:linear-gradient(135deg,{r_color},{r_color}cc);
         animation:ballFloat 5.4s ease-in-out .7s infinite;">
      <div style="font-size:.8rem;font-weight:900;color:#fff;line-height:1;">
        <span id="c2b">0</span>%
      </div>
    </div>
    <div class="card-icon">⚡</div>
    <div class="card-lbl">Readiness Score</div>
    <div class="card-val" style="color:{r_color};"><span id="c2v">0</span><span class="card-denom">%</span></div>
    <div class="feat">
      <div class="fcheck" style="background:{r_color};">✓</div>
      Overall preparedness
    </div>
    <div class="feat">
      <div class="fdot" style="background:{r_color}aa;"></div>
      Weighted behaviour score
    </div>
  </div>

  <!-- Card 3 · Predicted Grade -->
  <div class="card c3">
    <div class="badge-ball" style="background:linear-gradient(135deg,{grade_color},{grade_color}bb);
         animation:ballFloat 6s ease-in-out 1.4s infinite;">
      <div style="font-size:1.1rem;font-weight:900;color:#fff;">{grade}</div>
    </div>
    <div class="card-icon">🏅</div>
    <div class="card-lbl">Predicted Grade</div>
    <div class="card-val" style="color:{grade_color};font-size:2.5rem;">{grade}</div>
    <div class="feat">
      <div class="fcheck" style="background:{grade_color};">✓</div>
      {grade_label}
    </div>
    <div class="feat">
      <div class="fdot" style="background:{grade_color}aa;"></div>
      Based on ML prediction
    </div>
  </div>

  <!-- Card 4 · Improvement Trend -->
  <div class="card c4">
    <div class="badge-ball" style="background:linear-gradient(135deg,{trend_color},{trend_color}cc);
         animation:ballFloat 5.8s ease-in-out 2.1s infinite;">
      <div style="font-size:1.2rem;line-height:1;">📈</div>
    </div>
    <div class="card-icon">📊</div>
    <div class="card-lbl">Improvement Trend</div>
    <div class="card-val" style="color:{trend_color};font-size:.95rem;padding-top:4px;line-height:1.3;">{trend_label}</div>
    <div class="feat">
      <div class="fcheck" style="background:{trend_color};">✓</div>
      Based on 3 test marks
    </div>
    <div class="feat">
      <div class="fdot" style="background:{trend_color}aa;"></div>
      Slope trend analysis
    </div>
  </div>

</div>
<script>
function countUp(id, target, isFloat) {{
  var el = document.getElementById(id);
  if (!el) return;
  var s=0, inc=target/(1200/40);
  var t=setInterval(function(){{
    s+=inc; if(s>=target){{s=target;clearInterval(t);}}
    el.textContent=isFloat?parseFloat(s).toFixed(1):Math.round(s);
  }},40);
}}
countUp('c1b', {pred}, true);
countUp('c1v', {pred}, true);
countUp('c2b', {score}, false);
countUp('c2v', {score}, false);

// Auto-report height to Streamlit so iframe fits content on all screen sizes
(function resizeIframe() {{
  var h = document.body.scrollHeight || document.documentElement.scrollHeight;
  window.parent.postMessage({{type:'streamlit:setFrameHeight', height: h + 4}}, '*');
}})();
window.addEventListener('resize', function() {{
  var h = document.body.scrollHeight || document.documentElement.scrollHeight;
  window.parent.postMessage({{type:'streamlit:setFrameHeight', height: h + 4}}, '*');
}});
</script>
</body></html>"""
    components.html(_dot_cards, height=460, scrolling=False)

    # Charts row
    left, right = st.columns(2)

    with left:
        st.markdown(animated_header("Weekly Study Hours"), unsafe_allow_html=True)
        hours_df = pd.DataFrame({"Day": days, "Hours": daily_hours})
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.altair_chart(bar_chart(hours_df, "Day", "Hours", "#6366f1", h=210),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(animated_header("Test Marks Trend", "#22c55e"), unsafe_allow_html=True)
        marks_df = pd.DataFrame({
            "Test":  ["Test 1","Test 2","Test 3"],
            "Marks": [m1, m2, m3],
        })
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.altair_chart(line_chart(marks_df, "Test", "Marks", "#22c55e", h=210),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Readiness formula
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(animated_header("Readiness Score — How It's Calculated"), unsafe_allow_html=True)

    bd = result["readiness_breakdown"]
    cells = ""
    for feat, weight in READINESS_WEIGHTS.items():
        contrib = bd[feat]
        c_color = "#4ade80" if contrib > 0 else "#f87171"
        w_sign  = "+" if weight > 0 else ""
        cells += f"""
        <div class="formula-cell">
          <div class="fc-name">{FEATURE_LABELS[feat]}</div>
          <div class="fc-weight">Weight: {w_sign}{int(weight*100)}%</div>
          <div class="fc-score">{round(features[feat]*100)}%</div>
          <div class="fc-contrib" style="color:{c_color}">
            {'+' if contrib>=0 else ''}{contrib} pts
          </div>
        </div>"""

    st.markdown(f"""
    <div class="formula-row">{cells}</div>
    <p style="font-size:.8rem;color:#64748b;margin-top:8px">
      Total Readiness = <b style="color:{r_color}">{score}%</b>
      (sum of all contributions, capped at 0–100)
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  ·  BREAKDOWN & INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    # Feature score chart (horizontal, color-coded)
    st.markdown(animated_header("Performance Scores by Feature"), unsafe_allow_html=True)
    feat_df = pd.DataFrame([
        {"Feature": r["label"], "Score (%)": r["score"]} for r in ranked
    ]).sort_values("Score (%)", ascending=True)  # ascending for horizontal chart
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.altair_chart(color_bar_chart(feat_df, "Score (%)", "Feature", h=260),
                    use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Weakness / Strength cards
    st.markdown(animated_header("Key Insights", "#f59e0b"), unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-row">
      <div class="insight-card insight-card-red">
        <div class="ins-tag" style="color:#f87171">⚠ Biggest Weakness</div>
        <div class="ins-name" style="color:#fca5a5">{weakest['label']}</div>
        <div class="ins-score" style="color:#ef4444">{weakest['score']}%</div>
        <div class="ins-desc">Lowest scoring area — this has the most room for improvement.</div>
      </div>
      <div class="insight-card insight-card-green">
        <div class="ins-tag" style="color:#4ade80">✦ Strongest Area</div>
        <div class="ins-name" style="color:#86efac">{strongest['label']}</div>
        <div class="ins-score" style="color:#22c55e">{strongest['score']}%</div>
        <div class="ins-desc">Your best feature — keep maintaining this level.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Detailed progress bars
    st.markdown(animated_header("Detailed Breakdown"), unsafe_allow_html=True)
    bars_html = ""
    for item in sorted(ranked, key=lambda x: x["score"], reverse=True):
        bars_html += progress_bar(item["label"], item["score"])
    st.markdown(bars_html, unsafe_allow_html=True)

    # Suggestions
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(animated_header("Smart Suggestions", "#22c55e"), unsafe_allow_html=True)

    priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    for tip in suggestions:
        p = tip["priority"]
        st.markdown(f"""
        <div class="tip-card tip-{p}">
          <div class="tip-hd">
            <span>{priority_icon[p]}</span>
            <span class="tip-title">{tip['title']}</span>
            <span class="tip-pri pri-{p}">{p.title()} Priority</span>
          </div>
          <div class="tip-body">{tip['detail']}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  ·  MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab3:

    st.markdown(animated_header("Linear Regression vs Random Forest", "#22d3ee"), unsafe_allow_html=True)
    st.caption("Both models trained on same 2,000-sample synthetic dataset (80/20 split). "
               "The better R² model is used for predictions above.")

    comp = model.comparison_summary()

    # Metric KPI cards for both models
    mc1, mc2 = st.columns(2)
    for col, (name, metrics) in zip([mc1, mc2], comp.items()):
        is_best = (name == model.best_model_name)
        border_col = "rgba(99,102,241,.5)" if is_best else "rgba(99,102,241,.15)"
        badge = "✓ IN USE" if is_best else ""
        badge_html = (f'<span style="font-size:.65rem;background:rgba(99,102,241,.25);'
                      f'color:#818cf8;padding:3px 10px;border-radius:100px;'
                      f'font-weight:700;letter-spacing:.08em">{badge}</span>') if badge else ""
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-color:{border_col}">
              <div class="kpi-label">{name} {badge_html}</div>
              <div style="display:flex;justify-content:center;gap:32px;margin-top:14px">
                <div style="text-align:center">
                  <div style="font-size:.68rem;color:#64748b;margin-bottom:4px">R² SCORE</div>
                  <div style="font-size:1.8rem;font-weight:800;color:#818cf8">{metrics['R²']:.3f}</div>
                  <div style="font-size:.68rem;color:#475569">Higher = better</div>
                </div>
                <div style="text-align:center">
                  <div style="font-size:.68rem;color:#64748b;margin-bottom:4px">MAE (marks)</div>
                  <div style="font-size:1.8rem;font-weight:800;color:#f97316">{metrics['MAE']:.2f}</div>
                  <div style="font-size:.68rem;color:#475569">Lower = better</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # R² and MAE charts side by side
    ch1, ch2 = st.columns(2)
    comp_df = pd.DataFrame([
        {"Model": name, "R²": round(m["R²"],4), "MAE": round(m["MAE"],2)}
        for name, m in comp.items()
    ])

    with ch1:
        st.markdown('<div class="sec-hd">R² Score Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.altair_chart(bar_chart(comp_df, "Model", "R²", "#6366f1", h=180),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="sec-hd">MAE Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.altair_chart(bar_chart(comp_df, "Model", "MAE", "#f97316", h=180),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Feature importance
    st.markdown(animated_header("Random Forest — Feature Importance", "#22d3ee"), unsafe_allow_html=True)
    st.caption("How much each feature influenced the RF model's decisions. "
               "Higher = stronger driver of exam marks.")
    imp_df = pd.DataFrame({
        "Feature":    [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "Importance": [round(model.rf_importances[f]*100, 1) for f in FEATURE_COLS],
    }).sort_values("Importance", ascending=True)
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.altair_chart(color_bar_chart(imp_df, "Importance", "Feature", h=260),
                    use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # LR coefficients
    st.markdown('<div class="sec-hd">Linear Regression — Coefficients</div>',
                unsafe_allow_html=True)
    st.caption("Change in predicted marks per unit increase in that feature (after scaling). "
               "Negative = penalises marks.")
    coef_df = pd.DataFrame({
        "Feature":     [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "Coefficient": [round(float(c), 3) for c in model.lr.coef_],
    }).sort_values("Coefficient", ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  ·  WHAT-IF SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab4:

    st.markdown(animated_header("What-If Simulator", "#f97316"), unsafe_allow_html=True)
    st.caption("Adjust these sliders to simulate improvements. "
               "The comparison updates instantly to show the impact.")

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        wif_hours    = st.slider("Study Hours/Day",   0.0, 10.0, float(np.mean(daily_hours)), 0.5, key="wif_h")
        wif_focused  = st.slider("Focused Time/Day",  0.0,  8.0, focused_time,               0.5, key="wif_f")
    with wc2:
        wif_revision = st.slider("Revision/Week",     0,   10,   revision_freq,                   key="wif_r")
        wif_distract = st.slider("Distraction Level", 0.0, 10.0, distraction,                0.5, key="wif_d")
    with wc3:
        wif_subject  = st.slider("Subject Strength",  1.0, 10.0, subject_strength,           0.5, key="wif_s")

    wif_features = engineer_features(
        [m1,m2,m3], [wif_hours]*7, wif_focused, wif_revision, wif_distract, wif_subject,
    )
    wif_result = model.predict(wif_features)
    wif_grade, wif_grade_label = get_grade(wif_result["predicted_marks"])

    d_marks  = round(wif_result["predicted_marks"] - pred, 1)
    d_ready  = round(wif_result["readiness_score"]  - score, 1)
    d_sign   = lambda v: f"+{v}" if v >= 0 else str(v)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Result cards
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <span class="kpi-icon">🎯</span>
        <div class="kpi-label">Current Prediction</div>
        <div class="kpi-val">{pred}<span class="kpi-denom">/100</span></div>
        <div class="kpi-sub">Grade: {grade}</div>
      </div>
      <div class="kpi-card" style="border-color:rgba(99,102,241,.45)">
        <span class="kpi-icon">✨</span>
        <div class="kpi-label">What-If Prediction</div>
        <div class="kpi-val" style="color:{'#4ade80' if d_marks>=0 else '#f87171'}">
          {wif_result['predicted_marks']}<span class="kpi-denom">/100</span>
        </div>
        <div class="kpi-sub">
          <span style="color:{'#4ade80' if d_marks>=0 else '#f87171'}">
            {d_sign(d_marks)} marks
          </span> vs current
        </div>
      </div>
      <div class="kpi-card">
        <span class="kpi-icon">⚡</span>
        <div class="kpi-label">Current Readiness</div>
        <div class="kpi-val" style="color:{r_color}">{score}%</div>
        <div class="kpi-sub">&nbsp;</div>
      </div>
      <div class="kpi-card" style="border-color:rgba(99,102,241,.45)">
        <span class="kpi-icon">🚀</span>
        <div class="kpi-label">What-If Readiness</div>
        <div class="kpi-val" style="color:{'#4ade80' if d_ready>=0 else '#f87171'}">
          {wif_result['readiness_score']}%
        </div>
        <div class="kpi-sub">
          <span style="color:{'#4ade80' if d_ready>=0 else '#f87171'}">
            {d_sign(d_ready)}%
          </span> vs current
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature comparison chart
    st.markdown(animated_header("Feature Score: Current vs What-If", "#f97316"), unsafe_allow_html=True)
    wif_ranked = rank_features(wif_features)
    label_map  = {r["feature"]: r["label"] for r in ranked}

    curr_scores = {r["feature"]: r["score"] for r in ranked}
    wif_scores  = {r["feature"]: r["score"] for r in wif_ranked}

    cmp_rows = []
    for feat in FEATURE_COLS:
        cmp_rows.append({"Feature": FEATURE_LABELS[feat],
                         "Type": "Current",  "Score": curr_scores[feat]})
        cmp_rows.append({"Feature": FEATURE_LABELS[feat],
                         "Type": "What-If",  "Score": wif_scores[feat]})

    cmp_df = pd.DataFrame(cmp_rows)
    cmp_chart = (
        alt.Chart(cmp_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            y=alt.Y("Feature:N", sort=None,
                    axis=alt.Axis(labelColor="#94a3b8", domainColor="#2d3561")),
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0,100]),
                    axis=alt.Axis(labelColor="#64748b", gridColor="#1e2540",
                                   domainColor="#2d3561")),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=["Current","What-If"], range=["#6366f1","#22c55e"]
            )),
            xOffset="Type:N",
            tooltip=["Feature:N","Type:N","Score:Q"],
        )
        .properties(height=280, background="transparent")
        .configure_view(stroke="transparent", fill="transparent")
        .configure_axis(gridColor="#1e2540", domainColor="#2d3561")
        .configure_legend(labelColor="#94a3b8", titleColor="#94a3b8",
                          fillColor="#12172a", strokeColor="#2d3561",
                          padding=8, cornerRadius=8)
    )
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.altair_chart(cmp_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5  ·  HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────

# hero-ascii-one ported to HTML — UnicornStudio ASCII art background with
# technical mono layout (corner accents, header bar, CTA panel, footer bar)
HERO_ASCII_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  *{margin:0;padding:0;box-sizing:border-box;}
  html,body{width:100%;height:100%;overflow:hidden;background:#000;
            font-family:'Courier New',Courier,monospace;color:#fff;}

  /* Background animation (left 60%) */
  #us-bg{position:absolute;top:0;left:0;width:60%;height:100%;overflow:hidden;}

  /* Mobile fallback: star field */
  #stars{
    position:absolute;inset:0;
    background-image:
      radial-gradient(1px 1px at 15% 25%,#fff,transparent),
      radial-gradient(1px 1px at 55% 65%,#fff,transparent),
      radial-gradient(1px 1px at 80% 15%,#fff,transparent),
      radial-gradient(1px 1px at 35% 75%,#fff,transparent),
      radial-gradient(1px 1px at 70% 45%,#fff,transparent),
      radial-gradient(1px 1px at 25% 55%,#fff,transparent),
      radial-gradient(1px 1px at 90% 70%,#fff,transparent),
      radial-gradient(1px 1px at 45% 35%,#fff,transparent);
    opacity:.25;
  }

  /* Top header bar */
  #hdr{
    position:absolute;top:0;left:0;right:0;z-index:20;
    border-bottom:1px solid rgba(255,255,255,.18);
    display:flex;align-items:center;justify-content:space-between;
    padding:10px 24px;
  }
  .hdr-brand{font-size:1.1rem;font-weight:700;letter-spacing:.18em;
             font-style:italic;transform:skewX(-10deg);}
  .hdr-sep{width:1px;height:14px;background:rgba(255,255,255,.35);margin:0 10px;}
  .hdr-est{font-size:.62rem;color:rgba(255,255,255,.5);letter-spacing:.08em;}
  .hdr-coords{font-size:.65rem;color:rgba(255,255,255,.45);display:flex;
              align-items:center;gap:8px;}
  .hdr-dot{width:4px;height:4px;background:rgba(255,255,255,.4);border-radius:50%;}

  /* Corner frame accents */
  .corner{position:absolute;width:28px;height:28px;z-index:20;}
  .c-tl{top:0;left:0;border-top:2px solid rgba(255,255,255,.28);border-left:2px solid rgba(255,255,255,.28);}
  .c-tr{top:0;right:0;border-top:2px solid rgba(255,255,255,.28);border-right:2px solid rgba(255,255,255,.28);}
  .c-bl{bottom:32px;left:0;border-bottom:2px solid rgba(255,255,255,.28);border-left:2px solid rgba(255,255,255,.28);}
  .c-br{bottom:32px;right:0;border-bottom:2px solid rgba(255,255,255,.28);border-right:2px solid rgba(255,255,255,.28);}

  /* Right content panel */
  #panel{
    position:absolute;top:0;right:0;width:50%;height:calc(100% - 32px);
    display:flex;flex-direction:column;justify-content:center;
    padding:0 48px 0 32px;
    background:linear-gradient(90deg,transparent 0%,rgba(0,0,0,.55) 30%,rgba(0,0,0,.82) 100%);
  }

  /* Decorative line */
  .deco-line{display:flex;align-items:center;gap:8px;margin-bottom:10px;opacity:.55;}
  .deco-line hr{flex:1;border:none;border-top:1px solid #fff;}
  .deco-inf{font-size:.75rem;}

  /* Dot row */
  .dot-row{display:flex;gap:4px;margin-bottom:10px;opacity:.35;}
  .dot-row span{width:3px;height:3px;background:#fff;border-radius:50%;}

  /* Main title */
  #main-title{
    font-size:2rem;font-weight:700;letter-spacing:.12em;
    line-height:1.15;margin-bottom:10px;white-space:nowrap;
  }

  /* Description */
  #desc{font-size:.75rem;color:rgba(255,255,255,.75);line-height:1.7;
        margin-bottom:18px;max-width:340px;}

  /* CTA buttons */
  .btns{display:flex;gap:10px;flex-wrap:wrap;}
  .btn{
    padding:8px 18px;font-family:inherit;font-size:.72rem;
    font-weight:600;letter-spacing:.08em;cursor:pointer;
    border:1px solid #fff;background:transparent;color:#fff;
    transition:background .18s,color .18s;position:relative;
  }
  .btn:hover{background:#fff;color:#000;}
  .btn .corner-tl,.btn .corner-br{
    position:absolute;width:6px;height:6px;
    opacity:0;transition:opacity .18s;
  }
  .btn .corner-tl{top:-2px;left:-2px;
    border-top:1px solid #fff;border-left:1px solid #fff;}
  .btn .corner-br{bottom:-2px;right:-2px;
    border-bottom:1px solid #fff;border-right:1px solid #fff;}
  .btn:hover .corner-tl,.btn:hover .corner-br{opacity:1;}

  /* Bottom notation */
  .bottom-note{display:flex;align-items:center;gap:8px;margin-top:18px;opacity:.38;}
  .bottom-note hr{flex:1;border:none;border-top:1px solid #fff;}
  .bottom-note span{font-size:.6rem;}

  /* Footer bar */
  #ftr{
    position:absolute;bottom:0;left:0;right:0;z-index:20;height:32px;
    border-top:1px solid rgba(255,255,255,.18);
    background:rgba(0,0,0,.5);backdrop-filter:blur(6px);
    display:flex;align-items:center;justify-content:space-between;
    padding:0 20px;
  }
  .ftr-left,.ftr-right{display:flex;align-items:center;gap:10px;
                        font-size:.6rem;color:rgba(255,255,255,.45);}
  .bar-eq{display:flex;gap:2px;align-items:flex-end;}
  .bar-eq div{width:3px;background:rgba(255,255,255,.3);}
  .pulse{border-radius:50%;background:rgba(255,255,255,.6);
         animation:blink 1.4s ease-in-out infinite;}
  @keyframes blink{0%,100%{opacity:.7}50%{opacity:.2}}

  /* Responsive */
  @media(max-width:640px){
    #us-bg{width:100%;height:55%;}
    #panel{position:relative;top:auto;right:auto;width:100%;height:auto;
           padding:16px;background:rgba(0,0,0,.7);
           margin-top:calc(55% - 32px);}
    #main-title{font-size:1.25rem;white-space:normal;}
    #desc{font-size:.68rem;margin-bottom:12px;}
    .hdr-coords{display:none;}
    .btns{gap:6px;}
    .btn{padding:6px 12px;font-size:.65rem;}
  }
  @media(max-width:400px){
    #main-title{font-size:1.05rem;}
    #hdr{padding:8px 12px;}
    .hdr-brand{font-size:.9rem;}
  }
</style>
</head>
<body>

<!-- ASCII / UnicornStudio background -->
<div id="us-bg">
  <div id="stars"></div>
  <div data-us-project="OMzqyUv6M3kSnv0JeAtC"
       style="width:100%;height:100%;min-height:100%"></div>
</div>

<!-- Corner accents -->
<div class="corner c-tl"></div>
<div class="corner c-tr"></div>
<div class="corner c-bl"></div>
<div class="corner c-br"></div>

<!-- Top header -->
<div id="hdr">
  <div style="display:flex;align-items:center">
    <div class="hdr-brand">STUDENT AI</div>
    <div class="hdr-sep"></div>
    <div class="hdr-est">EST. 2025</div>
  </div>
  <div class="hdr-coords">
    <span>ML MODEL: ACTIVE</span>
    <div class="hdr-dot"></div>
    <span>FEATURES: 6</span>
    <div class="hdr-dot"></div>
    <span>SAMPLES: 2,000</span>
  </div>
</div>

<!-- Right content panel -->
<div id="panel">
  <div class="deco-line">
    <hr><span class="deco-inf">∞</span><hr>
  </div>

  <div id="main-title">HOW THE<br>AI WORKS</div>

  <div class="dot-row">
    <span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span><span></span>
  </div>

  <div id="desc">
    Six behavioral signals. Two competing models.
    One transparent prediction. Scroll down to see
    how every number is calculated — step by step.
  </div>

  <div class="btns">
    <button class="btn" onclick="document.getElementById('steps-section').scrollIntoView()">
      <span class="corner-tl"></span>
      <span class="corner-br"></span>
      SEE THE STEPS
    </button>
    <button class="btn">
      MODEL: LINEAR REG
    </button>
  </div>

  <div class="bottom-note">
    <span>∞</span><hr><span>PERFORMANCE.PROTOCOL</span>
  </div>
</div>

<!-- Footer bar -->
<div id="ftr">
  <div class="ftr-left">
    <span>SYSTEM.ACTIVE</span>
    <div class="bar-eq">
      <div style="height:6px"></div><div style="height:10px"></div>
      <div style="height:8px"></div><div style="height:14px"></div>
      <div style="height:6px"></div><div style="height:12px"></div>
      <div style="height:9px"></div><div style="height:7px"></div>
    </div>
    <span>V1.0.0</span>
  </div>
  <div class="ftr-right">
    <span>RENDERING</span>
    <div style="width:5px;height:5px" class="pulse"></div>
    <div style="width:5px;height:5px;animation-delay:.25s" class="pulse"></div>
    <div style="width:5px;height:5px;animation-delay:.5s" class="pulse"></div>
    <span>FRAME: ∞</span>
  </div>
</div>

<!-- UnicornStudio loader -->
<script>
!function(){
  if(!window.UnicornStudio){
    window.UnicornStudio={isInitialized:false};
    var s=document.createElement('script');
    s.src='https://cdn.jsdelivr.net/gh/hiunicornstudio/unicornstudio.js@v1.4.33/dist/unicornStudio.umd.js';
    s.onload=function(){
      if(!window.UnicornStudio.isInitialized){
        UnicornStudio.init();
        window.UnicornStudio.isInitialized=true;
      }
    };
    (document.head||document.body).appendChild(s);
  }
}();

// Hide any branding that loads inside the canvas container
var _hb=function(){
  document.querySelectorAll('[data-us-project] *').forEach(function(el){
    var t=(el.textContent||'').toLowerCase();
    var h=(el.getAttribute('href')||'').toLowerCase();
    if(t.includes('made with')||t.includes('unicorn')||h.includes('unicorn.studio')){
      el.style.cssText='display:none!important;visibility:hidden!important;';
      try{el.remove();}catch(e){}
    }
  });
};
[0,500,1500,3000,6000].forEach(function(d){setTimeout(_hb,d);});
</script>
</body>
</html>
"""

with tab5:

    # hero-ascii-one panel at top of How It Works
    components.html(HERO_ASCII_HTML, height=420)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    steps = [
        ("1", "#6366f1", "Data Collection",
         "You enter 6 behavioral inputs: previous test marks, daily study hours, "
         "focused time, revision frequency, distraction level, and subject strength."),
        ("2", "#a78bfa", "Feature Engineering",
         "Raw inputs are transformed into meaningful signals. "
         "<b>Consistency</b> is derived from the standard deviation of your study hours — "
         "low variation means high consistency. "
         "<b>Improvement Trend</b> uses linear regression (polyfit) on your 3 marks to "
         "detect genuine progress vs random fluctuation. All features are normalized to [0,1]."),
        ("3", "#38bdf8", "Machine Learning",
         "Two models are trained on 2,000 synthetic student records (80% train / 20% test):<br>"
         "<b>Linear Regression</b> — simple, explainable straight-line model.<br>"
         "<b>Random Forest</b> — 100 decision trees averaged together, more flexible.<br>"
         "The app automatically selects the better model by R² score."),
        ("4", "#22c55e", "Prediction",
         "The active model predicts your expected marks. A confidence band of ±1.5×MAE "
         "is shown — this represents a realistic range your actual marks might fall within."),
        ("5", "#f59e0b", "Readiness Score",
         "A transparent weighted formula combines your 6 feature scores into a 0–100 "
         "readiness %. This is independent of the ML model — you can verify it by hand "
         "using the breakdown table in the Prediction tab."),
        ("6", "#f97316", "Insights & Suggestions",
         "The app identifies your weakest and strongest feature, then generates specific, "
         "priority-ranked suggestions using your actual score values — not generic advice."),
    ]

    for num, color, title, body in steps:
        st.markdown(f"""
        <div style="display:flex;gap:16px;margin-bottom:18px;align-items:flex-start">
          <div style="min-width:36px;height:36px;border-radius:50%;
               background:{color};display:flex;align-items:center;justify-content:center;
               font-size:.85rem;font-weight:800;color:#0e1117;flex-shrink:0">{num}</div>
          <div>
            <div style="font-size:.95rem;font-weight:700;color:#e2e8f0;margin-bottom:3px">{title}</div>
            <div style="font-size:.82rem;color:#94a3b8;line-height:1.6">{body}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hd">Limitations</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#12172a;border:1px solid rgba(239,68,68,.2);border-radius:12px;padding:18px 20px">
      <ul style="color:#94a3b8;font-size:.83rem;line-height:2;margin:0;padding-left:18px">
        <li>Trained on <b>synthetic data</b> — no real student records were used.</li>
        <li>Human factors (sleep, stress, health, teaching quality) are <b>not captured</b>.</li>
        <li>Active model: <b style="color:#818cf8">{model.best_model_name}</b> ·
            R² ≈ {model.r2:.3f} · MAE ≈ {model.mae:.1f} marks
            (average prediction error).</li>
        <li>Use this as a <b>self-reflection tool</b>, not a definitive exam forecast.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hd" style="margin-top:22px">Glossary</div>',
                unsafe_allow_html=True)
    glossary = [
        ("R² Score", "How well the model explains variance in marks. 1.0 = perfect, 0 = random guessing."),
        ("MAE", "Mean Absolute Error — average distance between predicted and actual marks."),
        ("Feature Importance", "How much each input variable influenced the Random Forest's decisions."),
        ("Confidence Range", "The band (±1.5×MAE) within which actual marks are likely to fall."),
        ("Readiness Score", "A transparent weighted formula reflecting overall exam preparedness."),
        ("Consistency", "Derived from std-dev of study hours — lower daily variation = higher score."),
        ("Improvement Trend", "Linear regression slope on your 3 test marks, normalized to 0–1."),
    ]
    for term, meaning in glossary:
        st.markdown(f"""
        <div style="display:flex;gap:12px;padding:10px 0;border-bottom:1px solid rgba(99,102,241,.1)">
          <div style="min-width:170px;font-size:.82rem;font-weight:700;color:#818cf8">{term}</div>
          <div style="font-size:.82rem;color:#94a3b8">{meaning}</div>
        </div>
        """, unsafe_allow_html=True)
