#!/usr/bin/env python3
"""
BRACU DUBURI Vision Pipeline Automation Dashboard.

Install:
    pip install streamlit plotly pandas pillow requests

Usage:
    streamlit run dashboard.py
"""

import shutil
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BRACU Duburi — Vision Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

N8N_URL = "http://localhost:5678"

# ── Global styles ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #EEF2F7; }
.block-container { padding-top: 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1B3A5C 100%);
    border-right: none;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: #F1F5F9 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }
[data-testid="stSidebar"] .stCaption { color: #64748B !important; }

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0D1B2A 0%, #1B3A5C 50%, #0E4D8C 100%);
    padding: 36px 40px;
    border-radius: 16px;
    margin: 8px 0 28px 0;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(13,27,42,0.25);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; right: -40px;
    width: 360px; height: 360px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 42px;
    color: #FFFFFF;
    letter-spacing: 5px;
    margin: 0 0 6px 0;
    text-shadow: 0 4px 24px rgba(0,0,0,0.4);
    line-height: 1.15;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #60A5FA;
    letter-spacing: 5px;
    text-transform: uppercase;
    margin: 0;
}
.hero-run {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: rgba(255,255,255,0.45);
    margin-top: 14px;
    letter-spacing: 1px;
}

/* ── Metric cards ── */
.metric-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 22px 20px 18px 20px;
    text-align: center;
    box-shadow: 0 2px 16px rgba(13,27,42,0.08);
    border-top: 5px solid #2563EB;
    height: 100%;
}
.metric-card.green  { border-top-color: #059669; }
.metric-card.purple { border-top-color: #7C3AED; }
.metric-card.cyan   { border-top-color: #0891B2; }
.metric-card.amber  { border-top-color: #D97706; }
.metric-card.slate  { border-top-color: #475569; }
.metric-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94A3B8;
    margin-bottom: 10px;
}
.metric-value {
    font-size: 34px;
    font-weight: 900;
    color: #0F172A;
    line-height: 1;
}
.metric-sub {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 6px;
    font-weight: 500;
}

/* ── Section headers ── */
.section-header {
    font-size: 13px;
    font-weight: 800;
    color: #1E3A5F;
    text-transform: uppercase;
    letter-spacing: 3px;
    border-left: 4px solid #2563EB;
    padding: 6px 0 6px 14px;
    margin: 0 0 18px 0;
    background: rgba(37,99,235,0.04);
    border-radius: 0 6px 6px 0;
}

/* ── Pipeline step cards ── */
.step-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(13,27,42,0.07);
    position: relative;
    height: 100%;
}
.step-num {
    width: 28px; height: 28px;
    background: #2563EB;
    color: #fff;
    border-radius: 50%;
    font-size: 12px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px auto;
}
.step-title {
    font-size: 12px;
    font-weight: 800;
    color: #0F172A;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.step-desc {
    font-size: 10px;
    color: #64748B;
    line-height: 1.5;
}

/* ── Gallery nav ── */
.stButton button {
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    border: 2px solid #E2E8F0 !important;
    color: #1E3A5F !important;
    background: #FFFFFF !important;
    transition: all 0.15s ease !important;
}
.stButton button:hover {
    border-color: #2563EB !important;
    color: #2563EB !important;
    background: #EFF6FF !important;
}

/* ── Divider ── */
hr { border-color: #E2E8F0 !important; margin: 28px 0 !important; }

/* ── n8n status ── */
.n8n-online  { color: #059669 !important; font-weight: 800 !important; font-size: 22px !important; }
.n8n-offline { color: #DC2626 !important; font-weight: 800 !important; font-size: 22px !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, sub: str = "", color: str = "") -> None:
    cls = f"metric-card {color}".strip()
    st.markdown(
        f'<div class="{cls}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def load_results(run_dir: Path) -> pd.DataFrame | None:
    csv = run_dir / "results.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    return df


def find_images(run_dir: Path, pattern: str) -> list[Path]:
    return sorted(run_dir.glob(pattern))


SAVED_RUNS_DIR = Path(__file__).parent / "saved_runs"
SAVE_PATTERNS  = [
    "results.csv", "args.yaml",
    "train_batch*.jpg", "train_batch*.png",
    "val_batch*.jpg",   "val_batch*.png",
    "confusion_matrix*.png",
    "BoxPR_curve.png", "PR_curve.png",
    "BoxF1_curve.png", "F1_curve.png",
    "BoxP_curve.png",  "P_curve.png",
    "BoxR_curve.png",  "R_curve.png",
    "labels.jpg",      "labels.png",
    "results.png",
]

def save_run(src: Path, name: str) -> Path:
    dest = SAVED_RUNS_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    for pattern in SAVE_PATTERNS:
        for f in src.glob(pattern):
            shutil.copy2(f, dest / f.name)
    return dest

def saved_run_names() -> list[str]:
    if not SAVED_RUNS_DIR.exists():
        return []
    return sorted(
        [d.name for d in SAVED_RUNS_DIR.iterdir()
         if d.is_dir() and (d / "results.csv").exists()],
        reverse=True,
    )


PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FAFC",
    font=dict(color="#0F172A", size=12, family="Inter, sans-serif"),
    legend=dict(
        bgcolor="#FFFFFF", bordercolor="#E2E8F0", borderwidth=1,
        font=dict(size=12, color="#334155"),
    ),
    margin=dict(l=50, r=20, t=50, b=40),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1E3A5F", font_color="#FFFFFF", font_size=12),
)

COLORS = {
    "train": "#2563EB",
    "val":   "#059669",
    "map50": "#DC2626",
    "map95": "#D97706",
    "prec":  "#7C3AED",
    "rec":   "#0891B2",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## BRACU Duburi")
    st.markdown("<p style='color:#60A5FA;font-size:11px;font-weight:600;letter-spacing:2px'>VISION PIPELINE</p>", unsafe_allow_html=True)
    st.divider()

    # ── Saved runs (no directory needed) ─────────────────────────────────────
    cached = saved_run_names()
    source = "saved"

    if cached:
        st.markdown("<p style='color:#94A3B8;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase'>Saved Runs</p>", unsafe_allow_html=True)
        selected_run = st.selectbox("", cached, key="saved_sel", label_visibility="collapsed")
        run_dir = SAVED_RUNS_DIR / selected_run
        source  = "saved"
        st.divider()

    # ── Import from directory ─────────────────────────────────────────────────
    with st.expander("Import from directory" if cached else "Runs directory", expanded=not cached):
        runs_dir_input = st.text_input(
            "Path",
            value=r"C:\Users\T2520744\An-efficient-object-detection-technique-for-autonomous-vehicles-using-advanced-GAN-model-\runs\detect\runs\train",
            label_visibility="collapsed",
        )
        runs_dir = Path(runs_dir_input)
        dir_runs: list[str] = []
        if runs_dir.exists():
            dir_runs = sorted(
                [d.name for d in runs_dir.iterdir()
                 if d.is_dir() and (d / "results.csv").exists()],
                reverse=True,
            )
        if dir_runs:
            dir_sel = st.selectbox("Run", dir_runs, key="dir_sel")
            if st.button("Load this run", use_container_width=True):
                selected_run = dir_sel
                run_dir      = runs_dir / dir_sel
                source       = "dir"
                st.session_state["active_run"]     = dir_sel
                st.session_state["active_run_dir"] = str(runs_dir / dir_sel)
        else:
            st.caption("No runs found in that path.")

    # Restore from session if set via Load button
    if "active_run" in st.session_state and source == "saved":
        _name = st.session_state["active_run"]
        _dir  = Path(st.session_state["active_run_dir"])
        if _dir.exists() and (_dir / "results.csv").exists():
            selected_run = _name
            run_dir      = _dir
            source       = "dir"

    if not (run_dir.exists() and (run_dir / "results.csv").exists()):
        st.warning("No run selected. Import a run from a directory or save one first.")
        st.stop()

    # ── Save button ───────────────────────────────────────────────────────────
    if source == "dir":
        st.divider()
        if st.button("Save run (no weights)", use_container_width=True, type="primary"):
            save_run(run_dir, selected_run)
            st.success(f"Saved '{selected_run}'")
            st.rerun()

    st.divider()
    auto_refresh = st.toggle("Live training mode", value=False)
    refresh_sec  = 30
    if auto_refresh:
        refresh_sec = st.slider("Refresh interval (s)", 5, 120, 30, 5)

    st.divider()
    st.caption(f"Active run: {selected_run}")
    st.caption(f"Source: {'local cache' if source == 'saved' else 'directory'}")


# ── Load data ─────────────────────────────────────────────────────────────────

df = load_results(run_dir)
if df is None:
    st.error(f"results.csv not found in `{run_dir}`")
    st.stop()

map50_col = next((c for c in df.columns if "mAP50" in c and "95" not in c), None)
map95_col = next((c for c in df.columns if "mAP50-95" in c), None)
prec_col  = next((c for c in df.columns if "precision" in c.lower()), None)
rec_col   = next((c for c in df.columns if "recall" in c.lower()), None)

best_map50 = df[map50_col].max()       if map50_col else 0
best_map95 = df[map95_col].max()       if map95_col else 0
best_epoch = int(df[map50_col].idxmax() + 1) if map50_col else 0
best_prec  = df[prec_col].max()        if prec_col  else 0
best_rec   = df[rec_col].max()         if rec_col   else 0
total_ep   = len(df)
epochs     = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)


# ── Hero banner ───────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">BRACU DUBURI<br>VISION PIPELINE AUTOMATION</div>
    <div class="hero-sub">Autonomous Underwater Vehicle &nbsp;|&nbsp; Object Detection System</div>
    <div class="hero-run">Run &nbsp;/&nbsp; {selected_run} &nbsp;&nbsp;|&nbsp;&nbsp; Best mAP50 &nbsp;{best_map50:.3f} &nbsp;@ epoch {best_epoch}</div>
</div>
""", unsafe_allow_html=True)


# ── Metric cards ──────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
with c1: metric_card("mAP50",     f"{best_map50:.3f}", f"epoch {best_epoch}", "")
with c2: metric_card("mAP50-95",  f"{best_map95:.3f}", f"epoch {best_epoch}", "green")
with c3: metric_card("Precision", f"{best_prec:.3f}",  "best val",            "purple")
with c4: metric_card("Recall",    f"{best_rec:.3f}",   "best val",            "cyan")
with c5: metric_card("Epochs",    str(total_ep),        "completed",           "amber")

st.markdown("<br>", unsafe_allow_html=True)


# ── Training curves ───────────────────────────────────────────────────────────

section("Training Curves")
curve_col1, curve_col2 = st.columns(2)

with curve_col1:
    box_tr = next((c for c in df.columns if "box_loss" in c and "train" in c.lower()), None) or \
             next((c for c in df.columns if "box_loss" in c), None)
    cls_tr = next((c for c in df.columns if "cls_loss" in c and "train" in c.lower()), None) or \
             next((c for c in df.columns if "cls_loss" in c), None)
    dfl_tr = next((c for c in df.columns if "dfl_loss" in c and "train" in c.lower()), None) or \
             next((c for c in df.columns if "dfl_loss" in c), None)
    box_vl = next((c for c in df.columns if "box_loss" in c and "val" in c.lower()), None)
    cls_vl = next((c for c in df.columns if "cls_loss" in c and "val" in c.lower()), None)
    dfl_vl = next((c for c in df.columns if "dfl_loss" in c and "val" in c.lower()), None)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Box Loss", "Class Loss", "DFL Loss"),
        vertical_spacing=0.09,
    )
    for row, (tr_col, vl_col) in enumerate(
        [(box_tr, box_vl), (cls_tr, cls_vl), (dfl_tr, dfl_vl)], start=1
    ):
        if tr_col and tr_col in df.columns:
            fig.add_trace(go.Scatter(
                x=epochs, y=df[tr_col], name="Train",
                line=dict(color=COLORS["train"], width=2.5),
                showlegend=(row == 1),
            ), row=row, col=1)
        if vl_col and vl_col in df.columns:
            fig.add_trace(go.Scatter(
                x=epochs, y=df[vl_col], name="Val",
                line=dict(color=COLORS["val"], width=2.5, dash="dot"),
                showlegend=(row == 1),
            ), row=row, col=1)

    fig.update_layout(height=480, title_text="<b>Loss Curves</b>", **PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#E2E8F0", linecolor="#CBD5E1", title_text="Epoch", row=3, col=1)
    fig.update_yaxes(gridcolor="#E2E8F0", linecolor="#CBD5E1")
    st.plotly_chart(fig, use_container_width=True)

with curve_col2:
    fig2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("mAP Scores", "Precision & Recall"),
        vertical_spacing=0.12,
    )
    if map50_col and map50_col in df.columns:
        fig2.add_trace(go.Scatter(
            x=epochs, y=df[map50_col], name="mAP50",
            line=dict(color=COLORS["map50"], width=2.5),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.06)",
        ), row=1, col=1)
    if map95_col and map95_col in df.columns:
        fig2.add_trace(go.Scatter(
            x=epochs, y=df[map95_col], name="mAP50-95",
            line=dict(color=COLORS["map95"], width=2.5),
            fill="tozeroy", fillcolor="rgba(217,119,6,0.06)",
        ), row=1, col=1)
    if prec_col and prec_col in df.columns:
        fig2.add_trace(go.Scatter(
            x=epochs, y=df[prec_col], name="Precision",
            line=dict(color=COLORS["prec"], width=2.5),
        ), row=2, col=1)
    if rec_col and rec_col in df.columns:
        fig2.add_trace(go.Scatter(
            x=epochs, y=df[rec_col], name="Recall",
            line=dict(color=COLORS["rec"], width=2.5),
        ), row=2, col=1)

    if map50_col:
        best_ep_idx = int(df[map50_col].idxmax())
        x_val = epochs.iloc[best_ep_idx] if hasattr(epochs, "iloc") else best_ep_idx + 1
        fig2.add_vline(
            x=x_val, line_dash="dash", line_color="#94A3B8",
            annotation_text="Best", annotation_font_color="#475569",
            annotation_font_size=11,
        )

    fig2.update_layout(height=480, title_text="<b>Metrics</b>", **PLOTLY_LAYOUT)
    fig2.update_xaxes(gridcolor="#E2E8F0", linecolor="#CBD5E1", title_text="Epoch", row=2, col=1)
    fig2.update_yaxes(gridcolor="#E2E8F0", linecolor="#CBD5E1")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()


# ── Detection images ──────────────────────────────────────────────────────────

section("Detection Images")

def image_gallery(container, title: str, images: list[Path], key: str) -> None:
    with container:
        st.markdown(f"<p style='font-size:13px;font-weight:700;color:#1E3A5F;margin-bottom:8px'>{title} <span style='font-weight:400;color:#94A3B8'>({len(images)})</span></p>", unsafe_allow_html=True)
        if not images:
            st.info("No images found.")
            return
        idx_key = f"{key}_idx"
        if idx_key not in st.session_state:
            st.session_state[idx_key] = 0
        idx = max(0, min(st.session_state[idx_key], len(images) - 1))

        nav1, nav2, nav3 = st.columns([1, 5, 1])
        with nav1:
            if st.button("←", key=f"{key}_prev", disabled=(idx == 0)):
                st.session_state[idx_key] = idx - 1
                st.rerun()
        with nav3:
            if st.button("→", key=f"{key}_next", disabled=(idx == len(images) - 1)):
                st.session_state[idx_key] = idx + 1
                st.rerun()
        with nav2:
            st.caption(f"{images[idx].name}  ({idx + 1} / {len(images)})")

        try:
            st.image(Image.open(images[idx]), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load image: {e}")


train_imgs = find_images(run_dir, "train_batch*.jpg") or find_images(run_dir, "train_batch*.png")
val_pred   = find_images(run_dir, "val_batch*pred*.jpg") or find_images(run_dir, "val_batch*pred*.png")
val_label  = find_images(run_dir, "val_batch*label*.jpg") or find_images(run_dir, "val_batch*label*.png")

img_col1, img_col2 = st.columns(2)
image_gallery(img_col1, "Train Batches",   train_imgs, "train_imgs")
image_gallery(img_col2, "Val Predictions", val_pred,   "val_pred")

if val_label:
    st.markdown("<br>", unsafe_allow_html=True)
    section("Val Labels vs Predictions")
    lb_col, pr_col = st.columns(2)
    image_gallery(lb_col, "Ground Truth Labels", val_label, "val_label")
    image_gallery(pr_col, "Model Predictions",   val_pred,  "val_pred2")

st.divider()


# ── Evaluation plots ──────────────────────────────────────────────────────────

section("Evaluation Plots")

eval_images = {
    "Confusion Matrix": find_images(run_dir, "confusion_matrix_normalized.png") or find_images(run_dir, "confusion_matrix.png"),
    "PR Curve":         find_images(run_dir, "BoxPR_curve.png") or find_images(run_dir, "PR_curve.png"),
    "F1 Curve":         find_images(run_dir, "BoxF1_curve.png") or find_images(run_dir, "F1_curve.png"),
    "P Curve":          find_images(run_dir, "BoxP_curve.png")  or find_images(run_dir, "P_curve.png"),
    "R Curve":          find_images(run_dir, "BoxR_curve.png")  or find_images(run_dir, "R_curve.png"),
    "Labels":           find_images(run_dir, "labels.jpg")      or find_images(run_dir, "labels.png"),
}

eval_cols = st.columns(len(eval_images))
for col, (title, imgs) in zip(eval_cols, eval_images.items()):
    with col:
        st.markdown(f"<p style='font-size:11px;font-weight:700;color:#475569;text-align:center;text-transform:uppercase;letter-spacing:1px'>{title}</p>", unsafe_allow_html=True)
        if imgs:
            try:
                st.image(Image.open(imgs[0]), use_container_width=True)
            except Exception:
                st.info("Could not load.")
        else:
            st.markdown("<p style='color:#CBD5E1;text-align:center;font-size:12px'>Not found</p>", unsafe_allow_html=True)

st.divider()


# ── n8n Pipeline Section ─────────────────────────────────────────────────────

section("Annotation Pipeline — Automated by n8n")

PIPELINE_STEPS = [
    ("01", "Split Video",     ""),
    ("02", "Send Emails",     ""),
    ("03", "Annotate",        ""),
    ("04", "Receive Zips",    ""),
    ("05", "Merge & Split",   ""),
    ("06", "Train",           ""),
]

step_cols = st.columns(len(PIPELINE_STEPS))
for col, (num, title, desc) in zip(step_cols, PIPELINE_STEPS):
    with col:
        st.markdown(
            f'<div class="step-card">'
            f'<div class="step-num">{num}</div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

try:
    import requests as _req
    resp = _req.get(f"{N8N_URL}/healthz", timeout=2)
    n8n_online = resp.ok
except Exception:
    n8n_online = False

status_color = "#059669" if n8n_online else "#DC2626"
status_text  = "Online"  if n8n_online else "Offline"
status_bg    = "#F0FDF4" if n8n_online else "#FEF2F2"
status_bdr   = "#BBF7D0" if n8n_online else "#FECACA"

n8n_left, n8n_right = st.columns([4, 1])
with n8n_left:
    st.markdown(
        f'<div style="background:{status_bg};border:1.5px solid {status_bdr};border-radius:12px;padding:18px 24px;display:flex;align-items:center;gap:16px">'
        f'<div style="width:12px;height:12px;border-radius:50%;background:{status_color};flex-shrink:0;box-shadow:0 0 8px {status_color}88"></div>'
        f'<div>'
        f'<div style="font-size:13px;font-weight:800;color:{status_color}">n8n {status_text}</div>'
        f'<div style="font-size:11px;color:#64748B;margin-top:2px">{N8N_URL} &nbsp;|&nbsp; Workflow automation engine</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with n8n_right:
    st.link_button("Open n8n Editor", N8N_URL, use_container_width=True)

if n8n_online:
    st.markdown("<br>", unsafe_allow_html=True)
    st.components.v1.iframe(N8N_URL, height=540, scrolling=True)

st.markdown(
    "<p style='text-align:center;color:#CBD5E1;font-size:11px;font-weight:600;letter-spacing:2px;margin-top:40px'>"
    "BRACU DUBURI &nbsp;|&nbsp; VISION PIPELINE AUTOMATION &nbsp;|&nbsp; AUV OBJECT DETECTION"
    "</p>",
    unsafe_allow_html=True,
)

# ── Auto-refresh ──────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
