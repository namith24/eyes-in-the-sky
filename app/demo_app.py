"""Eyes in the Sky: ground-station web UI for the UAV Survivor Detection capstone.

Run from the project root:
    streamlit run app/demo_app.py

Pages: Dashboard · Live Drone · Survivors · Results · Story
(The previous single-page tab layout is kept in app/demo_app_v1.py.)
"""

import contextlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # field use: the laptop is on the drone's WiFi with no internet

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from uav_survivor_detection.models import custom_yolo  # noqa: E402,F401  (registers CBAM)
from uav_survivor_detection.inference.drone_link import DroneLink, discover, list_windows, on_drone_wifi  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# ============================================================================ data & models
TAG = "final_comparison"
RES_DIR = ROOT / "results" / TAG
LOGS = ROOT / "logs"
WEIGHTS = {"baseline": ROOT / "runs" / TAG / "baseline" / "weights" / "best.pt",
           "enhanced": ROOT / "runs" / TAG / "enhanced" / "weights" / "best.pt"}
RESULTS = RES_DIR / "inference_variants_test.json"
SELECTION = RES_DIR / "selection.json"
TEST_IMAGES = ROOT / "data" / "datasets" / "new_dataset3_subset" / "test" / "images"
if not TEST_IMAGES.exists():  # GitHub / cloud copy ships a 40-image sample of the test set
    TEST_IMAGES = ROOT / "data" / "demo_test" / "images"
TEST_LABELS = TEST_IMAGES.parent / "labels"
ON_CLOUD = sys.platform != "win32"  # the ground-station laptop is Windows; Streamlit Cloud is Linux


def _proposed():
    """Proposed configuration chosen on the validation set (results/final_comparison/selection.json)."""
    try:
        m, v = json.loads(SELECTION.read_text())["proposed"].split(":")
        kind, size = v.split("@")
        return WEIGHTS[m], int(size), kind == "tta"
    except Exception:
        return WEIGHTS["baseline"], 800, False


_pp, _psz, _ptta = _proposed()
MODELS = {  # name -> (weights, inference image size, test-time augmentation)
    "Our detector: small-object aware (proposed)": (_pp, _psz, _ptta),
    "Standard YOLO11n (baseline)": (WEIGHTS["baseline"], 480, False),
    "Everyday person-spotter (close range)": (ROOT / "yolo11n.pt", 640, False),
}
ENHANCED, BASELINE, EVERYDAY = list(MODELS)
AUTO = "Auto: up close + from the air (recommended for live)"

RES = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
SEL = json.loads(SELECTION.read_text()) if SELECTION.exists() else {}
HAVE_RESULTS = "baseline:full@480" in RES and SEL.get("proposed") in RES
A, T = "all", "tiny (<10px)"

PAGES = ["Dashboard", "Live Drone", "Survivors", "Results", "Story"]
RAIL = [("Dashboard", ":material/home:"), ("Live Drone", ":material/flight:"),
        ("Survivors", ":material/person_search:"), ("Results", ":material/donut_large:"),
        ("Story", ":material/menu_book:")]

# ============================================================================ state
st.set_page_config(page_title="Eyes in the Sky", page_icon="🛸", layout="wide", initial_sidebar_state="collapsed")
ss = st.session_state
ss.setdefault("nav", "Dashboard")
# ---- settings that survive restarts (avatar menu): app/ui_settings.json
SETTINGS_FILE = Path(__file__).parent / "ui_settings.json"
SETTINGS_DEFAULTS = {"team": "Rescue Team", "theme": "light", "live_model": "", "live_conf": 0.5,
                     "save_shots": True, "notif_seen": 0.0, "intro": True}


def load_settings():
    try:
        return {**SETTINGS_DEFAULTS, **json.loads(SETTINGS_FILE.read_text())}
    except Exception:
        return dict(SETTINGS_DEFAULTS)


def save_settings(**changes):
    cfg = {**load_settings(), **changes}
    try:
        SETTINGS_FILE.write_text(json.dumps(cfg, indent=2))
    except OSError:
        pass
    return cfg


CFG = load_settings()
ss.setdefault("theme", CFG["theme"])
# Drone settings live in plain (non-widget) keys so they survive page switches.
ss.setdefault("drone_mode", "wifi")
ss.setdefault("drone_url", "http://192.168.4.1/stream")
ss.setdefault("drone_stream_url", "")

# A rerun abandons any running live loop: close its reader thread.
_old = ss.pop("_drone_link", None)
if _old is not None:
    _old.stop(join=0.5)

# ============================================================================ theme
THEMES = {
    "light": dict(
        bg="linear-gradient(135deg,#F9D7E2 0%,#F4E3EE 45%,#E5E1F8 100%)",
        panel="rgba(255,255,255,.58)", panel_border="rgba(255,255,255,.75)",
        card="rgba(255,255,255,.86)", card_border="rgba(255,255,255,.9)",
        text="#141414", muted="#8D8896", line="rgba(20,20,20,.07)",
        ink="#111111", ink_text="#FFFFFF", chip="#FFFFFF",
        lav="#B8B3F3", pink="#F4A3BB", lav_soft="#ECEBFD", pink_soft="#FDE7EE",
        gpink="linear-gradient(140deg,#FFE9F0 0%,#F8BED0 100%)",
        glav="linear-gradient(140deg,#EEEDFE 0%,#C8C4F7 100%)",
        shadow="rgba(150,80,110,.10)", input="#FFFFFF",
        hero="#161418", hero_text="#F7F2F4", hero_muted="rgba(247,242,244,.6)", chg="#1B1B1B"),
    "dark": dict(
        bg="linear-gradient(135deg,#1A1620 0%,#251B27 50%,#1B1A2C 100%)",
        panel="rgba(36,32,43,.66)", panel_border="rgba(255,255,255,.06)",
        card="rgba(50,45,58,.88)", card_border="rgba(255,255,255,.05)",
        text="#F4F1F6", muted="#A7A1B2", line="rgba(255,255,255,.08)",
        ink="#F4F1F6", ink_text="#141414", chip="rgba(255,255,255,.09)",
        lav="#9089EC", pink="#E8869F", lav_soft="#36334F", pink_soft="#4A2E3B",
        gpink="linear-gradient(140deg,#4E2D3B 0%,#74405A 100%)",
        glav="linear-gradient(140deg,#2E2B4E 0%,#4A4485 100%)",
        shadow="rgba(0,0,0,.35)", input="rgba(255,255,255,.07)",
        hero="linear-gradient(165deg,#3B3566 0%,#56304C 100%)", hero_text="#FBF6F8", hero_muted="rgba(251,246,248,.62)", chg="#FBF6F8"),
}
TH = THEMES[ss.theme]

# Fonts ship with the app (app/static/fonts, served by Streamlit static serving), so they work offline.
_FONT_CSS = Path(__file__).parent / "static" / "fonts" / "fonts.css"
FONTS = _FONT_CSS.read_text() if _FONT_CSS.exists() else ""
CSS = """
:root { --sans: 'Geist', 'Segoe UI', system-ui, sans-serif; --serif: 'Instrument Serif', Georgia, serif; --mono: 'Geist Mono', ui-monospace, monospace; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp div,
.stApp span:not([data-testid="stIconMaterial"]):not([data-testid="stExpanderIcon"]), .stApp [data-baseweb] { font-family: var(--sans) !important; }
.stApp [data-testid="stExpanderIcon"] { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: var(--bg); background-attachment: fixed; color: var(--text); }
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stSidebarCollapsedControl"] { display:none !important; }
.block-container { max-width: 1320px; margin: 26px auto 40px; padding: 22px 30px 30px !important;
  background: var(--panel); backdrop-filter: blur(26px); -webkit-backdrop-filter: blur(26px);
  border-radius: 36px; border: 1px solid var(--panel_border); box-shadow: 0 40px 90px var(--shadow); }
h1, h2, h3, h4 { font-family: var(--sans) !important; color: var(--text) !important; letter-spacing:-.01em; font-weight:500 !important; }
.stMarkdown, .stMarkdown p, label, [data-testid="stWidgetLabel"] p, .stCaption, [data-testid="stCaptionContainer"] { color: var(--text) !important; }
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
[data-testid="stImage"] img { border-radius: 18px; }
[data-testid="stVerticalBlock"] { gap: .9rem; }

/* ---------- cards ---------- */
div[class*="st-key-card"] { background: var(--card); border: 1px solid var(--card_border); border-radius: 26px;
  padding: 20px 22px 22px; box-shadow: 0 10px 34px var(--shadow); }
div[class*="st-key-gpink"], div[class*="st-key-glav"] { border-radius: 22px; padding: 18px 18px 16px; }
div[class*="st-key-gpink"] { background: var(--gpink); }
div[class*="st-key-glav"] { background: var(--glav); }
.hcard { background: var(--card); border: 1px solid var(--card_border); border-radius: 26px; padding: 22px;
  box-shadow: 0 10px 34px var(--shadow); height: 100%; color: var(--text); }
.hcard .ico { width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center;
  font-size:1.35rem; background: var(--lav_soft); margin-bottom: 12px; }
.hcard.pink .ico { background: var(--pink_soft); }
.hcard h4 { margin: 0 0 6px; font-size: 1.15rem; }
.hcard p { margin: 0; color: var(--muted); font-size: .98rem; line-height: 1.45; }
.stMarkdown p.ctitle { font-size: 1.45rem !important; font-weight: 500; color: var(--text) !important; line-height: 1.15; margin: 0 !important; }
.stMarkdown p.csub, .stMarkdown span.csub { font-size: .84rem !important; color: var(--muted) !important; margin: 3px 0 0 !important; line-height: 1.35; }
.stMarkdown p.gtitle { font-size: 1.3rem !important; font-weight: 500; color: var(--text) !important; line-height: 1.2; margin: 0 !important; }
.stMarkdown p.gsub { font-size: .9rem !important; color: var(--text) !important; opacity: .7; margin: 6px 0 4px !important; line-height: 1.35; }
.stMarkdown p.greet-sub { color: var(--muted) !important; }
.hcard p { color: var(--muted) !important; }

/* ---------- header ---------- */
.logo { width: 52px; height: 52px; border-radius: 16px; background: #111; display:flex; align-items:center;
  justify-content:center; box-shadow: 0 8px 20px rgba(0,0,0,.25); }
.hicons { display:flex; justify-content:flex-end; gap: 10px; align-items:center; }
.hicon { width: 46px; height: 46px; border-radius: 15px; background: var(--chip); display:flex; align-items:center;
  justify-content:center; position: relative; color: var(--text); }
.hicon .dot { position:absolute; top: 11px; right: 12px; width: 7px; height: 7px; border-radius: 50%; background:#F0446B; }
.avatar { width: 48px; height: 48px; border-radius: 15px; background: linear-gradient(135deg,#F7B3C7,#B7B2F2);
  display:flex; align-items:center; justify-content:center; font-weight:600; color:#2a1d2a; }
.greet { font-size: 2.6rem; font-weight: 400; letter-spacing: -.03em; color: var(--text); margin: 6px 0 2px; }
.greet-sub { color: var(--muted); margin: 0 0 6px; font-size: 1rem; }

/* ---------- top pill navigation ---------- */
.st-key-nav [data-testid="stButtonGroup"], .st-key-nav > div { display:flex; justify-content:center; }
.st-key-nav [data-testid="stButtonGroup"] > div { background: var(--chip); border-radius: 18px; padding: 5px; gap: 2px; }
.st-key-nav button { border: none !important; background: transparent !important; border-radius: 14px !important;
  padding: 10px 22px !important; color: var(--text) !important; box-shadow: none !important; }
.st-key-nav button p { font-size: .95rem !important; font-weight: 500; }
.st-key-nav button[data-testid="stBaseButton-segmented_controlActive"] { background: var(--ink) !important;
  color: var(--ink_text) !important; box-shadow: 0 6px 16px rgba(0,0,0,.22) !important; }
.st-key-nav button[data-testid="stBaseButton-segmented_controlActive"] p { color: var(--ink_text) !important; }

/* small segmented controls inside cards (Week / Month style) */
div[class*="st-key-seg"] [data-testid="stButtonGroup"] { display:flex; justify-content:flex-end; }
div[class*="st-key-seg"] button { border: none !important; background: var(--chip) !important; border-radius: 13px !important;
  padding: 7px 16px !important; color: var(--text) !important; }
div[class*="st-key-seg"] button[data-testid="stBaseButton-segmented_controlActive"] { background: var(--ink) !important; }
div[class*="st-key-seg"] button[data-testid="stBaseButton-segmented_controlActive"] p { color: var(--ink_text) !important; }

/* ---------- left icon rail ---------- */
div[class*="st-key-rail"] button { width: 48px !important; height: 48px; min-height: 48px; padding: 0 !important;
  border-radius: 15px !important; border: none !important; background: var(--chip) !important; color: var(--text) !important;
  box-shadow: 0 4px 14px var(--shadow); }
div[class*="st-key-rail"] button[data-testid="stBaseButton-primary"] { background: #111 !important; color: #fff !important; }
div[class*="st-key-rail"] [data-testid="stVerticalBlock"] { gap: .55rem; align-items: center; }
.block-container { position: relative; padding-left: 104px !important; }
.st-key-rail_nav, .st-key-rail_theme { position: fixed !important; left: 26px; width: 52px !important; z-index: 5; gap: .55rem; }
.st-key-rail_nav { top: 196px; }
.st-key-rail_theme { bottom: 30px; }
@media (max-width: 760px) {
  .block-container { padding-left: 18px !important; padding-right: 18px !important; margin: 8px; border-radius: 26px; }
  .st-key-rail_nav, .st-key-rail_theme { display: none !important; }
  .greet { font-size: 2rem; }
}

/* ---------- buttons ---------- */
button[data-testid="stBaseButton-primary"] { background: var(--ink) !important; color: var(--ink_text) !important;
  border: none !important; border-radius: 13px !important; }
button[data-testid="stBaseButton-primary"] p { color: var(--ink_text) !important; }
button[data-testid="stBaseButton-secondary"] { background: var(--chip) !important; color: var(--text) !important;
  border: 1px solid var(--line) !important; border-radius: 13px !important; }
div[class*="st-key-gpink"] button[data-testid="stBaseButton-primary"],
div[class*="st-key-glav"] button[data-testid="stBaseButton-primary"] { background:#111 !important; }
div[class*="st-key-gpink"] button[data-testid="stBaseButton-primary"] p,
div[class*="st-key-glav"] button[data-testid="stBaseButton-primary"] p { color:#fff !important; }

/* ---------- inputs ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"], .stTextInput input, .stNumberInput input {
  background: var(--input) !important; border-radius: 13px !important; border-color: var(--line) !important; color: var(--text) !important; }
[data-testid="stExpander"] details { background: var(--chip); border-radius: 16px; border: 1px solid var(--line); }
[data-testid="stFileUploaderDropzone"] { background: var(--chip); border-radius: 16px; }

/* ---------- small pieces ---------- */
.chip { display:inline-flex; align-items:center; gap:7px; background: var(--chip); color: var(--text); border-radius: 12px;
  padding: 8px 12px; font-size: .85rem; font-weight: 500; white-space: nowrap; }
.cstat { float:right; display:inline-flex; align-items:center; gap:6px; padding: 5px 10px; border-radius: 99px;
  background: rgba(255,255,255,.78); color:#1b1b1b; font-size:.72rem; font-family: var(--mono) !important; }
.cstat i { width: 8px; height: 8px; border-radius: 50%; background: #9A95A3; }
.cstat.on i { background: #2FB574; box-shadow: 0 0 0 3px rgba(47,181,116,.18); }
/* header controls: logo button + popovers styled as the design's icon tiles */
.st-key-logo { position: relative; width: 52px; }
.st-key-logo .stButton { position: absolute; inset: 0; margin: 0; }
.st-key-logo .stButton button { width: 52px !important; height: 52px; opacity: 0; cursor: pointer; }
.st-key-hdr_row { flex-direction: row !important; justify-content: flex-end; gap: 10px !important; flex-wrap: nowrap; }
.st-key-hdr_row > div { width: auto !important; flex: 0 0 auto !important; }
div[class*="st-key-hdr"] { align-items: flex-end; }
div[class*="st-key-hdr"] button { width: 48px !important; height: 48px; min-height: 48px; border-radius: 15px !important;
  border: none !important; background: var(--chip) !important; color: var(--text) !important; padding: 0 !important;
  box-shadow: none !important; justify-content: center; }
div[class*="st-key-hdr"] button [data-testid="stIconMaterial"] { font-size: 1.3rem; }
div[class*="st-key-hdr"] button > div:has(> svg) { display: none !important; }  /* hide popover chevron */
[data-testid="stPopoverBody"]:has(.notif-list) { max-height: 70vh; overflow-y: auto; }
.st-key-hdr_avatar button { background: linear-gradient(135deg,#F7B3C7,#B7B2F2) !important; color: #2a1d2a !important; }
.st-key-hdr_avatar button p { font-weight: 600; color: #2a1d2a !important; }
.st-key-hdr_bell_unread { position: relative; }
.st-key-hdr_bell_unread::after { content: ""; position: absolute; top: 11px; right: 12px; width: 8px; height: 8px;
  border-radius: 50%; background: #F0446B; box-shadow: 0 0 0 2px var(--chip); pointer-events: none; }
[data-testid="stPopoverBody"] { border-radius: 18px !important; min-width: 320px; }
.bell { float:right; width: 34px; height: 34px; border-radius: 11px; background: rgba(255,255,255,.75);
  display:flex; align-items:center; justify-content:center; font-size: .95rem; }
.pillstat { background: linear-gradient(140deg,#FFE3EC,#F7B8CB); border-radius: 18px; padding: 10px 18px; display:inline-block;
  box-shadow: 0 10px 24px rgba(240,140,170,.28); color:#1b1b1b; }
.pillstat .n { font-size: 1.9rem; font-weight: 400; line-height: 1.05; }
.pillstat .l { font-size: .74rem; opacity: .8; }
.legend { display:flex; gap: 16px; font-size: .88rem; color: var(--text); margin: 4px 0 2px; }
.legend i { display:inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }
.pbar-row { display:flex; justify-content: space-between; font-size: .9rem; color: var(--text); margin: 14px 0 7px; }
.pbar { height: 34px; border-radius: 11px; overflow: hidden; }
.pbar > div { height: 100%; border-radius: 11px; }
.score { display:flex; align-items:center; gap: 12px; margin: 12px 0; }
.score .av { width: 42px; height: 42px; border-radius: 13px; display:flex; align-items:center; justify-content:center; font-size: 1.2rem; flex: none; }
.score .mid { flex: 1; }
.score .nm { font-size: .9rem; color: var(--text); margin-bottom: 7px; }
.score .tr { height: 5px; border-radius: 99px; background: var(--line); }
.score .tr > div { height: 100%; border-radius: 99px; background: var(--ink); }
.score .pc { font-size: 1.9rem; font-weight: 300; color: var(--text); min-width: 82px; text-align: right; }
.big { font-size: 3rem; font-weight: 300; color: var(--text); line-height: 1; letter-spacing:-.03em; }
.delta { display:inline-block; margin-top: 10px; border-radius: 10px; padding: 4px 10px; font-size: .82rem; font-weight: 500;
  background: var(--lav_soft); color: var(--text); }
.delta.neg { background: var(--pink_soft); }
.live-pill { display:inline-block; color:#fff; border-radius: 11px; padding: 5px 12px; font-weight: 600; font-size: .8rem; }
.live-pill.blink { animation: blink 1.4s infinite; }
@keyframes blink { 50% { opacity: .5; } }
.waiting { border: 2px dashed var(--line); border-radius: 22px; padding: 46px 20px; text-align: center; color: var(--muted); }
.waiting .e { font-size: 2.6rem; }
.quote { font-size: 1.35rem; font-weight: 300; color: var(--text); line-height: 1.35; }

/* ================= fixed app shell: the page never scrolls ================= */
html, body, .stApp, [data-testid="stAppViewContainer"] { height: 100vh; overflow: hidden !important; }
section[data-testid="stMain"] { overflow: hidden !important; height: 100vh; }
.block-container { height: calc(100vh - 32px); margin: 16px auto !important; padding-top: 16px !important;
  padding-bottom: 16px !important; overflow: hidden; display: flex; flex-direction: column; }
.block-container > div[data-testid="stVerticalBlockBorderWrapper"] { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
.block-container > div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] { flex: 1 1 auto; min-height: 0; flex-wrap: nowrap; }
.block-container > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] { flex: 0 0 auto !important; }
/* page area fills what is left under the header; content is sized to fit it */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> .st-key-mainwrap) { flex: 1 1 0; min-height: 0; overflow: hidden auto;
  scrollbar-width: none; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(> .st-key-mainwrap)::-webkit-scrollbar { display: none; }
.st-key-mainwrap { gap: .8rem; }
.st-key-mainwrap [data-testid="stVerticalBlock"] { gap: .8rem; }
.greet { font-size: clamp(1.7rem, 4.2vh, 2.5rem); margin: 0; line-height: 1.1; }
.stMarkdown p.greet-sub { margin: 4px 0 0 !important; font-size: .95rem; }
div[class*="st-key-card"] { padding: 16px 20px 18px; }
/* live video, test photos and thumbnails scale with the window height */
.st-key-card_live [data-testid="stImage"] img { max-height: calc(100vh - 545px); object-fit: contain; }
.st-key-card_live .waiting { padding: 0 20px; height: calc(100vh - 500px); display:flex; flex-direction:column; justify-content:center; }
.st-key-card_sightings [data-testid="stImage"] img { height: 7vh; object-fit: cover; }
.st-key-card_sightings [data-testid="stImage"] { gap: 0; }
.st-key-card_gt [data-testid="stImage"] img, .st-key-card_std [data-testid="stImage"] img,
.st-key-card_ours [data-testid="stImage"] img { max-height: calc(100vh - 515px); object-fit: contain; }
.hcard.compact { padding: 16px 18px; }
.hcard.compact .head { display:flex; align-items:center; gap: 10px; margin-bottom: 8px; }
.hcard.compact .ico { width: 36px; height: 36px; border-radius: 11px; font-size: 1.05rem; margin: 0; flex: none; }
.hcard.compact h4 { margin: 0; font-size: 1.02rem; }
.hcard.compact p { font-size: .9rem; line-height: 1.4; }
.pbar-row { margin: 10px 0 6px; }
.pbar { height: 24px; }
.score { margin: 8px 0; }
.score .av { width: 38px; height: 38px; }
.score .pc { font-size: 1.6rem; }
/* short laptop screens: drop the page sub-heading line */
@media (max-height: 820px) { .stMarkdown p.greet-sub { display: none; } .block-container { padding-top: 12px !important; } }
/* ================= type ================= */
.stApp .greet, .stApp .greet em, .stMarkdown p.ctitle, .stMarkdown p.gtitle, .stApp .quote, .stApp .big, .stApp .pillstat .n,
.stApp .score .pc, .stApp .s-serif, .stApp .s-serif em {
  font-family: var(--serif) !important; font-weight: 400 !important; letter-spacing: -.01em; }
.greet { font-size: clamp(2.1rem, 5.4vh, 3.1rem); }
.greet em, .s-serif em { font-style: italic; }
.stMarkdown p.ctitle { font-size: 1.7rem !important; }
.stMarkdown p.gtitle { font-size: 1.55rem !important; }
.big { font-size: 3.4rem; }
.score .pc { font-size: 2rem; }
.pillstat .n { font-size: 2.1rem; }
.stMarkdown p.csub, .stApp .chip, .stApp .delta, .stApp .live-pill, .stApp .s-kicker, .stApp .s-mono, .stApp .s-legend span {
  font-family: var(--mono) !important; letter-spacing: .01em; }
.stMarkdown p.csub { font-size: .76rem !important; text-transform: none; }
.st-key-nav button p { font-family: var(--sans) !important; }

/* the panel must never scroll, not even when the browser scrolls a clicked control into view */
section[data-testid="stMain"], .block-container { overflow: clip !important; }

/* ================= story page ================= */
.story { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.75fr); grid-template-rows: auto minmax(0, 1fr);
  gap: 14px; height: calc(100vh - 248px); min-height: 440px; }
.s-card { border-radius: 26px; padding: 22px 24px; position: relative; overflow: hidden; }
.s-change .s-kicker { padding-right: 70px; }
.s-kicker { font-size: .72rem; text-transform: uppercase; letter-spacing: .14em; opacity: .65; }
.s-hero { grid-row: 1 / span 2; background: var(--hero); color: var(--hero_text); display: flex; flex-direction: column;
  box-shadow: 0 24px 50px var(--shadow); }
.s-hero .s-serif { font-size: clamp(2.1rem, 5vh, 3.1rem); line-height: 1.02; margin: 14px 0 0; color: var(--hero_text); }
.s-hero .s-serif em { color: #F6A9C0; }
.s-scale { display: flex; align-items: flex-end; gap: 18px; margin: auto 0 18px; }
.s-scale .px { display: flex; flex-direction: column; align-items: center; gap: 8px; font-size: .7rem; color: var(--hero_muted); }
.s-scale .p10 { width: 10px; height: 10px; background: #F6A9C0; border-radius: 2px; box-shadow: 0 0 0 6px rgba(246,169,192,.15); }
.s-scale .p32 { width: 32px; height: 32px; background: #B9B4F5; border-radius: 4px; }
.s-scale .note { font-size: .85rem; color: var(--hero_muted); line-height: 1.4; max-width: 190px; }
.s-bar { display: flex; height: 14px; border-radius: 99px; overflow: hidden; gap: 3px; }
.s-bar > div { height: 100%; border-radius: 99px; }
.s-legend { display: grid; grid-template-columns: repeat(3, auto); justify-content: space-between; margin-top: 12px; gap: 8px; }
.s-legend b { display: block; font-family: var(--serif) !important; font-weight: 400; font-size: 1.9rem; color: var(--hero_text); line-height: 1; }
.s-legend span { font-size: .7rem; color: var(--hero_muted); }
.s-flow { background: var(--card); border: 1px solid var(--card_border); box-shadow: 0 10px 34px var(--shadow); color: var(--text); }
.s-flow .s-serif { font-size: 2.05rem; margin: 6px 0 18px; line-height: 1.1; }
.s-steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; position: relative; }
.s-steps::before { content: ""; position: absolute; top: 17px; left: 18px; right: 18px; border-top: 1.5px dashed var(--line); filter: contrast(3); }
.s-step { position: relative; }
.s-node { width: 36px; height: 36px; border-radius: 12px; background: var(--ink); color: var(--ink_text); display: flex;
  align-items: center; justify-content: center; font-family: var(--mono) !important; font-size: .78rem; margin-bottom: 12px; position: relative; }
.s-step:nth-child(2) .s-node, .s-step:nth-child(4) .s-node { background: var(--pink); color: #1b1b1b; }
.s-step:nth-child(3) .s-node { background: var(--lav); color: #1b1b1b; }
.s-step h5 { font-size: .98rem; font-weight: 600; margin: 0 0 4px; color: var(--text); }
.s-step p { font-size: .84rem; color: var(--muted); margin: 0; line-height: 1.4; }
.s-changes { display: grid; grid-template-columns: 1.15fr 1fr 1fr; gap: 14px; min-height: 0; }
.s-change { display: flex; flex-direction: column; color: var(--chg); }
.s-change.c1 { background: var(--gpink); } .s-change.c2 { background: var(--glav); }
.s-change.c3 { background: var(--card); border: 1px solid var(--card_border); color: var(--text); box-shadow: 0 10px 34px var(--shadow); }
.s-change .s-serif { font-size: 1.8rem; line-height: 1.1; margin: 8px 0 8px; color: inherit; }
.s-change p { font-size: .86rem; line-height: 1.45; margin: 0; opacity: .78; color: inherit; }
.s-metric { margin-top: auto; padding-top: 12px; display: flex; align-items: baseline; gap: 8px; }
.s-metric b { font-family: var(--serif) !important; font-weight: 400; font-size: 2.3rem; line-height: 1; }
.s-metric span { font-family: var(--mono) !important; font-size: .7rem; opacity: .7; line-height: 1.3; }
.s-tag { position: absolute; top: 18px; right: 18px; font-family: var(--mono) !important; font-size: .66rem; text-transform: uppercase;
  letter-spacing: .1em; padding: 5px 9px; border-radius: 99px; background: rgba(255,255,255,.65); color: #1b1b1b; }
.s-change.c3 .s-tag { background: var(--chip); color: var(--text); border: 1px solid var(--line); }
@media (max-height: 820px) { .s-step p { display: none; } .s-scale { margin-bottom: 12px; } .story { height: calc(100vh - 205px); } }

/* phones: let it scroll normally */
@media (max-width: 760px) {
  html, body, .stApp, [data-testid="stAppViewContainer"], section[data-testid="stMain"] { height: auto; overflow: auto !important; }
  .block-container { height: auto; overflow: visible !important; }
  section[data-testid="stMain"] { overflow: auto !important; }
  .story { grid-template-columns: 1fr; height: auto; } .s-hero { grid-row: auto; } .s-steps, .s-changes { grid-template-columns: 1fr 1fr; }
  div[data-testid="stVerticalBlockBorderWrapper"]:has(> .st-key-mainwrap) { overflow: visible; }
}
"""
st.markdown("<style>" + FONTS + ":root{" + "".join(f"--{k}:{v};" for k, v in TH.items()) + "}" + CSS + "</style>",
            unsafe_allow_html=True)

# ============================================================================ intro animation
# Plays once per browser session (first run only), CSS only, then lifts away. Off switch in Settings;
# skipped for people whose system asks for reduced motion.
INTRO_CSS = """
.intro { position: fixed; inset: 0; z-index: 100000; display: grid; place-items: center; background: var(--bg);
  animation: intro-out .75s cubic-bezier(.7,0,.25,1) 3.1s forwards; pointer-events: none; }
.block-container:has(.intro) { backdrop-filter: none; -webkit-backdrop-filter: none;
  animation: intro-blur 0s 3.9s forwards; }
@keyframes intro-blur { to { backdrop-filter: blur(26px); -webkit-backdrop-filter: blur(26px); } }
@keyframes intro-out { to { opacity: 0; transform: translateY(-4%); visibility: hidden; } }
.intro .stage { display: flex; flex-direction: column; align-items: center; gap: 22px; }
.intro .radar { position: relative; width: 210px; height: 210px; border-radius: 50%;
  background: radial-gradient(circle, transparent 0 34%, var(--line) 34.5% 35%, transparent 35.5% 59%,
              var(--line) 59.5% 60%, transparent 60.5% 84%, var(--line) 84.5% 85%, transparent 85.5%);
  animation: intro-pop .7s cubic-bezier(.2,.9,.3,1.25) both; }
.intro .sweep { position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(from 0deg, transparent 0 290deg, color-mix(in srgb, var(--lav) 55%, transparent) 360deg);
  animation: intro-spin 1.4s linear .25s 2 both; }
.intro .core { position: absolute; left: 50%; top: 50%; width: 64px; height: 64px; margin: -32px 0 0 -32px;
  border-radius: 20px; background: #111; display: grid; place-items: center;
  box-shadow: 0 14px 34px var(--shadow); animation: intro-pop .6s cubic-bezier(.2,.9,.3,1.3) .15s both; }
.intro .core svg { width: 32px; height: 32px; }
.intro .ping { position: absolute; width: 10px; height: 10px; margin: -5px; border-radius: 50%; background: var(--pink);
  opacity: 0; animation: intro-ping 1.1s ease-out both; }
.intro .ping::after { content: ""; position: absolute; inset: -6px; border-radius: 50%; border: 2px solid var(--pink);
  animation: intro-ring 1.1s ease-out both; animation-delay: inherit; }
.intro .p1 { left: 74%; top: 26%; animation-delay: .65s; }
.intro .p2 { left: 22%; top: 68%; animation-delay: 1.05s; }
.intro .p3 { left: 80%; top: 70%; animation-delay: 1.45s; background: var(--lav); }
.intro .p3::after { border-color: var(--lav); }
.intro .p4 { left: 30%; top: 22%; animation-delay: 1.8s; }
.intro h1 { margin: 0; padding: 0; font-family: var(--serif) !important; font-weight: 400; font-size: 64px;
  letter-spacing: -.5px; color: var(--text); line-height: 1; }
.intro h1 span { display: inline-block; opacity: 0; animation: intro-up .7s cubic-bezier(.2,.8,.2,1) both; }
.intro h1 span:nth-child(1) { animation-delay: .55s; } .intro h1 span:nth-child(2) { animation-delay: .7s; }
.intro h1 span:nth-child(3) { animation-delay: .85s; } .intro h1 span:nth-child(4) { animation-delay: 1.0s; }
.stApp .intro h1, .stApp .intro h1 span, .stApp .intro h1 em { font-family: var(--serif) !important; }
.intro h1 em { font-style: italic; }
.intro .sub { font-family: var(--mono) !important; font-size: 12px; letter-spacing: 2.5px; text-transform: uppercase;
  color: var(--muted); opacity: 0; animation: intro-up .7s ease 1.25s both; }
.intro .bar { width: 180px; height: 3px; border-radius: 3px; background: var(--line); overflow: hidden;
  opacity: 0; animation: intro-up .4s ease 1.3s both; }
.intro .bar i { display: block; height: 100%; width: 100%; border-radius: 3px; transform-origin: left;
  background: linear-gradient(90deg, var(--pink), var(--lav)); animation: intro-load 1.6s cubic-bezier(.6,0,.2,1) 1.35s both; }
.intro .status { font-family: var(--mono) !important; font-size: 11px; color: var(--muted); height: 14px; position: relative; width: 260px; text-align: center; }
.intro .status b { position: absolute; inset: 0; font-weight: 400; opacity: 0; }
.intro .status b:nth-child(1) { animation: intro-flash .8s ease 1.4s both; }
.intro .status b:nth-child(2) { animation: intro-flash .8s ease 2.1s both; }
.intro .status b:nth-child(3) { animation: intro-stay .5s ease 2.8s both; color: var(--text); }
@keyframes intro-pop { from { transform: scale(.55); opacity: 0; } to { transform: scale(1); opacity: 1; } }
@keyframes intro-spin { from { transform: rotate(0); } to { transform: rotate(360deg); } }
@keyframes intro-ping { 0% { opacity: 0; transform: scale(.2); } 30% { opacity: 1; transform: scale(1.25); } 100% { opacity: 1; transform: scale(1); } }
@keyframes intro-ring { from { transform: scale(.4); opacity: .9; } to { transform: scale(2.4); opacity: 0; } }
@keyframes intro-up { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes intro-load { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes intro-flash { 0% { opacity: 0; } 25%, 75% { opacity: 1; } 100% { opacity: 0; } }
@keyframes intro-stay { from { opacity: 0; } to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .intro { display: none; } .block-container:has(.intro) { animation: none; backdrop-filter: blur(26px); } }
"""

if CFG.get("intro", True) and not ss.get("_intro_played"):
    ss["_intro_played"] = True
    st.markdown(
        "<style>" + INTRO_CSS + "</style>"
        '<div class="intro"><div class="stage">'
        '<div class="radar"><div class="sweep"></div>'
        '<i class="ping p1"></i><i class="ping p2"></i><i class="ping p3"></i><i class="ping p4"></i>'
        '<div class="core"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8">'
        '<circle cx="5" cy="5" r="3"/><circle cx="19" cy="5" r="3"/><circle cx="5" cy="19" r="3"/>'
        '<circle cx="19" cy="19" r="3"/><path d="M7.5 7.5 10 10M16.5 7.5 14 10M7.5 16.5 10 14M16.5 16.5 14 14"/>'
        '<rect x="9.5" y="9.5" width="5" height="5" rx="1.5"/></svg></div></div>'
        '<h1><span>Eyes</span> <span>in</span> <span>the</span> <span><em>Sky</em></span></h1>'
        '<div class="sub">UAV survivor detection · ground station</div>'
        '<div class="bar"><i></i></div>'
        '<div class="status"><b>loading detector…</b><b>scanning for drone link…</b><b>ready</b></div>'
        "</div></div>",
        unsafe_allow_html=True)

# ============================================================================ helpers
SVG = {
    "search": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    "bell": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>',
    "logo": '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8"><circle cx="5" cy="5" r="3"/><circle cx="19" cy="5" r="3"/><circle cx="5" cy="19" r="3"/><circle cx="19" cy="19" r="3"/><path d="M7.5 7.5 10 10M16.5 7.5 14 10M7.5 16.5 10 14M16.5 16.5 14 14"/><rect x="9.5" y="9.5" width="5" height="5" rx="1.5"/></svg>',
}


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


def predict(model, frame, conf, imgsz, coco=False, augment=False):
    kwargs = dict(conf=conf, imgsz=imgsz, verbose=False, augment=augment)
    if coco:
        kwargs["classes"] = [0]
    return model.predict(frame, **kwargs)[0]


def draw(frame, r, color=(171, 116, 236)):
    out = frame.copy()
    for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{c * 100:.0f}%", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out, len(r.boxes)


def detect(model, frame, conf, imgsz, coco=False, augment=False, color=(171, 116, 236)):
    return draw(frame, predict(model, frame, conf, imgsz, coco=coco, augment=augment), color)


PINK_BGR, LAV_BGR = (171, 116, 236), (230, 140, 150)  # detections (pink) / ground truth (lavender-blue)


def draw_ground_truth(img, label_file):
    h, w = img.shape[:2]
    n = 0
    if label_file.exists():
        for line in label_file.read_text().split("\n"):
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                          (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), LAV_BGR, 2)
            n += 1
    return img, n


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def title(t, sub=""):
    html(f'<p class="ctitle">{t}</p>' + (f'<p class="csub">{sub}</p>' if sub else ""))


def hcard(ico, t, d, pink=False, compact=False):
    cls = "hcard" + (" pink" if pink else "") + (" compact" if compact else "")
    if compact:
        return f'<div class="{cls}"><div class="head"><div class="ico">{ico}</div><h4>{t}</h4></div><p>{d}</p></div>'
    return f'<div class="{cls}"><div class="ico">{ico}</div><h4>{t}</h4><p>{d}</p></div>'


def pbar(label, frac, soft, solid):
    frac = max(0.0, min(1.0, frac))
    return (f'<div class="pbar-row"><span>{label}</span><span>{frac * 100:.0f}/100%</span></div>'
            f'<div class="pbar" style="background:{soft}"><div style="width:{frac * 100:.1f}%;'
            f'background:linear-gradient(90deg,{solid}AA,{solid})"></div></div>')


def score(ico, bg, name, frac):
    return (f'<div class="score"><div class="av" style="background:{bg}">{ico}</div><div class="mid">'
            f'<div class="nm">{name}</div><div class="tr"><div style="width:{frac * 100:.0f}%"></div></div></div>'
            f'<div class="pc">{frac * 100:.0f}%</div></div>')


def waiting(e, t, d):
    html(f'<div class="waiting"><div class="e">{e}</div><h4>{t}</h4><p>{d}</p></div>')


def go(page, **state):
    """Button callback: set some state and jump to a page."""
    for k, v in state.items():
        ss[k] = v
    ss["nav"] = page


@st.cache_data
def hero_photo_b64(width=720):
    """A real aerial test photo (JPEG, base64) used as the Story page backdrop."""
    import base64
    imgs = sorted(TEST_IMAGES.glob("*.*"))
    if not imgs:
        return ""
    img = cv2.imread(str(imgs[min(37, len(imgs) - 1)]))
    if img is None:
        return ""
    img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def load_sessions():
    frames = []
    for f in sorted(LOGS.glob("drone_session_*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df):
            df["session"] = f.stem.replace("drone_session_", "")
            frames.append(df)
    return frames


def latest_sightings(n=4):
    dirs = sorted((d for d in LOGS.glob("sightings_*") if d.is_dir() and any(d.iterdir())), reverse=True)
    return (sorted(dirs[0].glob("*.jpg"))[-n:], dirs[0].name) if dirs else ([], "")


def notifications(limit=6):
    """Recent events from the logs: sighting photos and finished flights, newest first."""
    events = []
    for d in LOGS.glob("sightings_*"):
        for f in d.glob("*.jpg"):
            try:
                n = int(f.stem.split("_")[-1].replace("people", ""))
            except ValueError:
                n = 0
            events.append({"t": f.stat().st_mtime, "kind": "sighting", "path": f,
                           "text": f"{n} {'person' if n == 1 else 'people'} spotted"})
    for f in LOGS.glob("drone_session_*.csv"):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df):
            events.append({"t": f.stat().st_mtime, "kind": "flight", "path": f,
                           "text": f"Flight logged: {len(df)} frames, peak {int(df['people'].max())} people"})
    return sorted(events, key=lambda e: e["t"], reverse=True)[:limit]


def flight_label(stamp):
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%d %b, %H:%M")
    except ValueError:
        return stamp


@st.cache_data(ttl=8, show_spinner=False)
def esp32_online():
    from uav_survivor_detection.inference.drone_link import tcp_reachable
    return tcp_reachable("192.168.4.1", 80, 0.25) or tcp_reachable("192.168.4.1", 81, 0.25)


def ago(t):
    d = time.time() - t
    return "just now" if d < 60 else f"{d / 60:.0f} min ago" if d < 3600 else \
        f"{d / 3600:.0f} h ago" if d < 86400 else datetime.fromtimestamp(t).strftime("%d %b")


def axis(**kw):
    opts = dict(grid=False, domain=False, ticks=False, labelColor=TH["muted"], titleColor=TH["muted"],
                labelFont="Geist Mono", titleFont="Geist", labelFontSize=12)
    return alt.Axis(**{**opts, **kw})


# ============================================================================ header
h1, h2, h3 = st.columns([1, 5, 1.6], vertical_alignment="center")
with h1, st.container(key="logo"):
    html(f'<div class="logo">{SVG["logo"]}</div>')
    st.button("Home", key="logo_btn", help="Dashboard", on_click=go, args=("Dashboard",))
with h2:
    st.segmented_control("Navigation", PAGES, key="nav", label_visibility="collapsed")

EVENTS = notifications()
UNREAD = sum(e["t"] > CFG["notif_seen"] for e in EVENTS)
TEAM = CFG["team"].strip() or "Rescue Team"
INITIALS = "".join(w[0] for w in TEAM.split()[:2]).upper() or "RT"
with h3, st.container(key="hdr_row"):
    # ---- quick find
    with st.container(key="hdr_search"):
        with st.popover("", icon=":material/search:", help="Quick find"):
            q = st.text_input("Find a page, action or flight", key="qf", placeholder="e.g. webcam, results, flight")
            items = [(f"Go to {p_}", "page", p_) for p_ in PAGES] + [
                ("Connect the ESP32 drone", "act", dict(drone_mode="wifi", drone_url="http://192.168.4.1/stream")),
                ("Start the laptop webcam (backup)", "act", dict(drone_mode="webcam")),
                ("Replay the recorded flight", "act", dict(drone_mode="file")),
                ("Switch to dark mode" if ss.theme == "light" else "Switch to light mode", "theme", None)]
            for f in sorted(LOGS.glob("drone_session_*.csv"), reverse=True)[:12]:
                stamp = f.stem.replace("drone_session_", "")
                items.append((f"Flight {flight_label(stamp)}", "flight", stamp))
            words = q.lower().split()
            hits = [it for it in items if all(w in it[0].lower() for w in words)] if words else items[:7]
            for i, (label, kind, arg) in enumerate(hits[:9]):
                if kind == "page":
                    st.button(label, key=f"qf_{i}", on_click=go, args=(arg,), use_container_width=True)
                elif kind == "act":
                    st.button(label, key=f"qf_{i}", on_click=go, args=("Live Drone",), kwargs=arg,
                              use_container_width=True)
                elif kind == "flight":
                    st.button(label, key=f"qf_{i}", icon=":material/show_chart:", on_click=go,
                              args=("Dashboard",), kwargs={"flight_pick": arg}, use_container_width=True)
                else:
                    nxt = "dark" if ss.theme == "light" else "light"
                    st.button(label, key=f"qf_{i}", on_click=lambda t=nxt: (ss.update(theme=t), save_settings(theme=t)),
                              use_container_width=True)
            if words and not hits:
                st.caption("Nothing matches. Try: drone, webcam, results, flight.")
    # ---- notifications
    with st.container(key="hdr_bell_unread" if UNREAD else "hdr_bell"):
        with st.popover("", icon=":material/notifications:", help=f"{UNREAD} new" if UNREAD else "Notifications"):
            html(f'<b class="notif-list">Notifications</b>' + (f" · {UNREAD} new" if UNREAD else ""))
            if not EVENTS:
                st.caption("Nothing yet. Sightings and finished flights appear here.")
            for k, e in enumerate(EVENTS):
                a_, b_ = st.columns([1, 2.4], vertical_alignment="center")
                if e["kind"] == "sighting":
                    a_.image(str(e["path"]), use_container_width=True)
                else:
                    a_.markdown("📈")
                new = " · **new**" if e["t"] > CFG["notif_seen"] else ""
                b_.markdown(f"{e['text']}  \n<span class='csub'>{ago(e['t'])}{new}</span>", unsafe_allow_html=True)
            if EVENTS:
                st.button("Mark all as read", key="notif_read", use_container_width=True,
                          on_click=lambda: save_settings(notif_seen=time.time()))
    # ---- settings (avatar)
    with st.container(key="hdr_avatar"):
        with st.popover(INITIALS, help="Settings"):
            st.markdown("**Settings**")
            team = st.text_input("Team name (shown in the greeting)", value=TEAM, key="set_team")
            if team.strip() and team.strip() != TEAM:
                save_settings(team=team.strip())
                st.rerun()
            th = st.segmented_control("Theme", ["light", "dark"], default=ss.theme, key="set_theme",
                                      format_func=str.title)
            if th and th != ss.theme:
                ss.theme = th
                save_settings(theme=th)
                st.rerun()
            st.markdown("**Live AI defaults**")
            opts = [AUTO] + list(MODELS.keys())
            dm = st.selectbox("Model", opts, index=opts.index(CFG["live_model"]) if CFG["live_model"] in opts else 0,
                              key="set_model")
            dc = st.slider("How sure before marking?", 0.05, 0.9, float(CFG["live_conf"]), 0.05, key="set_conf")
            dsv = st.checkbox("Save sighting photos", value=bool(CFG["save_shots"]), key="set_save")
            intro_on = st.checkbox("Play intro animation on start", value=bool(CFG.get("intro", True)), key="set_intro")
            if intro_on != bool(CFG.get("intro", True)):
                save_settings(intro=intro_on)
            if (dm, dc, dsv) != (CFG["live_model"] or AUTO, CFG["live_conf"], CFG["save_shots"]):
                save_settings(live_model=dm, live_conf=dc, save_shots=dsv)
            if st.button("Open logs folder", icon=":material/folder_open:", use_container_width=True):
                try:
                    os.startfile(str(LOGS))  # Windows: opens Explorer on the logs folder
                except Exception as err:
                    st.caption(f"Logs are in {LOGS} ({err})")
            st.caption(f"Settings are saved in app/{SETTINGS_FILE.name}")
page = ss.get("nav") or ss.get("_last_page", "Dashboard")
ss["_last_page"] = page

# Left icon rail: pinned to the panel's left edge with CSS (see .st-key-rail_nav / .st-key-rail_theme).
with st.container(key="rail_nav"):
    for name, icon in RAIL:
        st.button("", icon=icon, key=f"railbtn_{name}", help=name, on_click=go, args=(name,),
                  type="primary" if page == name else "secondary")
with st.container(key="rail_theme"):
    st.button("", icon=":material/light_mode:", key="rail_light", help="Light", on_click=lambda: (ss.update(theme="light"), save_settings(theme="light")),
              type="primary" if ss.theme == "light" else "secondary")
    st.button("", icon=":material/dark_mode:", key="rail_dark", help="Dark", on_click=lambda: (ss.update(theme="dark"), save_settings(theme="dark")),
              type="primary" if ss.theme == "dark" else "secondary")
main = st.container(key="mainwrap")

hour = datetime.now().hour
GREETING = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"

# ============================================================================ pages
with main:
    # ------------------------------------------------------------------ Dashboard
    if page == "Dashboard":
        html(f'<div class="greet">{GREETING}, <em>{TEAM}</em></div>'
             '<p class="greet-sub">Eyes in the Sky · AI that finds people from a drone, on an ordinary laptop.</p>')
        left, right = st.columns([2.15, 1], gap="medium")
        with left:
            with st.container(key="card_activity"):
                c1, c2, c3 = st.columns([2.1, 1.5, 1.6], vertical_alignment="top")
                with c1:
                    title("Mission Activity", "People spotted per frame during flights")
                    html(f'<div class="legend"><span><i style="background:{TH["lav"]}"></i>People per frame</span>'
                         f'<span><i style="background:{TH["pink"]}"></i>Trend</span></div>')
                if HAVE_RESULTS:
                    b, e = RES["baseline:full@480"], RES[SEL["proposed"]]
                    gain = (e[T]["recall50_conf0.25"] - b[T]["recall50_conf0.25"]) * 100
                    c2.markdown(f'<div class="pillstat"><div class="n">{gain:+.0f} pts</div>'
                                f'<div class="l">tiny-survivor recall vs standard</div></div>', unsafe_allow_html=True)
                with c3, st.container(key="seg_range"):
                    rng = st.segmented_control("Range", ["Last flight", "All flights"], default="Last flight",
                                               key="act_range", label_visibility="collapsed") or "Last flight"
                sessions = load_sessions()
                if not sessions:
                    waiting("🛸", "No flights yet", "Start watching on the Live Drone page with the AI on. "
                            "Every frame's sightings will be charted here.")
                else:
                    pick = ss.get("flight_pick")
                    picked = [d for d in sessions if d["session"].iloc[0] == pick] if pick else []
                    if picked:
                        df = picked[0]
                        cx1, cx2 = st.columns([4, 1], vertical_alignment="center")
                        cx1.caption(f"Showing the flight from {flight_label(pick)} (picked in Quick find)")
                        cx2.button("Clear", key="clear_pick", on_click=lambda: ss.pop("flight_pick", None))
                    else:
                        df = sessions[-1] if rng == "Last flight" else pd.concat(sessions, ignore_index=True)
                    df = df.reset_index(drop=True)
                    df["step"] = np.arange(1, len(df) + 1)
                    peak = df["people"].max()
                    df["hot"] = df["people"] >= max(1, 0.8 * peak)
                    base = alt.Chart(df)
                    bars = base.mark_rule(strokeWidth=3, opacity=.9, strokeCap="round").encode(
                        x=alt.X("step:Q", title=None, axis=axis(tickCount=8)),
                        y=alt.Y("people:Q", title=None, axis=axis(tickCount=4)),
                        y2=alt.datum(0),
                        color=alt.condition("datum.hot", alt.value(TH["pink"]), alt.value(TH["lav"])),
                        tooltip=[alt.Tooltip("step:Q", title="Frame"), alt.Tooltip("people:Q", title="People"),
                                 alt.Tooltip("utc_time:N", title="Time (UTC)")])
                    trend = base.transform_window(avg="mean(people)", frame=[-6, 6]).mark_line(
                        interpolate="monotone", color=TH["pink"], strokeWidth=2.4).encode(x="step:Q", y="avg:Q")
                    st.altair_chart((bars + trend).properties(height=175).configure_view(stroke=None)
                                    .configure(background="transparent", font="Geist"), use_container_width=True)
                    st.caption(f"{len(df)} frames · most people in one frame: {peak} · "
                               f"{len(sessions)} flight{'s' if len(sessions) != 1 else ''} logged")

            l2, r2 = st.columns(2, gap="medium")
            with l2, st.container(key="card_progress"):
                title("Detection Progress", "300 unseen test photos")
                with st.container(key="seg_prog"):
                    which = st.segmented_control("Model", ["Our model", "Standard"], default="Our model",
                                                 key="prog_model", label_visibility="collapsed") or "Our model"
                if HAVE_RESULTS:
                    r = RES[SEL["proposed"]] if which == "Our model" else RES["baseline:full@480"]
                    html(pbar("People found", r[A]["recall50_conf0.25"], TH["lav_soft"], TH["lav"])
                         + pbar("Tiny people found (&lt;10 px)", r[T]["recall50_conf0.25"], TH["pink_soft"], TH["pink"]))
                else:
                    waiting("⏳", "Evaluation running", "Results appear here automatically.")
            with r2, st.container(key="card_scores"):
                title("Model Scoreboard", "Overall accuracy (mAP@50) on the test set")
                if HAVE_RESULTS:
                    both = SEL.get("both")
                    rows = [("🛰️", TH["pink_soft"], f"Ours · {SEL['proposed'].split('@')[1]} px", RES[SEL["proposed"]])]
                    if both in RES:
                        rows.append(("🧠", TH["lav_soft"], f"P2 + CBAM · {both.split('@')[1]} px", RES[both]))
                    rows.append(("📦", TH["chip"], "Standard YOLO11n · 480 px", RES["baseline:full@480"]))
                    html("".join(score(i, bg, n, r[A]["AP50"]) for i, bg, n, r in rows))

        with right, st.container(key="card_connect"):
            title("Connect a drone", "Pick how its video reaches this laptop.")
            CONNECT = [
                ("gpink1", "ESP32-CAM Drone", "Live video over its WiFi hotspot", "📶 192.168.4.1",
                 dict(drone_mode="wifi", drone_url="http://192.168.4.1/stream")),
                ("glav1", "Laptop Webcam", "Emergency backup, live AI on this laptop", "💻 Camera 0",
                 dict(drone_mode="webcam")),
                ("gpink2", "Recorded Flight", "Replay a flight through the same AI", "🎬 Backup clip",
                 dict(drone_mode="file")),
            ]
            clip_ok = any((ROOT / "demo_output").glob(e_) for e_ in ("*.avi", "*.mp4", "*.mov", "*.mkv"))
            STATUS = {"gpink1": ("on", "Online") if esp32_online() else ("off", "Not in range"),
                      "glav1": ("on", "Ready"),
                      "gpink2": ("on", "Ready") if clip_ok else ("off", "No clip")}
            for key, t, sub, chip, state in CONNECT:
                with st.container(key=key):
                    cls, txt = STATUS[key]
                    tip = ("Checks every few seconds whether the drone answers at 192.168.4.1" if key == "gpink1"
                           else "Uses this laptop's camera" if key == "glav1" else "Backup clip in demo_output/")
                    html(f'<span class="cstat {cls}" title="{tip}"><i></i>{txt}</span>'
                         f'<p class="gtitle">{t}</p><p class="gsub">{sub}</p>')
                    a, b = st.columns([1.15, 1], vertical_alignment="center")
                    a.markdown(f'<span class="chip">{chip}</span>', unsafe_allow_html=True)
                    b.button("Connect", icon=":material/arrow_forward:", key=f"btn_{key}", type="primary",
                             use_container_width=True, on_click=go, args=("Live Drone",), kwargs=state)

    # ------------------------------------------------------------------ Live Drone
    elif page == "Live Drone":
        html('<div class="greet">Fly the <em>drone</em></div><p class="greet-sub">Connect the drone the way its own software '
             'does. Its live video appears here, with or without the AI marking people.</p>')
        if ON_CLOUD:
            st.info("This is the online demo. Live drone video and the webcam only work in the app on the "
                    "ground-station laptop, which sits on the drone's WiFi. Here you can replay the recorded flight.",
                    icon=":material/cloud:")
        MODES = {
            "wifi": "📶  Drone WiFi stream (ESP32-CAM / FPV)",
            "webcam": "💻  Laptop webcam (emergency backup)",
            "window": "🪟  Drone's own app on this laptop",
            "device": "🔌  HDMI capture card / webcam",
            "url": "🌐  Stream link (RTSP / RTMP / UDP)",
            "tello": "✈️  DJI / Ryze Tello",
            "file": "🎬  Recorded flight (backup)",
        }
        HOW = {
            "wifi": "Join the drone's own WiFi (no internet needed), then press <b>Find my drone</b>. Close the drone's "
                    "Python app first: the ESP32 streams to one viewer at a time.",
            "webcam": "Emergency backup: uses this laptop's own camera, so the panel still sees live AI detection "
                      "if the drone fails. Stand back 2 to 4 m so your whole body is in view.",
            "window": "Open the drone's PC software on its camera view and keep it <b>visible, not minimised</b>.",
            "device": "Controller HDMI → USB capture card. It appears as camera 0, 1, 2…",
            "url": "Point the drone app's <b>Live stream / RTMP</b> at this laptop, or paste its RTSP link.",
            "tello": "Join the <b>TELLO-xxxx</b> WiFi. Its camera is switched on automatically.",
            "file": "No drone? Replay a recorded flight through exactly the same software.",
        }
        left, right = st.columns([2.15, 1], gap="medium")

        # Left: placeholders, filled by the live loop below.
        with left:
            with st.container(key="card_live"):
                lt, ls = st.columns([1, 2.2], vertical_alignment="center")
                with lt:
                    title("Live feed")
                status_box = ls.empty()
                frame_box = st.empty()
                info_box = st.empty()
            with st.container(key="card_sightings"):
                shots, shot_dir_name = latest_sightings()
                title("Latest sightings", f"Saved automatically · logs/{shot_dir_name}/" if shots
                      else "Photos are saved whenever the AI spots people.")
                if shots:
                    cols = st.columns(4)
                    for col, p in zip(cols, shots):
                        col.image(str(p), use_container_width=True,
                                  caption=p.stem.split("_")[-1].replace("people", " people"))

        # Right: connection controls.
        with right, st.container(key="card_conn"):
            title("Connect the drone", "Step 1 · how is it connected?")
            if st.button("Find my drone", icon=":material/radar:", use_container_width=True, type="secondary"):
                with st.spinner("Looking for drones on this network…"):
                    ss["drone_scan"] = discover()
                hit = next((r for r in ss["drone_scan"] if r["found"]), None)
                if hit:
                    ss["drone_mode"] = "tello" if hit["source"] == "tello" else (
                        "wifi" if hit["source"].startswith("http") else "url")
                    ss["drone_url"] = ss["drone_stream_url"] = hit["source"]
            if "drone_scan" in ss:
                found = [r for r in ss["drone_scan"] if r["found"]]
                if found:
                    st.success("Found: " + ", ".join(f"**{r['name']}**" for r in found))
                elif on_drone_wifi():
                    st.warning(f"You are on the drone's WiFi ({on_drone_wifi()}), but no video answered. Close the "
                               "drone's own app (the ESP32 serves one viewer at a time) and power-cycle the drone.")
                else:
                    st.info("This laptop is not on the drone's WiFi. Open the WiFi list, join the drone's network "
                            "(it will say *No internet*, that's fine), then try again.")
            mode = st.selectbox("Connection", list(MODES), format_func=MODES.get,
                                index=list(MODES).index(ss["drone_mode"]), label_visibility="collapsed")
            ss["drone_mode"] = mode
            st.caption(HOW[mode].replace("<b>", "**").replace("</b>", "**"))

            source = None
            if mode == "wifi":
                source = st.text_input("Drone stream address", value=ss["drone_url"])
                ss["drone_url"] = source
            elif mode == "window":
                wins = [w for w in list_windows() if "Eyes in the Sky" not in w]
                if wins:
                    pick = st.selectbox("Window showing the drone camera", wins)
                    source = f"window:{pick}"
                else:
                    st.warning("No app windows found. Open the drone's software (not minimised).")
                st.button("Refresh windows", icon=":material/refresh:", use_container_width=True)
            elif mode == "webcam":
                cam = st.number_input("Camera number (0 = built-in, 1 = USB webcam)", 0, 9, 0, key="webcam_idx")
                st.checkbox("Mirror the picture (like a selfie camera)", value=True, key="webcam_mirror")
                source = str(cam)
            elif mode == "device":
                source = str(st.number_input("Camera / capture card number", 0, 9, 0))
            elif mode == "url":
                source = st.text_input("Stream link", value=ss["drone_stream_url"],
                                       placeholder="rtsp://192.168.1.1:7070/webcam")
                ss["drone_stream_url"] = source
            elif mode == "tello":
                source = "tello"
            else:
                clips = sorted(str(p.relative_to(ROOT)) for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv")
                               for p in (ROOT / "demo_output").glob(ext))
                source = st.selectbox("Recorded flight", clips) if clips else st.text_input("Path to a video file")

            html('<p class="csub" style="margin-top:6px">Step 2 · AI</p>')
            ai_on = st.toggle("AI survivor spotting", value=True,
                              help="Off = show only the drone's own camera picture, untouched.")
            with st.expander("AI settings", icon=":material/tune:"):
                _opts = [AUTO] + list(MODELS.keys())
                model_name = st.selectbox("Which eye to use?", _opts,
                                          index=_opts.index(CFG["live_model"]) if CFG["live_model"] in _opts else 0,
                                          help="Auto: if someone is close to the camera (green boxes) it uses the general person "
                                               "model; otherwise it scans for tiny people far below with our aerial "
                                               "model (pink boxes).")
                live_conf = st.slider("How sure before marking?", 0.05, 0.9, float(CFG["live_conf"]), 0.05, key="live_conf",
                                      help="Higher = fewer false boxes. 0.5 suits a live camera; lower it (0.3-0.4) when flying high "
                                           "to catch more tiny people.")
                save_shots = st.checkbox("Save a photo whenever people are spotted", value=bool(CFG["save_shots"]))
            html('<p class="csub" style="margin-top:6px">Step 3 · go</p>')
            run = st.toggle("Start watching", value=False, disabled=not source, key="run_live")

        PILL = {"live": ("● LIVE", "#E5476B", True), "connecting": ("◌ CONNECTING", "#8D8896", True),
                "reconnecting": ("◌ SIGNAL LOST · RECONNECTING", "#E58A3C", True),
                "ended": ("■ FINISHED", "#7A74E0", False), "failed": ("✕ NOT CONNECTED", "#8D8896", False),
                "stopped": ("■ STOPPED", "#8D8896", False)}

        if not run:
            status_box.markdown(f'<div style="text-align:right"><span class="live-pill" style="background:#8D8896">'
                                f'○ OFFLINE</span></div>', unsafe_allow_html=True)
            with frame_box.container():
                waiting("🛸", "Waiting for take-off", "Connect the drone on the right, then switch on <b>Start watching</b>.")
        else:
            infer = None
            if ai_on:
                names = [EVERYDAY, ENHANCED] if model_name == AUTO else [model_name]
                if any(not MODELS[nm][0].exists() for nm in names):
                    st.error("That model's weights are missing. Pick the everyday person-spotter in AI settings.")
                    st.stop()

                def run_one(nm, img, conf):
                    path, isz, aug = MODELS[nm]
                    return detect(load_model(str(path)), img, conf, isz, coco=path.name == "yolo11n.pt", augment=aug)

                if model_name == AUTO:
                    def infer(img):
                        # Measured on our own live ESP32 frames vs aerial test photos: when the drone is close,
                        # the general model's person box covers >90% of the frame; from the air its boxes are
                        # <1%. So a person box >= 5% of the frame means "close range": trust the general model
                        # and skip the aerial model, whose tiny-person guesses are false alarms up close.
                        p, isz, aug = MODELS[EVERYDAY]
                        r = predict(load_model(str(p)), img, live_conf, isz, coco=True)
                        h, w = img.shape[:2]
                        areas = ((r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])) / (h * w)
                        if len(r.boxes) and float(areas.max()) >= 0.05:
                            return draw(img, r, color=(120, 190, 90))  # close-range person: green
                        return run_one(ENHANCED, img, live_conf)  # from the air: tiny people, pink
                else:
                    def infer(img):
                        return run_one(model_name, img, live_conf)
            try:
                link = DroneLink(source if mode != "file" else str(ROOT / source)).start()
            except ValueError as err:
                st.error(f"📡 {err}")
                st.stop()
            ss["_drone_link"] = link

            stamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = LOGS / f"drone_session_{stamp}.csv"
            shot_dir = LOGS / f"sightings_{stamp}"

            panel = {"key": None}  # what the video area currently shows instead of video (avoid redraw flicker)

            def show_status(s, n=None, ai_fps=0.0):
                label, color, blink = PILL.get(s.status, (s.status.upper(), "#8D8896", False))
                busy = "black frames" in (s.last_error or "")
                frozen = bool(s.last_error) and (s.age_ms is None or s.age_ms > 1500)  # no fresh picture, known why
                if frozen and s.status != "ended":
                    label, color, blink = (("◌ CAMERA BUSY", "#E58A3C", True) if busy
                                           else ("◌ SIGNAL LOST · RECONNECTING", "#E58A3C", True))
                if s.status == "live" and not frozen:
                    alert = ("Scanning…" if not n else f"<b>{n} {'person' if n == 1 else 'people'} spotted</b>")                         if ai_on else "Drone camera"
                    detail = (f"{s.resolution} · {'camera' if mode == 'webcam' else 'drone'} {s.fps_in:.0f} fps"
                              + (f" · AI {ai_fps:.1f} fps" if ai_on else "")
                              + (f" · delay {s.age_ms:.0f} ms" if s.age_ms is not None else "")
                              + (f" · reconnected {s.reconnects}×" if s.reconnects else ""))
                else:
                    alert, detail = link.spec.label, ("End of recording" if s.status == "ended" else "")
                    key = (s.status, busy, s.last_error)
                    if s.status != "ended" and panel["key"] != key:
                        panel["key"] = key
                        # explain in the video area itself, where everyone is looking
                        with frame_box.container():
                            if busy:
                                waiting("📷", "The camera is busy", s.last_error)
                            elif s.last_error:
                                waiting("📡", "Reconnecting…", s.last_error + " Trying again automatically.")
                            else:
                                waiting("🛰️", "Connecting…", f"Opening {link.spec.label}. The first picture "
                                        "can take a few seconds.")
                status_box.markdown(
                    f'<div style="text-align:right"><span class="csub">{detail}</span> &nbsp; {alert} &nbsp; '
                    f'<span class="live-pill{" blink" if blink else ""}" style="background:{color}">{label}</span></div>',
                    unsafe_allow_html=True)

            last_idx, shown, ai_fps, t_prev, peak, sightings, last_shot = 0, 0, 0.0, time.time(), 0, 0, 0.0
            try:
                with (open(log_file, "w") if ai_on else contextlib.nullcontext()) as fh:
                    if fh:
                        fh.write("utc_time,frame,people,source\n")
                    while True:
                        s = link.stats()
                        fr = link.latest()
                        if s.status in ("ended", "failed"):
                            show_status(s)
                            break
                        if fr is None or fr.index == last_idx:
                            if s.status != "live" or s.last_error:
                                show_status(s)
                            time.sleep(0.02)
                            continue
                        last_idx = fr.index
                        img = cv2.flip(fr.image, 1) if mode == "webcam" and ss.get("webcam_mirror", True) else fr.image
                        n, out = None, img
                        if ai_on:
                            out, n = infer(img)
                            peak = max(peak, n)
                            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(fr.timestamp))},"
                                     f"{fr.index},{n},{link.spec.kind}\n")
                            if n and save_shots and time.time() - last_shot > 2:
                                shot_dir.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(shot_dir / f"frame{fr.index:06d}_{n}people.jpg"), out)
                                last_shot, sightings = time.time(), sightings + 1
                        now = time.time()
                        ai_fps = 0.9 * ai_fps + 0.1 / max(now - t_prev, 1e-3) if shown else 0.0
                        t_prev, shown = now, shown + 1
                        show_status(link.stats(), n, ai_fps)
                        if out.shape[1] > 1280:  # keep the browser fast for big window captures
                            out = cv2.resize(out, (1280, int(out.shape[0] * 1280 / out.shape[1])),
                                             interpolation=cv2.INTER_AREA)
                        frame_box.image(rgb(out), channels="RGB", use_container_width=True)
                        panel["key"] = None
                        if ai_on and shown % 15 == 0:
                            fh.flush()
                            info_box.caption(f"Most people in one frame: {peak} · photos saved: {sightings}"
                                             f" · log: logs/{log_file.name}")
            finally:
                link.stop(join=0.5)
                ss.pop("_drone_link", None)

            s = link.stats()
            if s.status == "failed":
                st.error(f"📡 Couldn't connect to {link.spec.label}. {s.last_error}")
            elif s.status == "ended":
                st.success(f"🎬 Flight replay finished: {shown} frames {'checked' if ai_on else 'shown'}. "
                           + (f"Most people in one frame: {peak}. Sightings saved to logs/{log_file.name}"
                              + (f" and {sightings} photos to logs/{shot_dir.name}/" if sightings else "")
                              if ai_on else ""))

    # ------------------------------------------------------------------ Survivors
    elif page == "Survivors":
        html('<div class="greet">Spot the <em>survivors</em></div><p class="greet-sub">Can you find them? '
             'Our model can. Compare it with the standard detector on real test scenes.</p>')
        with st.container(key="card_spot_ctrl"):
            images = sorted(TEST_IMAGES.glob("*.*"))
            c1, c2, c3 = st.columns([2, 1.2, 1.6], vertical_alignment="bottom")
            idx = c1.slider("Pick a scene", 1, max(len(images), 1), 1) - 1 if images else 0
            conf = c2.slider("How sure before marking?", 0.05, 0.9, 0.25, 0.05, key="sbs_conf")
            upload = c3.file_uploader("Or drop in your own aerial photo", type=["jpg", "jpeg", "png"])
            html(f'<div class="legend"><span><i style="background:#9690E6"></i>Real people (answer key)</span>'
                 f'<span><i style="background:#EC74AB"></i>What the AI found</span></div>')
        img, label_file = None, Path("__none__")
        if upload is not None:
            img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
        elif images:
            img = cv2.imread(str(images[idx]))
            label_file = TEST_LABELS / f"{images[idx].stem}.txt"
        if img is not None:
            cols = st.columns(3, gap="medium")
            gt_img, n_gt = draw_ground_truth(img.copy(), label_file)
            with cols[0], st.container(key="card_gt"):
                title("Answer key", f"{n_gt} people" if n_gt else "Original photo")
                st.image(rgb(gt_img), use_container_width=True)
            for col, name, t, k in [(cols[1], BASELINE, "Standard model", "card_std"),
                                    (cols[2], ENHANCED, "Our model", "card_ours")]:
                with col, st.container(key=k):
                    path, msz, maug = MODELS[name]
                    if not path.exists():
                        title(t)
                        waiting("⏳", "Still training", "")
                        continue
                    out, n = detect(load_model(str(path)), img, conf, msz, augment=maug)
                    title(("🌟 " if name == ENHANCED else "") + t, f"found {n}")
                    st.image(rgb(out), use_container_width=True)

    # ------------------------------------------------------------------ Results
    elif page == "Results":
        html('<div class="greet">Proof it <em>works</em></div><p class="greet-sub">Both detectors trained the same way, '
             'then tested on 300 photos neither had ever seen.</p>')
        if not HAVE_RESULTS:
            waiting("⏳", "Almost there", "The final evaluation is still running. Results will appear here automatically.")
        else:
            b, e = RES["baseline:full@480"], RES[SEL["proposed"]]

            def stat(key, num, lbl, new, old):
                d = (new - old) * 100
                with st.container(key=key):
                    html(f'<div class="big">{num}</div><p class="csub" style="font-size:.95rem;margin-top:8px">{lbl}</p>'
                         f'<span class="delta{" neg" if d < 0 else ""}">{"▲" if d >= 0 else "▼"} {abs(d):.1f} pts '
                         f'vs standard</span>')

            c = st.columns(3, gap="medium")
            with c[0]:
                stat("card_s1", f"{e[A]['recall50_conf0.25'] * 10:.1f}/10", "people on the ground our detector finds",
                     e[A]["recall50_conf0.25"], b[A]["recall50_conf0.25"])
            with c[1]:
                stat("card_s2", f"{e[T]['recall50_conf0.25'] * 100:.0f}%", "of the tiniest people (under 10 px) found",
                     e[T]["recall50_conf0.25"], b[T]["recall50_conf0.25"])
            with c[2]:
                stat("card_s3", f"{e[A]['AP50'] * 100:.0f}", "overall accuracy (mAP@50, out of 100)",
                     e[A]["AP50"], b[A]["AP50"])

            left, right = st.columns([2.15, 1], gap="medium")
            with left, st.container(key="card_compare"):
                title("Side by side", "Scores out of 100")
                html(f'<div class="legend"><span><i style="background:{TH["lav"]}"></i>Standard</span>'
                     f'<span><i style="background:{TH["pink"]}"></i>Ours</span></div>')
                rows = []
                for (bucket, k), name in [((A, "recall50_conf0.25"), "People found"),
                                          ((T, "recall50_conf0.25"), "Tiny people found"),
                                          ((A, "AP50"), "Overall accuracy"), ((A, "AP50_95"), "Box precision")]:
                    rows += [{"Measure": name, "Model": "Standard", "Score": b[bucket][k] * 100},
                             {"Measure": name, "Model": "Ours", "Score": e[bucket][k] * 100}]
                chart = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadius=6, size=16).encode(
                    x=alt.X("Measure:N", title=None, sort=None, axis=axis(labelAngle=0, labelFontSize=13)),
                    xOffset=alt.XOffset("Model:N", sort=["Standard", "Ours"]),
                    y=alt.Y("Score:Q", title=None, scale=alt.Scale(domain=[0, 100]), axis=axis(tickCount=5)),
                    color=alt.Color("Model:N", scale=alt.Scale(domain=["Standard", "Ours"],
                                                               range=[TH["lav"], TH["pink"]]), legend=None),
                    tooltip=["Measure", "Model", alt.Tooltip("Score:Q", format=".1f")],
                ).properties(height=210).configure_view(stroke=None).configure(background="transparent", font="Geist")
                st.altair_chart(chart, use_container_width=True)
            with right:
                with st.container(key="gpink_cost"):
                    html(f'<p class="gsub">Speed on a laptop CPU</p><div class="big">{1000 / e["latency_ms_640x480"]:.0f}'
                         f'<span style="font-size:1.1rem"> fps</span></div>')
                with st.container(key="glav_cost"):
                    html('<p class="gsub">Model size</p><div class="big">2.6 M<span style="font-size:1.1rem"> params</span></div>')
                with st.container(key="gpink_cost2"):
                    html('<p class="gsub">Graphics cards needed</p><div class="big">0</div>')

            with st.expander("Full details for the technical reviewers", icon=":material/table_chart:"):
                labels = {"baseline:full@480": "YOLO11n baseline (standard)", "enhanced:full@480": "+ P2 + CBAM",
                          SEL["proposed"]: "Proposed (selected on validation)", SEL.get("both"): "+ both"}
                table = [{"Variant": lab, "mAP@50": RES[k][A]["AP50"], "mAP@50:95": RES[k][A]["AP50_95"],
                          "Recall": RES[k][A]["recall50_conf0.25"], "Recall <10px": RES[k][T]["recall50_conf0.25"],
                          "Latency ms (640x480)": RES[k]["latency_ms_640x480"]} for k, lab in labels.items() if k in RES]
                st.dataframe(pd.DataFrame(table).set_index("Variant"), use_container_width=True)
                st.caption(f"Selection rule: {SEL.get('rule', '')}. Scored once on the held-out test split "
                           "(COCO-style, IoU 0.5; recall at confidence ≥ 0.25).")

    # ------------------------------------------------------------------ Story
    else:
        html('<div class="greet">Why this <em>matters</em></div>'
             '<p class="greet-sub">After a disaster, every minute matters. Here is the problem, and how we solved it.</p>')

        if HAVE_RESULTS:
            _b, _e = RES["baseline:full@480"], RES[SEL["proposed"]]
            _both = RES.get(SEL.get("both"), _e)
            tiny_gain = f'{(_e[T]["recall50_conf0.25"] - _b[T]["recall50_conf0.25"]) * 100:+.0f}'
            ap_delta = f'{(_both[A]["AP50"] - _e[A]["AP50"]) * 100:+.0f}'
            slow = f'{_both["latency_ms_640x480"] / _e["latency_ms_640x480"]:.1f}×'
            ms = f'{_e["latency_ms_640x480"]:.0f}'
        else:
            tiny_gain, ap_delta, slow, ms = "+", "≈0", "2×", "<100"
        try:
            _n = json.loads((RES_DIR / "size_stratified.json").read_text())["instances"]
            n_all, n_tiny, n_small, n_big = (_n["all"], _n["tiny (<10px)"], _n["small (10-32px)"], _n["medium+ (>=32px)"])
        except Exception:
            n_all, n_tiny, n_small, n_big = 10605, 4630, 5885, 90
        pt, ps, pb = (100 * n_tiny / n_all, 100 * n_small / n_all, 100 * n_big / n_all)

        _ph = hero_photo_b64()
        hero_bg = (f"background: linear-gradient(180deg, rgba(22,20,24,.94) 0%, rgba(22,20,24,.45) 38%, "
                   f"rgba(22,20,24,.55) 58%, rgba(22,20,24,.97) 80%), url(data:image/jpeg;base64,{_ph}) center/cover;"
                   if _ph else "")
        steps = [("01", "The drone looks", "Its camera streams live video to the laptop as it flies."),
                 ("02", "The laptop listens", "Every frame is caught; a dropped signal reconnects by itself."),
                 ("03", "The AI spots people", "Each frame is scanned and everyone found is boxed."),
                 ("04", "Rescuers act", "Sightings are saved with a time and a photo for the team.")]
        html(f'''<div class="story">
  <div class="s-card s-hero" style="{hero_bg}">
    <span class="s-kicker">01 / The problem</span>
    <div class="s-serif">From 50 metres up, a person is smaller than your <em>fingernail</em> on the screen.</div>
    <div class="s-scale">
      <div class="px"><div class="p10"></div>10 px</div>
      <div class="px"><div class="p32"></div>32 px</div>
      <div class="note">Shown at actual size. This is how big the survivors in our test photos are.</div>
    </div>
    <div class="s-bar"><div style="width:{pt:.1f}%;background:#F6A9C0"></div><div style="width:{ps:.1f}%;background:#B9B4F5"></div><div style="width:{max(pb, 1.2):.1f}%;background:#F7F2F4"></div></div>
    <div class="s-legend">
      <div><b>{pt:.0f}%</b><span class="s-mono">under 10 px</span></div>
      <div><b>{ps:.0f}%</b><span class="s-mono">10 to 32 px</span></div>
      <div><b>{pb:.0f}%</b><span class="s-mono">larger</span></div>
    </div>
    <span class="s-mono" style="font-size:.7rem;color:var(--hero_muted);margin-top:14px">{n_all:,} people labelled across 300 unseen test photos</span>
  </div>
  <div class="s-card s-flow">
    <span class="s-kicker">02 / How it works</span>
    <div class="s-serif">From a drone frame to a rescue, in about <em>{ms} milliseconds</em>.</div>
    <div class="s-steps">{"".join(f'<div class="s-step"><div class="s-node">{n}</div><h5>{t}</h5><p>{d}</p></div>' for n, t, d in steps)}</div>
  </div>
  <div class="s-changes">
    <div class="s-card s-change c1"><span class="s-tag">Kept</span><span class="s-kicker">03 / The fix</span>
      <div class="s-serif">More pixels per person</div>
      <p>The detector reads each frame in finer detail, so a 10-pixel survivor still carries enough detail to recognise.</p>
      <div class="s-metric"><b>{tiny_gain}</b><span>pts recall on<br>the tiniest people</span></div></div>
    <div class="s-card s-change c2"><span class="s-tag">Tested</span><span class="s-kicker">04 / Trade-off</span>
      <div class="s-serif">A heavier brain</div>
      <p>An extra detection layer plus attention (P2 + CBAM), taken from published papers.</p>
      <div class="s-metric"><b>{slow}</b><span>slower, {ap_delta} pts<br>accuracy. Dropped.</span></div></div>
    <div class="s-card s-change c3"><span class="s-tag">Fair</span><span class="s-kicker">05 / The test</span>
      <div class="s-serif">Same exam</div>
      <p>Both models learned from the same photos for the same time, then faced photos neither had seen.</p>
      <div class="s-metric"><b>300</b><span>unseen<br>test photos</span></div></div>
  </div>
</div>''')
