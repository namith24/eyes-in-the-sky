"""Panel demo app for the UAV Survivor Detection capstone.

Run from the project root:
    streamlit run app/demo_app.py
"""

import contextlib
import json
import sys
import time
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from uav_survivor_detection.models import custom_yolo  # noqa: E402,F401  (registers CBAM)
from uav_survivor_detection.inference.drone_link import DroneLink, discover, list_windows  # noqa: E402
from ultralytics import YOLO  # noqa: E402

TAG = "final_comparison"
RES_DIR = ROOT / "results" / TAG
WEIGHTS = {"baseline": ROOT / "runs" / TAG / "baseline" / "weights" / "best.pt",
           "enhanced": ROOT / "runs" / TAG / "enhanced" / "weights" / "best.pt"}


def _proposed():
    """Proposed configuration chosen on the validation set (results/final_comparison/selection.json)."""
    try:
        m, v = json.loads((RES_DIR / "selection.json").read_text())["proposed"].split(":")
        kind, size = v.split("@")
        return WEIGHTS[m], int(size), kind == "tta"
    except Exception:
        return WEIGHTS["baseline"], 800, False


_pp, _psz, _ptta = _proposed()
# name -> (weights, inference image size, test-time augmentation)
MODELS = {
    "Our detector: small-object aware (proposed)": (_pp, _psz, _ptta),
    "Standard YOLO11n (baseline)": (WEIGHTS["baseline"], 480, False),
    "Everyday person-spotter (close range)": (ROOT / "yolo11n.pt", 640, False),
}
ENHANCED, BASELINE, EVERYDAY = list(MODELS)
TEST_IMAGES = ROOT / "data" / "datasets" / "new_dataset3_subset" / "test" / "images"
TEST_LABELS = TEST_IMAGES.parent / "labels"
RESULTS = RES_DIR / "inference_variants_test.json"
SELECTION = RES_DIR / "selection.json"

ORANGE_BGR = (12, 89, 232)   # #E8590C detections
TEAL_BGR = (128, 150, 15)    # #0F9680 real people

st.set_page_config(page_title="Eyes in the Sky", page_icon="🧡", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=Nunito:wght@400;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'Nunito', sans-serif; }
h1, h2, h3, .serif { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

.hero {
  background: radial-gradient(circle at 85% 20%, #FFB46B 0%, transparent 45%),
              linear-gradient(135deg, #2B2118 0%, #5A2E14 55%, #E8590C 100%);
  border-radius: 28px; padding: 48px 44px; color: #FFF7EC; position: relative; overflow: hidden;
  margin-bottom: 18px;
}
.hero h1 { color: #FFF7EC; font-size: 3.1rem; line-height: 1.05; margin: 0 0 12px 0; }
.hero p { font-size: 1.2rem; max-width: 640px; opacity: .92; margin: 0; }
.hero .badge { display:inline-block; background: rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
  padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: .85rem; margin-bottom: 18px; }
.hero .drone { position:absolute; right: 40px; top: 30px; font-size: 6.5rem; opacity:.9;
  animation: hover 4s ease-in-out infinite; }
@keyframes hover { 0%,100% { transform: translateY(0) rotate(-3deg);} 50% { transform: translateY(-14px) rotate(3deg);} }

.stTabs [data-baseweb="tab-list"] { gap: 10px; background: transparent; }
.stTabs [data-baseweb="tab"] { background: #F3E9DA; border-radius: 999px; padding: 10px 22px;
  font-weight: 700; color: #5A2E14; border: none; }
.stTabs [aria-selected="true"] { background: #E8590C !important; color: #FFF7EC !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.card { background: #FFFFFF; border-radius: 22px; padding: 26px 24px; height: 100%;
  box-shadow: 0 6px 24px rgba(90,46,20,.08); border: 1px solid #F0E3D0; }
.card .emoji { font-size: 2.4rem; }
.card h3 { margin: 10px 0 6px 0; font-size: 1.3rem; color: #2B2118; }
.card p { color: #6B5A4B; margin: 0; font-size: 1rem; }
.step-num { display:inline-block; background:#E8590C; color:#fff; border-radius:999px; width:28px; height:28px;
  text-align:center; line-height:28px; font-weight:800; font-size:.9rem; margin-right:6px; }

.big-stat { background:#FFFFFF; border-radius: 22px; padding: 24px; text-align:center;
  border: 1px solid #F0E3D0; box-shadow: 0 6px 24px rgba(90,46,20,.08); }
.big-stat .num { font-family:'Fraunces',serif; font-size: 3rem; font-weight:800; color:#E8590C; line-height:1; }
.big-stat .lbl { color:#6B5A4B; margin-top:8px; font-size:1rem; }
.big-stat .delta { display:inline-block; margin-top:10px; background:#E6F4F1; color:#0F7A68; font-weight:800;
  border-radius:999px; padding:4px 12px; font-size:.9rem; }
.big-stat .delta.neg { background:#FDECEC; color:#B42318; }

.quote { font-family:'Fraunces',serif; font-size:1.5rem; color:#5A2E14; border-left: 6px solid #E8590C;
  padding: 6px 0 6px 18px; margin: 20px 0; }
.chip { display:inline-block; border-radius:999px; padding:4px 12px; font-weight:700; font-size:.85rem; margin-right:8px; }
.chip.teal { background:#E6F4F1; color:#0F7A68; } .chip.orange { background:#FDEBDD; color:#C2410C; }
.soft { color:#6B5A4B; }
.waiting { background:#FFF; border:2px dashed #F0C9A8; border-radius:22px; padding:34px; text-align:center; color:#8A5A35; }
.waiting .emoji { font-size:3rem; }
.live-pill { display:inline-block; background:#B42318; color:#fff; border-radius:999px; padding:4px 12px;
  font-weight:800; font-size:.85rem; animation: blink 1.4s infinite; }
@keyframes blink { 50% { opacity:.45; } }
[data-testid="stImage"] img { border-radius: 18px; }
.stButton button, .stToggle { font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_resource
def load_model(path: str):
    return YOLO(path)


def detect(model, frame, conf, imgsz, coco=False, augment=False):
    kwargs = dict(conf=conf, imgsz=imgsz, verbose=False, augment=augment)
    if coco:
        kwargs["classes"] = [0]
    r = model.predict(frame, **kwargs)[0]
    out = frame.copy()
    for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        cv2.rectangle(out, (x1, y1), (x2, y2), ORANGE_BGR, 2)
        cv2.putText(out, f"{c * 100:.0f}%", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE_BGR, 1, cv2.LINE_AA)
    return out, len(r.boxes)


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
                          (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), TEAL_BGR, 2)
            n += 1
    return img, n


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def card(emoji, title, text, step=None):
    num = f'<span class="step-num">{step}</span>' if step else ""
    return f'<div class="card"><div class="emoji">{emoji}</div><h3>{num}{title}</h3><p>{text}</p></div>'


def big_stat(num, label, delta=None, negative=False):
    d = f'<div class="delta {"neg" if negative else ""}">{delta}</div>' if delta else ""
    return f'<div class="big-stat"><div class="num">{num}</div><div class="lbl">{label}</div>{d}</div>'


def waiting(msg):
    st.markdown(f'<div class="waiting"><div class="emoji">⏳</div><h3>Almost there</h3><p>{msg}</p></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ hero
st.markdown("""
<div class="hero">
  <div class="drone">🚁</div>
  <div class="badge">🧡 Search &amp; Rescue · Capstone Project</div>
  <h1>Eyes in the Sky</h1>
  <p>After a disaster, every minute matters. Our drone software scans the ground from above and points
     rescuers to people who need help, running on an ordinary laptop with no special hardware.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🌍  The Story", "🏆  Proof It Works", "🔍  Spot the Survivors", "🛸  Fly the Drone"])

# ------------------------------------------------------------------ The Story
with tab1:
    st.markdown('<div class="quote">"From 50 metres up, a person is smaller than your fingernail on the screen."</div>',
                unsafe_allow_html=True)
    st.markdown("### How it works, in four steps")
    c = st.columns(4)
    steps = [
        ("🚁", "The drone looks", "A camera on the drone streams live video over WiFi as it flies over the area."),
        ("📡", "The laptop listens", "Our software catches every frame and reconnects on its own if the signal drops."),
        ("🧠", "The AI spots people", "Our model scans each frame and marks anyone it finds, even tiny figures far below."),
        ("🧭", "Rescuers act", "People are boxed on screen and every sighting is saved with a timestamp for the team."),
    ]
    for col, (e, t, d), i in zip(c, steps, range(1, 5)):
        col.markdown(card(e, t, d, i), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Why ordinary detectors miss people")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("""
<p class="soft" style="font-size:1.1rem">
In our test images, <b>more than 4 out of 10 people are smaller than 10 pixels</b> and almost all are under 32.
A standard detector looks at the picture in coarse blocks, so a small person gets blurred into
the background of rubble, grass and roofs.
</p>""", unsafe_allow_html=True)
    with right:
        st.markdown(big_stat("44%", "of people in our test images are smaller than 10×10 pixels"),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### What we changed")
    a, b, c3 = st.columns(3)
    a.markdown(card("🔎", "More pixels per person",
                    "Our detector looks at each frame in finer detail, so a 10-pixel survivor carries enough "
                    "detail to be recognised. This is where most of the gain comes from."), unsafe_allow_html=True)
    b.markdown(card("🧪", "We tested a heavier brain too",
                    "We also tried an extra detection layer plus attention (P2 + CBAM) from published papers. "
                    "It changed accuracy little and halved the speed, so we kept the lighter model."),
               unsafe_allow_html=True)
    c3.markdown(card("⚖️", "A fair test",
                     "Both models learned from the same photos, for the same time, then took the same exam "
                     "on photos <b>neither had ever seen</b>."), unsafe_allow_html=True)

# ------------------------------------------------------------------ Proof It Works
with tab2:
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    sel = json.loads(SELECTION.read_text()) if SELECTION.exists() else {}
    if "baseline:full@480" not in res or sel.get("proposed") not in res:
        waiting("The final evaluation is still running. The results will appear here automatically.")
    else:
        b, e = res["baseline:full@480"], res[sel["proposed"]]
        A, T = "all", "tiny (<10px)"
        st.markdown("### The exam results")
        st.markdown('<p class="soft">Both detectors were tested on 300 photos they had never seen before.</p>',
                    unsafe_allow_html=True)

        def delta(new, old, as_pts=True):
            d = (new - old) * 100
            return f"{'▲' if d >= 0 else '▼'} {abs(d):.1f} pts vs standard", d < 0

        c = st.columns(3)
        d1, n1 = delta(e[A]["recall50_conf0.25"], b[A]["recall50_conf0.25"])
        c[0].markdown(big_stat(f"{e[A]['recall50_conf0.25'] * 10:.1f}/10", "people on the ground that our detector finds",
                               d1, n1), unsafe_allow_html=True)
        d2, n2 = delta(e[T]["recall50_conf0.25"], b[T]["recall50_conf0.25"])
        c[1].markdown(big_stat(f"{e[T]['recall50_conf0.25'] * 100:.0f}%", "of the tiniest people (under 10 px) found",
                               d2, n2), unsafe_allow_html=True)
        d3, n3 = delta(e[A]["AP50"], b[A]["AP50"])
        c[2].markdown(big_stat(f"{e[A]['AP50'] * 100:.0f}", "overall accuracy score (mAP@50, out of 100)", d3, n3),
                      unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Side by side")
        rows = []
        for (bucket, k), name in [((A, "recall50_conf0.25"), "People found"), ((T, "recall50_conf0.25"), "Tiny people found"),
                                  ((A, "AP50"), "Overall accuracy"), ((A, "AP50_95"), "Box precision (strict)")]:
            rows += [{"Measure": name, "Model": "Standard", "Score": b[bucket][k] * 100},
                     {"Measure": name, "Model": "Ours", "Score": e[bucket][k] * 100}]
        chart = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=8, height=18).encode(
            y=alt.Y("Model:N", title=None, axis=alt.Axis(labelFontSize=13)),
            x=alt.X("Score:Q", title="Score (out of 100)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Model:N", scale=alt.Scale(domain=["Standard", "Ours"], range=["#D6C3AE", "#E8590C"]),
                            legend=alt.Legend(orient="top", title=None, labelFontSize=13)),
            row=alt.Row("Measure:N", title=None, header=alt.Header(labelFontSize=14, labelFontWeight="bold",
                                                                   labelAngle=0, labelAlign="left")),
            tooltip=["Measure", "Model", alt.Tooltip("Score:Q", format=".1f")],
        ).properties(height=50, width=700).configure_view(stroke=None).configure(background="transparent")
        st.altair_chart(chart)

        st.markdown("### What it costs")
        c = st.columns(3)
        c[0].markdown(big_stat(f"{1000 / e['latency_ms_640x480']:.0f}", "frames per second on a laptop CPU"),
                      unsafe_allow_html=True)
        c[1].markdown(big_stat("2.6 M", "parameters: small enough for any laptop"), unsafe_allow_html=True)
        c[2].markdown(big_stat("0", "graphics cards needed"), unsafe_allow_html=True)

        with st.expander("📋 Full details for the technical reviewers"):
            labels = {"baseline:full@480": "YOLO11n baseline (standard)", "enhanced:full@480": "+ P2 + CBAM",
                      sel["proposed"]: "Proposed (selected on validation)", sel.get("both"): "+ both"}
            table_rows = []
            for k, lab in labels.items():
                if k in res:
                    r = res[k]
                    table_rows.append({"Variant": lab, "mAP@50": r[A]["AP50"], "mAP@50:95": r[A]["AP50_95"],
                                       "Recall": r[A]["recall50_conf0.25"], "Recall <10px": r[T]["recall50_conf0.25"],
                                       "Latency ms (640x480)": r["latency_ms_640x480"]})
            st.dataframe(pd.DataFrame(table_rows).set_index("Variant"), use_container_width=True)
            st.caption(f"Selection rule: {sel.get('rule', '')}. Scored once on the held-out test split "
                       "(COCO-style, IoU 0.5; recall at confidence ≥ 0.25).")

# ------------------------------------------------------------------ Spot the Survivors
with tab3:
    st.markdown("### Can you spot them? Our model can.")
    st.markdown('<span class="chip teal">■ Real people (answer key)</span>'
                '<span class="chip orange">■ What the AI found</span>', unsafe_allow_html=True)
    images = sorted(TEST_IMAGES.glob("*.*"))
    c_a, c_b = st.columns([3, 1])
    upload = c_a.file_uploader("Or drop in your own aerial photo", type=["jpg", "jpeg", "png"])
    conf = c_b.slider("How sure before marking?", 0.05, 0.9, 0.25, 0.05, key="sbs_conf")
    img, label_file = None, Path("__none__")
    if upload is not None:
        img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
    elif images:
        idx = st.slider("Pick a scene", 1, len(images), 1) - 1
        img = cv2.imread(str(images[idx]))
        label_file = TEST_LABELS / f"{images[idx].stem}.txt"
    if img is not None:
        gt_img, n_gt = draw_ground_truth(img.copy(), label_file)
        cols = st.columns(3)
        cols[0].image(rgb(gt_img), use_container_width=True)
        cols[0].markdown(f"**👀 Answer key:** {n_gt} people" if n_gt else "**👀 Original photo**")
        for col, name, title in [(cols[1], BASELINE, "Standard model"), (cols[2], ENHANCED, "Our model")]:
            path, msz, maug = MODELS[name]
            if not path.exists():
                col.markdown('<div class="waiting"><div class="emoji">⏳</div><p>Still training…</p></div>',
                             unsafe_allow_html=True)
                continue
            out, n = detect(load_model(str(path)), img, conf, msz, augment=maug)
            col.image(rgb(out), use_container_width=True)
            col.markdown(f"**{'🌟 ' if name == ENHANCED else ''}{title}:** found {n}")

# ------------------------------------------------------------------ Fly the Drone
with tab4:
    # A rerun (any widget change) abandons the previous loop: make sure its reader thread is closed.
    _old = st.session_state.pop("_drone_link", None)
    if _old is not None:
        _old.stop(join=0.5)

    st.markdown("### Fly the drone")
    st.markdown('<p class="soft" style="font-size:1.05rem">Connect the drone to this laptop the way its own '
                'software does, and its live video appears right here. Our AI can then mark people on top of it, '
                'or you can switch the AI off and just watch the drone\'s camera.</p>', unsafe_allow_html=True)

    MODES = {
        "wifi": "📶  Drone WiFi stream (ESP32-CAM / WiFi FPV drone)",
        "window": "🪟  Drone's own app running on this laptop",
        "device": "🔌  Controller HDMI / USB capture card or webcam",
        "url": "🌐  Stream link (RTSP / RTMP / UDP)",
        "tello": "✈️  DJI / Ryze Tello",
        "file": "🎬  Recorded flight (backup)",
    }
    HOW = {
        "wifi": ("📶", "Join the drone's WiFi", "Connect this laptop to the drone's hotspot (ours: <b>SAR-DRONE</b>, "
                 "password <b>rescue1234</b>). Then press <b>Find my drone</b>."),
        "window": ("🪟", "Open the drone's app", "Start the drone's own PC software and show its camera view. "
                   "Keep that window <b>visible, not minimised</b>. We copy its picture into this page."),
        "device": ("🔌", "Plug in the controller", "Connect the remote's HDMI output through a USB capture card "
                   "(or use a webcam). It shows up as camera 0, 1, 2…"),
        "url": ("🌐", "Paste the stream link", "Point the drone app's <b>Live stream / RTMP</b> setting at this laptop, "
                "or paste the RTSP link printed in the drone's manual."),
        "tello": ("✈️", "Join TELLO-xxxx WiFi", "Connect to the Tello's WiFi. We switch its camera on automatically."),
        "file": ("🎬", "No drone? No problem", "Replay a recorded flight through exactly the same software."),
    }
    ss = st.session_state
    ss.setdefault("drone_mode", "wifi")
    ss.setdefault("drone_url", "http://192.168.4.1:81/stream")

    top_l, top_r = st.columns([3, 1])
    with top_r:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🔍  Find my drone", use_container_width=True):
            with st.spinner("Looking for drones on this network…"):
                ss["drone_scan"] = discover()
            hit = next((r for r in ss["drone_scan"] if r["found"]), None)
            if hit:
                ss["drone_mode"] = "tello" if hit["source"] == "tello" else (
                    "wifi" if hit["source"].startswith("http") else "url")
                ss["drone_url"] = ss["drone_stream_url"] = hit["source"]
    mode = top_l.selectbox("1 · How is the drone connected?", list(MODES), format_func=MODES.get, key="drone_mode")

    if "drone_scan" in ss:
        found = [r for r in ss["drone_scan"] if r["found"]]
        if found:
            st.success("Found: " + " · ".join(f"**{r['name']}** (`{r['source']}`)" for r in found))
        else:
            st.info("No drone answered on the usual addresses. Check the laptop is on the drone's WiFi, "
                    "or pick another connection type above.")

    e, t, d = HOW[mode]
    st.markdown(card(e, t, d), unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    source = None
    if mode == "wifi":
        source = st.text_input("Drone stream address", key="drone_url",
                               help="ESP32-CAM: http://192.168.4.1:81/stream")
    elif mode == "window":
        wins = [w for w in list_windows() if "Eyes in the Sky" not in w]
        w1, w2 = st.columns([4, 1])
        with w2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.button("↻ Refresh", use_container_width=True)
        if wins:
            pick = w1.selectbox("Which window shows the drone's camera?", wins)
            source = f"window:{pick}"
        else:
            w1.warning("No app windows found. Open the drone's software (not minimised) and press Refresh.")
    elif mode == "device":
        source = str(st.number_input("Camera / capture card number", 0, 9, 0))
    elif mode == "url":
        source = st.text_input("Stream link", key="drone_stream_url",
                               placeholder="rtsp://192.168.1.1:7070/webcam  or  rtmp://127.0.0.1/live/drone")
    elif mode == "tello":
        source = "tello"
    else:
        clips = sorted(str(p.relative_to(ROOT)) for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv")
                       for p in (ROOT / "demo_output").glob(ext))
        source = st.selectbox("Recorded flight", clips) if clips else st.text_input("Path to a video file")

    a1, a2 = st.columns([1, 2])
    ai_on = a1.toggle("🧠  AI survivor spotting", value=True,
                      help="Off = show only the drone's own camera picture, untouched.")
    with a2.expander("⚙️ AI settings", expanded=False):
        model_name = st.selectbox("Which eye to use?", list(MODELS.keys()),
                                  help="Flying low (people look big)? The everyday person-spotter often does better.")
        live_conf = st.slider("How sure before marking?", 0.05, 0.9, 0.35, 0.05, key="live_conf")
        save_shots = st.checkbox("Save a photo whenever people are spotted", value=True)

    run = st.toggle("▶  Start watching", value=False, disabled=not source)
    status_box, frame_box, info_box = st.empty(), st.empty(), st.empty()

    if not run:
        frame_box.markdown('<div class="waiting"><div class="emoji">🛸</div><h3>Waiting for take-off</h3>'
                           '<p>Pick how the drone is connected, then switch on <b>Start watching</b>.</p></div>',
                           unsafe_allow_html=True)
    else:
        model = live_imgsz = live_aug = coco = None
        if ai_on:
            path, live_imgsz, live_aug = MODELS[model_name]
            if not path.exists():
                st.error("That model's weights are missing. Pick the everyday person-spotter in AI settings.")
                st.stop()
            model, coco = load_model(str(path)), path.name == "yolo11n.pt"

        try:
            link = DroneLink(source if mode != "file" else str(ROOT / source)).start()
        except ValueError as err:
            st.error(f"📡 {err}")
            st.stop()
        ss["_drone_link"] = link

        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = ROOT / "logs" / f"drone_session_{stamp}.csv"
        shot_dir = ROOT / "logs" / f"sightings_{stamp}"
        PILL = {"live": ("● LIVE", "#B42318"), "connecting": ("◌ CONNECTING", "#8A5A35"),
                "reconnecting": ("◌ SIGNAL LOST, RECONNECTING", "#C2410C"), "ended": ("■ FINISHED", "#0F7A68"),
                "failed": ("✕ NOT CONNECTED", "#6B5A4B"), "stopped": ("■ STOPPED", "#6B5A4B")}

        def show_status(s, n=None, ai_fps=0.0):
            label, color = PILL.get(s.status, (s.status.upper(), "#6B5A4B"))
            if s.status == "live":
                alert = ("Scanning…" if not n else f"🧡 <b>{n} {'person' if n == 1 else 'people'} spotted!</b>") \
                    if ai_on else "Drone camera"
                detail = (f"{s.resolution} · drone {s.fps_in:.0f} fps"
                          + (f" · AI {ai_fps:.1f} fps" if ai_on else "")
                          + (f" · delay {s.age_ms:.0f} ms" if s.age_ms is not None else "")
                          + (f" · reconnected {s.reconnects}×" if s.reconnects else ""))
            else:
                alert = link.spec.label
                detail = ("End of recording" if s.status == "ended"
                          else s.last_error or "Waiting for the first picture…")
            status_box.markdown(
                f'<span class="live-pill" style="background:{color}">{label}</span> &nbsp; {alert} '
                f'&nbsp;·&nbsp; <span class="soft">{detail}</span>', unsafe_allow_html=True)

        last_idx, shown, ai_fps, t_prev, peak, sightings, last_shot = 0, 0, 0.0, time.time(), 0, 0, 0.0
        try:
            with open(log_file, "w") as fh:
                fh.write("utc_time,frame,people,source\n")
                while True:
                    s = link.stats()
                    fr = link.latest()
                    if s.status in ("ended", "failed"):
                        show_status(s)
                        break
                    if fr is None or fr.index == last_idx:
                        if s.status != "live":
                            show_status(s)
                        time.sleep(0.02)
                        continue
                    last_idx = fr.index
                    n = None
                    out = fr.image
                    if ai_on:
                        out, n = detect(model, fr.image, live_conf, live_imgsz, coco=coco, augment=live_aug)
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
