"""Panoramic X-ray cavity finder: Streamlit front end for a YOLO model (best.pt)."""
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "best.pt"
MAX_SIDE = 2000

st.set_page_config(
    page_title="Panoramic X-ray cavity finder", page_icon="🦷", layout="wide"
)

# ----------------------------------------------------------------------------
# Look and feel
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Public+Sans:wght@400;500;600&display=swap');
.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input { font-family: 'Public Sans', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

.head { margin: 0 0 1.6rem; }
.head h1 { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 2.7rem;
  line-height: 1.05; letter-spacing: -0.02em; margin: 0 0 .55rem; padding: 0; color: #13212C; }
.head p { font-size: 1.05rem; color: #42525C; max-width: 62ch; margin: 0; }

.side-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.25rem; margin: 0 0 .6rem; }

.sumline { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600; font-size: 1.95rem;
  line-height: 1.15; letter-spacing: -0.01em; color: #13212C; margin: .6rem 0 .3rem; }
.substats { color: #42525C; font-size: .95rem; margin: 0 0 1.2rem; }

.ruler { background: #FFFFFF; border: 1px solid #C9D5D2; border-radius: 14px; padding: 18px 24px 16px; margin-bottom: 1.4rem; }
.ruler h4 { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.1rem; margin: 0 0 .15rem; padding: 0; color: #13212C; }
.ruler .hint { color: #51626C; font-size: .86rem; margin: 0 0 .9rem; max-width: 90ch; }
.plot { position: relative; margin: 0 14px; }
.mk { position: absolute; width: 26px; }
.mk b { display: flex; align-items: center; justify-content: center; width: 26px; height: 26px; border-radius: 50%;
  background: var(--c); color: var(--ink); font-size: .78rem; font-weight: 700; box-shadow: 0 0 0 2px #FFFFFF, 0 0 0 3px rgba(19,33,44,.35); }
.mk s { position: absolute; top: 26px; left: 12px; width: 2px; background: var(--c); box-shadow: 0 0 0 .5px rgba(19,33,44,.35); }
.bands { display: flex; height: 8px; border-radius: 4px; overflow: hidden; position: relative; }
.bands i { display: block; height: 100%; }
.thr { position: absolute; top: 0; bottom: 0; width: 0; border-left: 2px dashed #13212C; z-index: 2; }
.zones { display: flex; margin: .4rem 14px 0; font-size: .8rem; color: #51626C; }
.zones span { display: block; }

.panel-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.35rem; margin: 0 0 .5rem; color: #13212C; }
.list { max-height: 560px; overflow-y: auto; border-top: 1px solid #C9D5D2; margin-bottom: 1rem; }
.row { display: grid; grid-template-columns: 34px 1fr; gap: 12px; padding: 14px 4px; border-bottom: 1px solid #C9D5D2; }
.row .n { width: 30px; height: 30px; border-radius: 50%; background: var(--c); color: var(--ink); font-weight: 700;
  font-size: .9rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 0 1px rgba(19,33,44,.35); }
.row .line { display: flex; justify-content: space-between; align-items: baseline; }
.row .line b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.1rem; color: #13212C; }
.row .pct { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.5rem; font-variant-numeric: tabular-nums; color: #13212C; }
.meter { height: 8px; border-radius: 4px; background: #D9E2DF; margin: .35rem 0 .45rem; overflow: hidden; }
.meter i { display: block; height: 100%; background: var(--c); border-radius: 4px; box-shadow: inset 0 0 0 1px rgba(19,33,44,.3); }
.row .sub { color: #51626C; font-size: .86rem; }

.detail .big { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 3.2rem; line-height: 1; letter-spacing: -0.02em; color: #13212C; }
.detail .cls { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600; font-size: 1.3rem; margin: .2rem 0 .8rem; color: #13212C; }
.detail dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: .35rem 1rem; font-size: .95rem; }
.detail dt { color: #51626C; }
.detail dd { margin: 0; color: #13212C; }

.empty { border: 1.5px dashed #9DB1AC; border-radius: 14px; padding: 1.2rem 1.4rem; color: #42525C; max-width: 70ch; }
.fineprint { color: #51626C; font-size: .85rem; max-width: 80ch; margin-top: 2rem; }

[data-testid="stImage"] { background: #0E1114; padding: 10px; border-radius: 14px; }
[data-testid="stImage"] img { border-radius: 8px; }
[data-testid="stImageCaption"] { color: #9FB0B8; }

@media (max-width: 900px) {
  .head h1 { font-size: 2rem; }
  .sumline { font-size: 1.5rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Class styling and small helpers
# ----------------------------------------------------------------------------
STYLES = {
    "cavity": {"color": "#FFB020", "ink": "#3A2600", "name": "Cavity",
               "one": "possible cavity", "many": "possible cavities"},
    "filling": {"color": "#62C9F7", "ink": "#06293A", "name": "Filling",
                "one": "filling", "many": "fillings"},
}
FALLBACK = [("#B9A2FF", "#1E1040"), ("#8BE8B0", "#06301A"), ("#FF8FA3", "#3D0A17")]


def style_for(cls_name):
    key = cls_name.lower().strip()
    if key in ("filling", "fillings"):
        key = "filling"
    if key in STYLES:
        return {**STYLES[key], "key": key}
    color, ink = FALLBACK[sum(map(ord, key)) % len(FALLBACK)]
    return {"color": color, "ink": ink, "name": cls_name, "one": key,
            "many": key + "s", "key": key}


def level_of(conf):
    if conf >= 0.70:
        return "High confidence"
    if conf >= 0.40:
        return "Medium confidence"
    return "Low confidence, check closely"


def position_of(box, size):
    w, h = size
    cx = (box[0] + box[2]) / 2 / w
    cy = (box[1] + box[3]) / 2 / h
    return f"{'Upper' if cy < 0.5 else 'Lower'} {'left' if cx < 0.5 else 'right'} of image"


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


@st.cache_resource(show_spinner=False)
def get_font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return YOLO(str(MODEL_PATH))


def prepare(raw):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


@st.cache_data(show_spinner=False)
def detect(raw, conf, iou):
    model = load_model()
    img = prepare(raw)
    t0 = time.perf_counter()
    res = model.predict(np.array(img)[:, :, ::-1], conf=conf, iou=iou, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        dets.append({"cls": model.names[int(b.cls)], "conf": float(b.conf),
                     "box": [x1, y1, x2, y2]})
    dets.sort(key=lambda d: -d["conf"])
    return dets, img.size, ms


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def annotate(img, dets, thickness=3, show_labels=True):
    """Corner brackets plus a numbered score tag. Tags dodge each other."""
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    W, H = base.size
    t = max(2, round(thickness * W / 800))
    font = get_font(max(13, round(W / 62)))
    placed = []

    for det in dets:
        x1, y1, x2, y2 = det["box"]
        rgb = hex_rgb(det["style"]["color"])
        ink = hex_rgb(det["style"]["ink"])

        d.rectangle([x1, y1, x2, y2], outline=rgb + (150,), width=1)
        ln = max(8, int(min(x2 - x1, y2 - y1) * 0.32))
        for cx, cy, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            d.line([(cx, cy), (cx + sx * ln, cy)], fill=rgb + (255,), width=t)
            d.line([(cx, cy), (cx, cy + sy * ln)], fill=rgb + (255,), width=t)

        if not show_labels:
            continue
        text = f"{det['n']}  {det['conf'] * 100:.0f}%"
        l, tp, r, b = d.textbbox((0, 0), text, font=font)
        pad = max(3, t)
        hpad = pad * 2
        lw, lh = (r - l) + 2 * hpad, (b - tp) + 2 * pad
        candidates = [(x1, y1 - lh - 2), (x1, y2 + 2), (x2 - lw, y1 - lh - 2),
                      (x2 - lw, y2 + 2), (x1, y1 + 2)]
        pos = candidates[0]
        for cx, cy in candidates:
            rect = (cx, cy, cx + lw, cy + lh)
            inside = rect[0] >= 0 and rect[1] >= 0 and rect[2] <= W and rect[3] <= H
            if inside and not any(overlap(rect, p) for p in placed):
                pos = (cx, cy)
                break
        rect = (pos[0], pos[1], pos[0] + lw, pos[1] + lh)
        placed.append(rect)
        d.rounded_rectangle(rect, radius=lh // 2, fill=rgb + (255,))
        d.text((pos[0] + hpad - l, pos[1] + pad - tp), text, font=font, fill=ink + (255,))

    return Image.alpha_composite(base, layer).convert("RGB")


def crop_finding(img, det, boost):
    x1, y1, x2, y2 = det["box"]
    pad = max(x2 - x1, y2 - y1) * 0.9
    box = (max(0, int(x1 - pad)), max(0, int(y1 - pad)),
           min(img.width, int(x2 + pad)), min(img.height, int(y2 + pad)))
    c = img.crop(box)
    if boost:
        c = ImageOps.autocontrast(c.convert("L"), cutoff=1).convert("RGB")
    s = max(1.0, 720 / c.width)
    if s > 1:
        c = c.resize((round(c.width * s), round(c.height * s)), Image.LANCZOS)
    shifted = dict(det, box=[(x1 - box[0]) * s, (y1 - box[1]) * s,
                             (x2 - box[0]) * s, (y2 - box[1]) * s])
    return annotate(c, [shifted], thickness=3, show_labels=False)


# ----------------------------------------------------------------------------
# HTML builders (kept flat: indented lines would render as code in markdown)
# ----------------------------------------------------------------------------
def summary_sentence(dets):
    if not dets:
        return "Nothing found at this threshold."
    counts, order = {}, []
    for d in dets:
        k = d["style"]["key"]
        if k not in counts:
            counts[k] = [0, d["style"]]
            order.append(k)
        counts[k][0] += 1
    parts = [f"{counts[k][0]} {counts[k][1]['one'] if counts[k][0] == 1 else counts[k][1]['many']}"
             for k in order]
    text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"Found {text}."


def ruler_html(dets, threshold):
    placed = []
    for det in dets:
        x = det["conf"] * 100
        lane = 0
        while any(abs(x - px) < 5.5 for px, pl in placed if pl == lane):
            lane += 1
        placed.append((x, lane))
    n_lanes = max(l for _, l in placed) + 1
    area = n_lanes * 32
    markers = ""
    for det, (x, lane) in zip(dets, placed):
        stem = area - (lane * 32 + 26)
        markers += (f'<div class="mk" style="left:calc({x}% - 13px);top:{lane * 32}px;'
                    f'--c:{det["style"]["color"]};--ink:{det["style"]["ink"]}">'
                    f'<b>{det["n"]}</b><s style="height:{stem}px"></s></div>')
    return (
        '<div class="ruler"><h4>Confidence at a glance</h4>'
        f'<p class="hint">Each numbered marker sits at the model\'s score for that finding. '
        f'The dashed line is your threshold ({threshold:.0%}). A score is the model\'s own estimate, '
        'not the chance that a finding is real.</p>'
        f'<div class="plot"><div style="position:relative;height:{area}px">{markers}</div>'
        '<div class="bands"><i style="width:40%;background:#D7E0DD"></i>'
        '<i style="width:30%;background:#A8C5C4"></i><i style="width:30%;background:#0F6E7A"></i></div>'
        f'<div class="thr" style="left:{threshold * 100}%"></div></div>'
        '<div class="zones"><span style="width:40%">Low, under 40%</span>'
        '<span style="width:30%">Medium, 40 to 70%</span><span style="width:30%">High, 70% and up</span></div>'
        '</div>'
    )


def rows_html(dets, size):
    out = '<div class="list">'
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        pct = d["conf"] * 100
        out += (f'<div class="row" style="--c:{d["style"]["color"]};--ink:{d["style"]["ink"]}">'
                f'<span class="n">{d["n"]}</span><div>'
                f'<div class="line"><b>{d["style"]["name"]}</b><span class="pct">{pct:.0f}%</span></div>'
                f'<div class="meter"><i style="width:{pct:.1f}%"></i></div>'
                f'<div class="sub">{d["level"]}. {position_of(d["box"], size)}, '
                f'{x2 - x1:.0f} x {y2 - y1:.0f} px</div></div></div>')
    return out + "</div>"


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="head"><h1>Panoramic X-ray cavity finder</h1>'
    '<p>Upload a panoramic dental X-ray. The model marks cavities and fillings and shows '
    'how confident it is in each mark.</p></div>',
    unsafe_allow_html=True,
)

try:
    model = load_model()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the model at {MODEL_PATH.name}. Put best.pt next to app.py. ({exc})")
    st.stop()

class_names = list(model.names.values())

with st.sidebar:
    st.markdown('<p class="side-title">Settings</p>', unsafe_allow_html=True)
    conf_thr = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05,
        help="Marks scoring below this are hidden. Lower it to see weaker candidates, "
             "raise it to keep only the strongest.",
    )
    shown = st.multiselect("Show", class_names, default=class_names)
    show_labels = st.checkbox("Number and score on marks", value=True)
    thickness = st.slider("Mark thickness", 1, 6, 3)
    with st.expander("Advanced"):
        iou_thr = st.slider(
            "Overlap merging (IoU)", 0.1, 0.9, 0.45, 0.05,
            help="Lower values merge overlapping boxes more aggressively.",
        )
    st.caption("Classes in this model: " + ", ".join(class_names))

upload = st.file_uploader("Panoramic X-ray (JPG or PNG)", type=["jpg", "jpeg", "png"])

if upload is None:
    st.markdown(
        '<div class="empty">Drop an X-ray above to begin. You get the image with every finding '
        'marked, a confidence reading for each mark, and a close-up viewer for checking '
        'individual teeth.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

raw = upload.getvalue()
with st.spinner("Reading the X-ray..."):
    all_dets, size, ms = detect(raw, conf_thr, iou_thr)
img = prepare(raw)

dets = [d for d in all_dets if d["cls"] in shown]
for i, d in enumerate(dets, 1):
    d["n"] = i
    d["style"] = style_for(d["cls"])
    d["level"] = level_of(d["conf"])

st.markdown(f'<div class="sumline">{summary_sentence(dets)}</div>', unsafe_allow_html=True)
if dets:
    avg = sum(d["conf"] for d in dets) / len(dets)
    st.markdown(
        f'<p class="substats">Highest confidence {max(d["conf"] for d in dets):.0%}, '
        f'average {avg:.0%}. Analysed a {size[0]} x {size[1]} px image in {ms:.0f} ms.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(ruler_html(dets, conf_thr), unsafe_allow_html=True)
else:
    st.markdown(
        '<p class="substats">Try lowering the confidence threshold in the sidebar, '
        'or check that all classes are selected under Show.</p>',
        unsafe_allow_html=True,
    )

marked = annotate(img, dets, thickness, show_labels)
left, right = st.columns([1.75, 1], gap="large")

with left:
    view = st.radio("View", ["Marked", "Side by side", "Original"], horizontal=True,
                    label_visibility="collapsed")
    if view == "Marked":
        st.image(marked, use_container_width=True)
    elif view == "Original":
        st.image(img, use_container_width=True)
    else:
        c1, c2 = st.columns(2)
        c1.image(img, caption="Original", use_container_width=True)
        c2.image(marked, caption="Marked", use_container_width=True)

with right:
    st.markdown('<div class="panel-title">Findings</div>', unsafe_allow_html=True)
    if dets:
        st.markdown(rows_html(dets, size), unsafe_allow_html=True)
        buf = io.BytesIO()
        marked.save(buf, "PNG")
        st.download_button("Download marked image", buf.getvalue(),
                           file_name="xray_marked.png", mime="image/png",
                           use_container_width=True)
        table = pd.DataFrame([{
            "number": d["n"], "class": d["style"]["name"],
            "confidence": round(d["conf"], 4), "level": d["level"],
            "position": position_of(d["box"], size),
            "x1": round(d["box"][0]), "y1": round(d["box"][1]),
            "x2": round(d["box"][2]), "y2": round(d["box"][3]),
        } for d in dets])
        st.download_button("Download findings (CSV)", table.to_csv(index=False),
                           file_name="findings.csv", mime="text/csv",
                           use_container_width=True)
    else:
        st.write("No findings to list.")

if dets:
    st.markdown("---")
    st.markdown('<div class="panel-title">Inspect a finding</div>', unsafe_allow_html=True)
    pick_col, boost_col = st.columns([2, 1])
    pick = pick_col.selectbox(
        "Finding", [d["n"] for d in dets],
        format_func=lambda n: f"#{n}  {dets[n - 1]['style']['name']}  {dets[n - 1]['conf']:.0%}",
    )
    boost = boost_col.checkbox("Boost contrast in close-up", value=False)
    sel = dets[pick - 1]
    x1, y1, x2, y2 = sel["box"]

    crop_col, info_col = st.columns([1.4, 1], gap="large")
    crop_col.image(crop_finding(img, sel, boost), use_container_width=True)
    info_col.markdown(
        f'<div class="detail"><div class="big">{sel["conf"]:.0%}</div>'
        f'<div class="cls">{sel["style"]["name"]}, mark {sel["n"]}</div>'
        f'<dl><dt>Reading</dt><dd>{sel["level"]}</dd>'
        f'<dt>Position</dt><dd>{position_of(sel["box"], size)}</dd>'
        f'<dt>Size</dt><dd>{x2 - x1:.0f} x {y2 - y1:.0f} px</dd></dl></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="fineprint">This tool supports screening and teaching. It does not diagnose. '
    'Position labels describe where a mark sits in the image, not tooth numbers. '
    'A dentist should review every finding.</p>',
    unsafe_allow_html=True,
)