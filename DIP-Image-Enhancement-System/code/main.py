import streamlit as st
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(
    page_title="DIP — Smart Image Enhancement",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #060612;
    color: #e0e6ff;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1f 0%, #060612 100%);
    border-right: 1px solid #1a1a3e;
}
[data-testid="stSidebar"] > div { padding: 1rem; }

.sidebar-logo {
    text-align: center;
    padding: 1.2rem 0.5rem 1.5rem;
    border-bottom: 1px solid #1a1a3e;
    margin-bottom: 1rem;
}
.sidebar-logo h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    color: #00f5d4;
    letter-spacing: 2px;
    margin: 0;
    text-shadow: 0 0 20px #00f5d466;
}
.sidebar-logo p { font-size: 0.68rem; color: #4a4a8a; margin: 0.3rem 0 0; letter-spacing: 1px; }

.stButton > button {
    width: 100%;
    background: transparent;
    border: 1px solid #1e1e4a;
    color: #8888bb;
    border-radius: 4px;
    padding: 0.55rem 1rem;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
    text-align: left;
    margin-bottom: 0.3rem;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #00f5d415, #7b2fff15);
    border-color: #00f5d4;
    color: #00f5d4;
    box-shadow: 0 0 12px #00f5d422;
    transform: translateX(3px);
}
.stButton > button:focus {
    background: linear-gradient(90deg, #00f5d420, #7b2fff20);
    border-color: #00f5d4;
    color: #00f5d4;
}

.main-header {
    background: linear-gradient(135deg, #0a0a1f 0%, #100820 100%);
    border: 1px solid #1a1a3e;
    border-radius: 8px;
    padding: 1.2rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00f5d4, #7b2fff, transparent);
}
.main-header h2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    color: #00f5d4;
    margin: 0 0 0.2rem;
    letter-spacing: 2px;
    text-shadow: 0 0 30px #00f5d455;
}
.main-header p { color: #4a4a8a; font-size: 0.8rem; margin: 0; letter-spacing: 1px; }

.phase-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.9rem 1.5rem;
    background: linear-gradient(90deg, #0a0a1f, #060612);
    border: 1px solid #1a1a3e;
    border-left: 3px solid #00f5d4;
    border-radius: 6px;
    margin-bottom: 1.2rem;
}
.phase-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #00f5d4;
    background: #00f5d415;
    border: 1px solid #00f5d433;
    border-radius: 3px;
    padding: 0.2rem 0.5rem;
    letter-spacing: 1px;
    white-space: nowrap;
}
.phase-title { font-size: 1.2rem; font-weight: 700; color: #e0e6ff; margin: 0; letter-spacing: 1px; }
.phase-desc { font-size: 0.78rem; color: #4a4a8a; margin: 0; }

.info-card {
    background: #0a0a1f;
    border: 1px solid #1a1a3e;
    border-radius: 6px;
    padding: 0.8rem 1.2rem;
}
.info-card .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #4a4a8a;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.info-card .value { font-size: 1.1rem; font-weight: 700; color: #00f5d4; }

.matrix-box {
    background: #0a0a1f;
    border: 1px solid #1a1a3e;
    border-radius: 6px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #7b2fff;
    line-height: 1.8;
    overflow-x: auto;
}

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #4a4a8a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: #1a1a3e; }

.styled-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 1rem 0; }
.styled-table th {
    background: #0e0e2a;
    color: #00f5d4;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1px;
    padding: 0.6rem 1rem;
    border: 1px solid #1a1a3e;
    text-align: left;
}
.styled-table td { padding: 0.55rem 1rem; border: 1px solid #1a1a3e; color: #9999cc; }
.styled-table tr:nth-child(even) td { background: #0a0a1f; }
.styled-table tr:hover td { background: #0e0e2a; color: #e0e6ff; }

hr { border-color: #1a1a3e; }

.stTabs [data-baseweb="tab-list"] { background: #0a0a1f; border-bottom: 1px solid #1a1a3e; gap: 0; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 1px;
    cNolor: #4a4a8a;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] { color: #00f5d4 !important; border-bottom-color: #00f5d4 !important; }

[data-testid="caption"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #4a4a8a;
    letter-spacing: 1px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────

def pil_to_cv(pil_img):
    arr = np.array(pil_img.convert('RGB'))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0a0a1f', edgecolor='none', dpi=110)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def phase_header(num, title, desc):
    st.markdown(f"""
    <div class="phase-header">
        <span class="phase-num">PHASE {num}</span>
        <div>
            <div class="phase-title">{title}</div>
            <div class="phase-desc">{desc}</div>
        </div>
    </div>""", unsafe_allow_html=True)

def sec(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h1>⚡ DIP </h1>
        <p>SMART IMAGE ENHANCEMENT SYSTEM</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","jfif","png","bmp","tiff","tif","webp"],
        label_visibility="collapsed"
    )
    st.markdown('<div style="font-family:JetBrains Mono;font-size:0.62rem;color:#2a2a5a;text-align:center;margin:-0.4rem 0 1rem;letter-spacing:1px;">JPG · JFIF · PNG · BMP · TIFF · WEBP</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-family:JetBrains Mono;font-size:0.62rem;color:#2a2a5a;letter-spacing:2px;margin-bottom:0.5rem;">NAVIGATION</div>', unsafe_allow_html=True)

    phases = {
    "🏠  Home": "home",
    "📷  Image Acquisition": "p61",
    "📐  Sampling & Quantization": "p62",
    "🔄  Geometric Transformations": "p63",
    "🌗  Intensity Transformations": "p64",
    "📊  Histogram Processing": "p65",
    "⚡  Image Enhancement Pipeline": "p66",
}

    if "active" not in st.session_state:
        st.session_state.active = "home"

    for label, key in phases.items():
        if st.button(label, key=f"btn_{key}"):
            st.session_state.active = key

    st.markdown("---")
    st.markdown("""
    <div style="font-family:JetBrains Mono;font-size:0.62rem;color:#2a2a5a;text-align:center;line-height:1.8;letter-spacing:1px;">
        Ahsan · 235139<br>Air University
    </div>""", unsafe_allow_html=True)


# ── Main Header ────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h2>SMART IMAGE ENHANCEMENT SYSTEM</h2>
    <p>DIGITAL IMAGE PROCESSING · AIR UNIVERSITY</p>
</div>""", unsafe_allow_html=True)

active = st.session_state.active

if uploaded is None and active != "home":
    st.markdown("""
    <div style="text-align:center;padding:3rem;background:#0a0a1f;border:1px dashed #1a1a3e;border-radius:8px;">
        <div style="font-size:2.5rem;margin-bottom:1rem;">📁</div>
        <div style="font-family:JetBrains Mono;font-size:0.8rem;color:#4a4a8a;letter-spacing:2px;">
            UPLOAD AN IMAGE FROM THE SIDEBAR TO BEGIN
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

if uploaded is not None:
    pil_img  = Image.open(uploaded).convert('RGB')
    img_bgr  = pil_to_cv(pil_img)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ══════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════
if active == "home":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
        <div style="background:#0a0a1f;border:1px solid #1a1a3e;border-radius:8px;padding:1.5rem;">
            <div style="font-family:JetBrains Mono;font-size:0.68rem;color:#00f5d4;letter-spacing:2px;margin-bottom:1rem;">PROJECT OVERVIEW</div>
            <div style="color:#8888bb;font-size:0.95rem;line-height:1.9;">
                A complete image enhancement pipeline integrating .
                Upload any image and navigate each phase using the sidebar.
            </div>
            <div style="margin-top:1.5rem;">
                <div style="font-family:JetBrains Mono;font-size:0.65rem;color:#2a2a5a;letter-spacing:2px;margin-bottom:0.8rem;">PIPELINE</div>
                <div style="display:flex;flex-direction:column;gap:0.4rem;">
                    <div style="background:#0e0e2a;border-left:3px solid #00f5d4;padding:0.55rem 1rem;border-radius:4px;font-size:0.88rem;color:#ccccee;">📷 &nbsp;Image Acquisition &amp; Understanding</div>
                    <div style="background:#0e0e2a;border-left:3px solid #00b4a4;padding:0.55rem 1rem;border-radius:4px;font-size:0.88rem;color:#ccccee;">📐 &nbsp;Sampling &amp; Quantization</div>
                    <div style="background:#0e0e2a;border-left:3px solid #7b2fff;padding:0.55rem 1rem;border-radius:4px;font-size:0.88rem;color:#ccccee;">🔄 &nbsp;Geometric Transformations</div>
                    <div style="background:#0e0e2a;border-left:3px solid #ff6b9d;padding:0.55rem 1rem;border-radius:4px;font-size:0.88rem;color:#ccccee;">🌗 &nbsp;Intensity Transformations</div>
                    <div style="background:#0e0e2a;border-left:3px solid #ffbe0b;padding:0.55rem 1rem;border-radius:4px;font-size:0.88rem;color:#ccccee;">📊 &nbsp;Histogram Processing</div>
                    <div style="background:#0e0e2a;border-left:3px solid #00f5d4;padding:0.55rem 1rem;border-radius:4px;font-size:0.88rem;color:#ccccee;">⚡ &nbsp;Final Enhanced Output</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        if uploaded:
            sec("UPLOADED IMAGE")
            st.image(img_rgb, use_container_width=True)
            h, w = img_gray.shape
            c1, c2, c3 = st.columns(3)
            fmt = uploaded.name.split('.')[-1].upper()
            with c1: st.markdown(f'<div class="info-card"><div class="label">Width</div><div class="value">{w}px</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="info-card"><div class="label">Height</div><div class="value">{h}px</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="info-card"><div class="label">Format</div><div class="value">{fmt}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:5rem 2rem;background:#0a0a1f;border:2px dashed #1a1a3e;border-radius:8px;">
                <div style="font-size:3rem;margin-bottom:1rem;">⬆</div>
                <div style="font-family:JetBrains Mono;color:#4a4a8a;font-size:0.75rem;letter-spacing:2px;">
                    UPLOAD AN IMAGE FROM THE SIDEBAR<br><br>
                    <span style="color:#2a2a5a">JPG · JFIF · PNG · BMP · TIFF · WEBP</span>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 6.1
# ══════════════════════════════════════════════════════════════════
elif active == "p61":
    phase_header("1", "Image Acquisition & Understanding", "Load image, convert to grayscale, and analyze properties")

    h, w  = img_gray.shape
    dtype = str(img_gray.dtype)
    ch    = img_bgr.shape[2]

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip([c1,c2,c3,c4],
                                ["Width","Height","Channels","Data Type"],
                                [f"{w}px", f"{h}px", str(ch), dtype]):
        with col:
            st.markdown(f'<div class="info-card"><div class="label">{label}</div><div class="value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        sec("ORIGINAL RGB")
        st.image(img_rgb, use_container_width=True, caption="Original Color Image")
    with col2:
        sec("GRAYSCALE")
        st.image(img_gray, use_container_width=True, clamp=True, caption="Converted to Grayscale")

    sec("PIXEL MATRIX — TOP-LEFT 8×8")
    matrix_str = ""
    for row in img_gray[:8, :8]:
        matrix_str += "  ".join(f"{v:3d}" for v in row) + "\n"
    st.markdown(f'<div class="matrix-box"><pre>{matrix_str}</pre></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 6.2
# ══════════════════════════════════════════════════════════════════
elif active == "p62":
    phase_header("2", "Sampling & Quantization", "Resize images and reduce bit depth")

    tab1, tab2 = st.tabs(["  📐 SAMPLING  ", "  🎨 QUANTIZATION  "])

    with tab1:
        scales = [0.25, 0.5, 1.0, 1.5, 2.0]
        labels = ["0.25×", "0.50×", "1.00× (Original)", "1.50×", "2.00×"]
        cols   = st.columns(5)
        rows   = ""
        for i, (s, lb) in enumerate(zip(scales, labels)):
            nh = int(img_gray.shape[0] * s)
            nw = int(img_gray.shape[1] * s)
            r  = cv2.resize(img_gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
            q  = ["Very Low","Low","Original","Good","High"][i]
            rows += f"<tr><td>{lb}</td><td>{nw} × {nh}</td><td>{nw*nh:,}</td><td>{q}</td></tr>"
            with cols[i]:
                st.image(r, use_container_width=True, clamp=True, caption=lb)

        sec("RESOLUTION VS QUALITY")
        st.markdown(f"""
        <table class="styled-table">
            <tr><th>Scale</th><th>Resolution</th><th>Total Pixels</th><th>Quality</th></tr>
            {rows}
        </table>""", unsafe_allow_html=True)

    with tab2:
        cols = st.columns(3)
        rows = ""
        for i, bits in enumerate([8, 4, 2]):
            step  = max(1, 256 // (2**bits))
            q_img = (img_gray.astype(np.uint16) // step * step).clip(0,255).astype(np.uint8)
            rows += f"<tr><td>{bits}-bit</td><td>{2**bits}</td><td>{step}</td></tr>"
            with cols[i]:
                st.image(q_img, use_container_width=True, clamp=True, caption=f"{bits}-bit · {2**bits} levels")

        sec("BIT DEPTH TABLE")
        st.markdown(f"""
        <table class="styled-table">
            <tr><th>Bit Depth</th><th>Intensity Levels</th><th>Step Size</th></tr>
            {rows}
        </table>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 6.3
# ══════════════════════════════════════════════════════════════════
elif active == "p63":
    phase_header("3", "Geometric Transformations", "Apply rotation, translation, and shearing")

    tab1, tab2, tab3 = st.tabs(["  🔄 ROTATION  ", "  ↔ TRANSLATE & SHEAR  ", "  🔁 INVERSE  "])
    h, w = img_gray.shape

    with tab1:
        angles = [30, 45, 60, 90, 120, 150, 180]
        cols = st.columns(4)

        for i, angle in enumerate(angles):
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)

            with cols[i % 4]:
                st.image(
                    cv2.warpAffine(img_gray, M, (w, h)),
                    use_container_width=True,
                    clamp=True,
                    caption=f"{angle}°"
                )

    with tab2:
        c1, c2, c3 = st.columns(3)
        M_t = np.float32([[1,0,50],[0,1,30]])
        M_s = np.float32([[1,0.3,0],[0,1,0]])
        nw  = w + int(0.3*h)
        with c1:
            sec("ORIGINAL")
            st.image(img_gray, use_container_width=True, clamp=True, caption="Original")
        with c2:
            sec("TRANSLATION (50, 30)")
            st.image(cv2.warpAffine(img_gray, M_t, (w,h)),
                     use_container_width=True, clamp=True, caption="tx=50, ty=30")
        with c3:
            sec("HORIZONTAL SHEAR")
            st.image(cv2.warpAffine(img_gray, M_s, (nw,h)),
                     use_container_width=True, clamp=True, caption="shear=0.3")

    with tab3:
        M_fwd = cv2.getRotationMatrix2D((w//2,h//2),  45, 1.0)
        M_inv = cv2.getRotationMatrix2D((w//2,h//2), -45, 1.0)
        rot45    = cv2.warpAffine(img_gray, M_fwd, (w,h))
        restored = cv2.warpAffine(rot45,    M_inv, (w,h))
        c1,c2,c3 = st.columns(3)
        with c1: st.image(img_gray,  use_container_width=True, clamp=True, caption="Original")
        with c2: st.image(rot45,     use_container_width=True, clamp=True, caption="Rotated 45°")
        with c3: st.image(restored,  use_container_width=True, clamp=True, caption="Restored (−45°)")


# ══════════════════════════════════════════════════════════════════
# PHASE 6.4
# ══════════════════════════════════════════════════════════════════
elif active == "p64":
    phase_header("4", "Intensity Transformations", "Apply negative, log, and gamma correction")

    d        = img_gray.astype(np.float64) / 255.0
    negative = 1.0 - d
    log_img  = np.log1p(d); log_img /= log_img.max()
    g05      = np.power(d, 0.5)
    g15      = np.power(d, 1.5)

    imgs = [d, negative, log_img, g05, g15]
    caps = ["Original", "Negative\ns = L−1−r", "Log Transform\ns = c·log(1+r)", "Gamma γ=0.5\n(Brighter)", "Gamma γ=1.5\n(Darker)"]
    cols = st.columns(5)
    for col, im, cap in zip(cols, imgs, caps):
        with col:
            st.image((im*255).astype(np.uint8), use_container_width=True, clamp=True, caption=cap)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#0a0a1f;border:1px solid #1a1a3e;border-left:3px solid #ffbe0b;padding:1rem 1.5rem;border-radius:6px;">
            <div style="font-family:JetBrains Mono;font-size:0.68rem;color:#ffbe0b;letter-spacing:1px;margin-bottom:0.4rem;">BEST FOR BRIGHTENING</div>
            <div style="font-size:1rem;color:#e0e6ff;font-weight:600;">Gamma Correction  γ = 0.5</div>
            <div style="font-size:0.82rem;color:#7777aa;margin-top:0.3rem;">Power-law curve lifts dark pixels, expanding shadow detail.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#0a0a1f;border:1px solid #1a1a3e;border-left:3px solid #00f5d4;padding:1rem 1.5rem;border-radius:6px;">
            <div style="font-family:JetBrains Mono;font-size:0.68rem;color:#00f5d4;letter-spacing:1px;margin-bottom:0.4rem;">BEST FOR HIGHLIGHTING DETAILS</div>
            <div style="font-size:1rem;color:#e0e6ff;font-weight:600;">Log Transformation</div>
            <div style="font-size:0.82rem;color:#7777aa;margin-top:0.3rem;">Compresses highlights, expands shadows — ideal for high dynamic range images.</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 6.5
# ══════════════════════════════════════════════════════════════════
elif active == "p65":
    phase_header("5", "Histogram Processing", "Analyze and enhance contrast using histograms")

    def manual_he(gray):
        hist, _ = np.histogram(gray.flatten(), 256, [0,256])
        cdf     = np.cumsum(hist / hist.sum())
        return np.uint8(255 * cdf)[gray]

    eq_m = manual_he(img_gray)
    eq_b = cv2.equalizeHist(img_gray)

    tab1, tab2 = st.tabs(["  🖼 IMAGES  ", "  📊 HISTOGRAMS  "])

    with tab1:
        cols = st.columns(3)
        for col, im, cap in zip(cols,
                                 [img_gray, eq_m, eq_b],
                                 ["Original", "Manual HE", "OpenCV histeq()"]):
            with col:
                sec(cap.upper())
                st.image(im, use_container_width=True, clamp=True, caption=cap)

    with tab2:
        fig, axes = plt.subplots(1, 3, figsize=(15,4), facecolor='#0a0a1f')
        palette   = ["#7b2fff","#00f5d4","#ffbe0b"]
        titles    = ["Original","Manual HE","OpenCV HE"]
        for ax, im, title, color in zip(axes,[img_gray,eq_m,eq_b],titles,palette):
            ax.set_facecolor('#0a0a1f')
            ax.hist(im.flatten(), 256, [0,256], color=color, alpha=0.85)
            ax.set_title(title, color='#e0e6ff', fontsize=10)
            ax.set_xlabel('Intensity', color='#4a4a8a', fontsize=8)
            ax.set_ylabel('Pixel Count', color='#4a4a8a', fontsize=8)
            ax.tick_params(colors='#4a4a8a', labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#1a1a3e')
        plt.tight_layout()
        st.image(fig_to_img(fig), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 6.6 — Strong Visible Enhancement
# ══════════════════════════════════════════════════════════════════
elif active == "p66":
    phase_header("6", "Image Enhancement Pipeline", "Gamma correction, CLAHE, sharpening, and denoising")

    def process_image(gray):
        # Step 1: Gamma — brighten dark regions
        d     = gray.astype(np.float64) / 255.0
        gamma = np.uint8(np.power(d, 0.55) * 255)

        # Step 2: CLAHE — adaptive local contrast (much stronger than basic HE)
        clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrasted = clahe.apply(gamma)

        # Step 3: Unsharp mask — sharpen edges and fine detail
        blurred = cv2.GaussianBlur(contrasted, (0,0), 3)
        sharp   = cv2.addWeighted(contrasted, 1.6, blurred, -0.6, 0)

        # Step 4: Denoise — clean up noise
        denoised = cv2.fastNlMeansDenoising(sharp, h=5,
                                            templateWindowSize=7,
                                            searchWindowSize=15)
        return denoised

    enhanced = process_image(img_gray)

    c1, c2 = st.columns(2)
    with c1:
        sec("INPUT IMAGE")
        st.image(img_gray,  use_container_width=True, clamp=True, caption="Original Input")
    with c2:
        sec("ENHANCED OUTPUT")
        st.image(enhanced, use_container_width=True, clamp=True, caption="Smart Enhanced Output")

    st.markdown("<br>", unsafe_allow_html=True)
    sec("HISTOGRAM COMPARISON")
    fig, axes = plt.subplots(1, 2, figsize=(12,3), facecolor='#0a0a1f')
    for ax, im, title, color in zip(axes,
                                     [img_gray, enhanced],
                                     ["Original","Enhanced"],
                                     ["#7b2fff","#00f5d4"]):
        ax.set_facecolor('#0a0a1f')
        ax.hist(im.flatten(), 256, [0,256], color=color, alpha=0.85)
        ax.set_title(title, color='#e0e6ff', fontsize=10)
        ax.set_xlabel('Intensity', color='#4a4a8a', fontsize=8)
        ax.tick_params(colors='#4a4a8a', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#1a1a3e')
    plt.tight_layout()
    st.image(fig_to_img(fig), use_container_width=True)

    st.markdown("""
    <div style="background:#0a0a1f;border:1px solid #1a1a3e;border-radius:8px;padding:1.2rem 1.5rem;margin-top:1rem;">
        <div style="font-family:JetBrains Mono;font-size:0.65rem;color:#00f5d4;letter-spacing:2px;margin-bottom:0.8rem;">PIPELINE STEPS</div>
        <div style="display:flex;gap:0.5rem;align-items:stretch;flex-wrap:wrap;">
            <div style="background:#0e0e2a;border-top:2px solid #00f5d4;padding:0.8rem 1rem;border-radius:6px;flex:1;min-width:130px;">
                <div style="font-family:JetBrains Mono;font-size:0.62rem;color:#00f5d4;">STEP 1</div>
                <div style="font-size:0.9rem;font-weight:600;color:#e0e6ff;margin-top:0.2rem;">Gamma  γ=0.55</div>
                <div style="font-size:0.75rem;color:#5a5a9a;margin-top:0.2rem;">Brighten dark regions</div>
            </div>
            <div style="color:#00f5d4;align-self:center;">→</div>
            <div style="background:#0e0e2a;border-top:2px solid #7b2fff;padding:0.8rem 1rem;border-radius:6px;flex:1;min-width:130px;">
                <div style="font-family:JetBrains Mono;font-size:0.62rem;color:#7b2fff;">STEP 2</div>
                <div style="font-size:0.9rem;font-weight:600;color:#e0e6ff;margin-top:0.2rem;">CLAHE</div>
                <div style="font-size:0.75rem;color:#5a5a9a;margin-top:0.2rem;">Local contrast boost</div>
            </div>
            <div style="color:#00f5d4;align-self:center;">→</div>
            <div style="background:#0e0e2a;border-top:2px solid #ff6b9d;padding:0.8rem 1rem;border-radius:6px;flex:1;min-width:130px;">
                <div style="font-family:JetBrains Mono;font-size:0.62rem;color:#ff6b9d;">STEP 3</div>
                <div style="font-size:0.9rem;font-weight:600;color:#e0e6ff;margin-top:0.2rem;">Unsharp Mask</div>
                <div style="font-size:0.75rem;color:#5a5a9a;margin-top:0.2rem;">Sharpen edges</div>
            </div>
            <div style="color:#00f5d4;align-self:center;">→</div>
            <div style="background:#0e0e2a;border-top:2px solid #ffbe0b;padding:0.8rem 1rem;border-radius:6px;flex:1;min-width:130px;">
                <div style="font-family:JetBrains Mono;font-size:0.62rem;color:#ffbe0b;">STEP 4</div>
                <div style="font-size:0.9rem;font-weight:600;color:#e0e6ff;margin-top:0.2rem;">Denoise</div>
                <div style="font-size:0.75rem;color:#5a5a9a;margin-top:0.2rem;">Clean output</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    buf = io.BytesIO()
    Image.fromarray(enhanced).save(buf, format='PNG')
    st.download_button(
        label="⬇  DOWNLOAD ENHANCED IMAGE",
        data=buf.getvalue(),
        file_name="enhanced_output.png",
        mime="image/png"
    )