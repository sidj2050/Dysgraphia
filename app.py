import html
import os
import sys

import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from predict import NoHandwritingError, active_model_info, analyze_image  # noqa: E402

# st.set_page_config(page_title="Dysgraphia Screening", page_icon="✍️", layout="centered",
#                    initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stAppDeployButton"], [data-testid="stHeader"],
[data-testid="stFileUploaderDeleteBtn"] ~ div small {display:none !important;}
.block-container {max-width: 720px; padding-top: 4.5rem; padding-bottom: 3rem;}
.hero {text-align:center; margin-bottom: 1.6rem;}
.hero h1 {font-size: 2.1rem; font-weight: 700; letter-spacing:-0.02em; margin:0 0 .5rem 0; padding:0;}
.hero p {color:#6b7280; font-size:1.02rem; margin:0;}
[data-testid="stFileUploader"] section {border:1.5px dashed #c7cbe0; border-radius:16px; background:#fafbff; padding:1.6rem;}
[data-testid="stFileUploader"] section:hover {border-color:#4f46e5; background:#f4f4ff;}
[data-testid="stFileUploaderDropzoneInstructions"] {justify-content:center; text-align:center;}
[data-testid="stImage"] {display:flex; justify-content:center;}
[data-testid="stImage"] img {max-height:300px; width:auto !important; border-radius:12px; border:1px solid #e5e7eb;}
.verdict {text-align:center; margin:1.6rem 0 .4rem 0;}
.verdict .tag {display:inline-block; padding:.2rem 1.1rem; border-radius:999px; font-weight:700; font-size:1.05rem; letter-spacing:.06em;}
.verdict .yes {background:#fee2e2; color:#b91c1c;} .verdict .no {background:#dcfce7; color:#15803d;}
.verdict h2 {font-size:1.35rem; font-weight:650; margin:.8rem 0 .2rem 0; padding:0;}
.pct {font-size:3.4rem; font-weight:750; line-height:1.1; margin-top:.4rem; letter-spacing:-0.03em;}
.pct-l {color:#6b7280; font-size:.9rem;}
.why {text-align:center; color:#374151; margin:1rem auto 1.6rem auto; max-width:560px; line-height:1.55;}
.note {text-align:center; color:#9a6700; font-size:.9rem; margin:-.6rem 0 1.2rem 0;}
table.t {width:100%; border-collapse:collapse; font-size:.9rem;}
table.t th {text-align:left; color:#6b7280; font-weight:600; padding:.55rem .6rem; border-bottom:1px solid #e5e7eb;}
table.t td {padding:.5rem .6rem; border-bottom:1px solid #f0f1f5; vertical-align:top;}
table.t td.f {white-space:nowrap; font-weight:600;} .u {color:#b45309;} .ty {color:#15803d;}
table.t td.r {color:#6b7280;}
.pill {display:inline-flex; align-items:center; gap:.45rem; margin-top:.9rem; padding:.25rem .8rem; border-radius:999px; background:#eef0ff; color:#3730a3; font-size:.82rem; font-weight:600;}
.pill i {width:.45rem; height:.45rem; border-radius:50%; background:#4f46e5; display:inline-block;}
.foot {text-align:center; color:#9ca3af; font-size:.8rem; margin-top:2rem;}
</style>
<div class="hero"><h1>Dysgraphia Screening</h1>
<p>Upload a photo</p>
<span class="pill"><i></i>Deep learning · classifying with the {MODEL} network</span></div>
"""
.replace("{MODEL}", active_model_info()["name"]), unsafe_allow_html=True)

uploaded = st.file_uploader("Upload handwriting image", label_visibility="collapsed",
                            type=["jpg", "jpeg", "png", "bmp", "webp"])


@st.cache_data(show_spinner=False, max_entries=32)
def run(data: bytes):
    import io
    return analyze_image(Image.open(io.BytesIO(data)))


if uploaded:
    data = uploaded.getvalue()
    try:
        preview = ImageOps.exif_transpose(Image.open(__import__("io").BytesIO(data))).convert("RGB")
    except Exception:
        st.error("This file could not be read as an image.")
        st.stop()
    import base64, io as _io
    thumb = preview.copy(); thumb.thumbnail((900, 600))
    buf = _io.BytesIO(); thumb.save(buf, format="JPEG", quality=88)
    st.markdown('<div style="text-align:center;margin-top:1rem"><img style="max-height:300px;max-width:100%;'
                'border-radius:12px;border:1px solid #e5e7eb" src="data:image/jpeg;base64,'
                + base64.b64encode(buf.getvalue()).decode() + '"></div>', unsafe_allow_html=True)
    try:
        with st.spinner("Analysing handwriting…"):
            r = run(data)
    except NoHandwritingError:
        st.markdown('<p class="why">We could not find enough handwriting in this image. '
                    'Please upload a clear, well-lit photo or scan of handwritten text.</p>',
                    unsafe_allow_html=True)
        st.stop()
    except Exception:
        st.markdown('<p class="why">Something went wrong analysing this image. '
                    'Please try another photo.</p>', unsafe_allow_html=True)
        st.stop()

    yes = r["final"] == "YES"
    st.markdown(f"""
<div class="verdict"><span class="tag {'yes' if yes else 'no'}">{r['final']}</span>
<h2>Dysgraphia-associated pattern {'detected' if yes else 'not detected'}</h2>
<div class="pct">{r['probability']:.1%}</div><div class="pct-l">Likelihood of dysgraphia-associated pattern</div></div>
<p class="why">{html.escape(r['explanation'])} {r['unusual_count']} of {r['total']} measured
characteristics fall outside the typical range.</p>
""", unsafe_allow_html=True)
    if r["borderline"]:
        st.markdown('<p class="note">Borderline result — consider testing with another writing sample.</p>',
                    unsafe_allow_html=True)


    rows = "".join(
        f"<tr><td class='f'>{html.escape(x['feature'])}</td>"
        f"<td class='f {'u' if x['status']=='Unusual' else 'ty'}'>{x['status']}</td>"
        f"<td class='r'>{html.escape(x['reason'])}</td></tr>" for x in r["findings"])
    st.markdown(f"<table class='t'><tr><th>Characteristic</th><th>Finding</th>"
                f"<th>Observation</th></tr>{rows}</table>", unsafe_allow_html=True)
    # st.markdown('<div class="foot">Screening support only — not a medical diagnosis.</div>',
                # unsafe_allow_html=True)
