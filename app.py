"""
app.py - the interface.
    Run with:   streamlit run app.py

    Upload a video -> extract_stickers() from stickers.py -> show the stickers
    in time order on a "cutting mat" -> download them all as transparent PNGs.

Styling: colours live in .streamlit/config.toml, everything else in CSS below.
"""

import base64
import io
import tempfile
import zipfile

import cv2
import streamlit as st

from pathlib import Path
from stickers import extract_stickers, make_gif

st.set_page_config(page_title="Motion Stickers!", page_icon="✂️", layout="wide")

# Load the page styling from styles/style.css
css = (Path(__file__).parent / "styles" / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sticker_figure(sticker, t, i):
    """Turn one BGRA sticker into an HTML <figure> for the sticker sheet."""
    h, w = sticker.shape[:2]
    if w > 400:                                   # shrink for display only
        sticker = cv2.resize(sticker, (400, int(h * 400 / w)))
    ok, png = cv2.imencode(".png", sticker)
    b64 = base64.b64encode(png.tobytes()).decode()
    tilt = [-3, 2, -1.5, 3, -2, 1][i % 6]         # varied tilt, like hand-placed stickers
    return (f'<figure class="sticker" style="--tilt:{tilt}deg">'
            f'<img src="data:image/png;base64,{b64}" alt="Sticker at {t:.1f} seconds">'
            f'<figcaption>{t:.1f} s</figcaption></figure>')


# ------------------------------------------------------------------ layout --
st.title("Motion Stickers!")
st.markdown(
    '<div class="steps">'
    '<span class="step"><b>1</b>Upload your footage</span>'
    '<span class="step"><b>2</b>We find what moves</span>'
    '<span class="step"><b>3</b>Peel off the stickers</span>'
    '</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    every_n = st.slider("Check every Nth frame", 1, 60, 15,
                        help="Lower means more stickers and slower processing.")
    st.caption("Works best with a fixed camera, like indoor CCTV or a doorbell cam.")

uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded and st.button("Make stickers"):
    # OpenCV needs a real file path, so save the upload to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    bar = st.progress(0.0, text="Looking for movement")
    try:
        stickers = extract_stickers(video_path, every_n, bar.progress)
    except ValueError as err:
        st.error(f"{err} Try an MP4 file.")
        st.stop()
    bar.empty()

    if not stickers:
        st.info("No movement found. Try a lower 'every Nth frame' value in Settings.")
        st.stop()

    st.subheader(f"{len(stickers)} stickers, in the order they appeared")

    figures = []
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i, (t, sticker) in enumerate(stickers):
            figures.append(sticker_figure(sticker, t, i))
            ok, png = cv2.imencode(".png", sticker)          # PNG keeps transparency
            zf.writestr(f"sticker_{i:03d}_{t:.1f}s.png", png.tobytes())

    st.markdown('<div class="sheet">' + "".join(figures) + "</div>", unsafe_allow_html=True)
    col_zip, col_gif = st.columns(2)
    with col_zip:
        st.download_button("Download all stickers (.zip)", zip_buffer.getvalue(),
                           file_name="stickers.zip", mime="application/zip")
    with col_gif:
        gif = make_gif(stickers)
        if gif:
            st.download_button("Download stop-motion (.gif)", gif,
                               file_name="stickers.gif", mime="image/gif")