"""
app.py - the interface.
    Run with:   streamlit run app.py

    Upload a video -> extract_stickers() from stickers.py -> show the stickers
    in time order on a "cutting mat" -> download them all as transparent PNGs.

Styling: colours live in .streamlit/config.toml, everything else in CSS below.
"""

import base64
import io
import random
import tempfile
import zipfile

import cv2
import streamlit as st

from pathlib import Path
from stickers import DUPLICATE_OVERLAP, extract_stickers, make_gif

st.set_page_config(page_title="Motion Stickers!", page_icon="✂️", layout="wide")

# Load the page styling from styles/style.css
css = (Path(__file__).parent / "styles" / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sticker_figure(sticker, t, i, kept=True):
    """Turn one BGRA sticker into an HTML <figure> for the sticker sheet.
    Unticked stickers get the "unticked" class so the CSS can fade them out."""
    h, w = sticker.shape[:2]
    if w > 400:                                   # shrink for display only
        sticker = cv2.resize(sticker, (400, int(h * 400 / w)))
    ok, png = cv2.imencode(".png", sticker)
    b64 = base64.b64encode(png.tobytes()).decode()
    tilt = [-3, 2, -1.5, 3, -2, 1][i % 6]         # varied tilt, like hand-placed stickers
    faded = "" if kept else " unticked"           # NEW: faded look for unticked stickers
    return (f'<figure class="sticker{faded}" style="--tilt:{tilt}deg">'
            f'<img src="data:image/png;base64,{b64}" alt="Sticker at {t:.1f} seconds">'
            f'<figcaption>{t:.1f} s</figcaption></figure>')


# NEW: confetti colours, matching the yellow buttons plus a few bright extras
CONFETTI_COLOURS = ["#f5c542", "#ff6b6b", "#4ecdc4", "#ffffff", "#ff9ff3", "#54a0ff"]


def confetti(pieces=120, slowest=8.0, fastest=5.0):
    """NEW: confetti that drifts down the screen once when the stickers are ready.
    Plain HTML and CSS (no JavaScript), so it works without internet too.
    slowest/fastest are how many seconds a piece takes to fall: raise them
    to slow it down."""
    bits = []
    for _ in range(pieces):
        size = random.uniform(6, 12)                        # px
        bits.append(
            f'<span style="left:{random.uniform(0, 100):.1f}vw;'
            f'width:{size:.0f}px;height:{size * 0.45:.0f}px;'
            f'background:{random.choice(CONFETTI_COLOURS)};'
            f'animation-duration:{random.uniform(fastest, slowest):.1f}s;'
            f'animation-delay:{random.uniform(0, 1.5):.1f}s;'      # not all at once
            f'--drift:{random.uniform(-12, 12):.0f}vw;'            # sideways sway
            f'--spin:{random.choice([-1, 1]) * random.randint(360, 1080)}deg"></span>')
    st.markdown(
        '<style>'
        '.confetti{position:fixed;inset:0;pointer-events:none;z-index:999999;overflow:hidden}'
        '.confetti span{position:absolute;top:-5vh;border-radius:2px;opacity:0;'
        'animation-name:confetti-fall;animation-timing-function:linear;'
        'animation-fill-mode:forwards}'
        '@keyframes confetti-fall{'
        '0%{opacity:1;transform:translate(0,0) rotate(0deg)}'
        '85%{opacity:1}'
        '100%{opacity:0;transform:translate(var(--drift),110vh) rotate(var(--spin))}}'
        '</style>'
        f'<div class="confetti">{"".join(bits)}</div>', unsafe_allow_html=True)


def preview_placeholder(message):
    """NEW: a dashed "empty sticker" for the preview corner when there's no GIF
    yet, so the top right of the page isn't just a blank gap."""
    return ('<figure class="sticker" style="--tilt:2deg">'
            '<div style="border:2px dashed #f5c542; border-radius:16px; '
            'width:100%; max-width:260px; height:150px; margin:auto; padding:12px; '
            'display:flex; align-items:center; justify-content:center; '
            f'text-align:center; opacity:0.85">{message}</div>'
            '<figcaption>Stop-motion preview</figcaption></figure>')


# ------------------------------------------------------------------ layout --
# NEW: title and steps on the left, stop-motion preview on the right.
# The GIF isn't made until further down the script, so keep an empty slot
# here and fill it in once the GIF exists
col_title, col_preview = st.columns([3, 1])
with col_title:
    st.title("Motion Stickers!")
    st.markdown(
        '<div class="steps">'
        '<span class="step"><b>1</b>Upload your footage</span>'
        '<span class="step"><b>2</b>We find what moves</span>'
        '<span class="step"><b>3</b>Peel off the stickers</span>'
        '</div>', unsafe_allow_html=True)
preview_slot = col_preview.empty()
# NEW: show the placeholder until there's a GIF to put here
preview_slot.markdown(preview_placeholder("Your stop-motion will appear here ✂️"),
                      unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    # Motion detection now looks at every frame (to keep the background
    # model up to date), so this slider only controls how often a sticker
    # is made, not how accurate the detection is
    every_n = st.slider("Check every Nth frame", 1, 60, 15,
                        help="How often to make a sticker. Lower means more stickers. "
                             "Movement is still tracked on every frame.")
    # De-duplication: skip a sticker that overlaps the last kept one by more
    # than this, so an object standing still doesn't fill the sheet
    overlap = st.slider("Skip near-duplicates", 0.5, 1.0, DUPLICATE_OVERLAP, 0.05,
                        help="Skips a sticker if it overlaps the last one by more than this. "
                             "Lower skips more. 1.0 keeps every sticker.")
    # Footage settings: Auto works for most clips, but the user can override
    # it if they know their footage (e.g. a bright infrared camera)
    AUTO_ON_OFF = {"Auto": None, "On": True, "Off": False}
    night_choice = st.selectbox("Night mode", list(AUTO_ON_OFF),
                                help="Brightens and boosts contrast for dark footage. "
                                     "Auto turns it on for dark clips.")
    shadow_choice = st.selectbox("Shadow removal", list(AUTO_ON_OFF),
                                 help="Removes shadows from the stickers. Auto turns it "
                                      "on for colour footage and off for black-and-white "
                                      "or infrared footage, where it doesn't work.")
    st.caption("Works best with a fixed camera, like indoor CCTV or a doorbell cam.")

# NEW: the upload box only takes up the left half of the page
col_upload, _ = st.columns(2)
with col_upload:
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded and st.button("Make stickers"):
    # OpenCV needs a real file path, so save the upload to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    bar = st.progress(0.0, text="Looking for movement")
    try:
        stickers = extract_stickers(video_path, every_n, bar.progress, overlap,
                                    night=AUTO_ON_OFF[night_choice],
                                    shadows=AUTO_ON_OFF[shadow_choice])
    except ValueError as err:
        st.session_state.stickers = []   # NEW: clear old stickers so they don't show under the error
        st.error(f"{err} Try an MP4 file.")
        st.stop()
    bar.empty()
    # NEW: save the stickers so they're still there when a checkbox is ticked.
    # Streamlit re-runs the whole script on every click, and the button above
    # is only True on the run straight after it's pressed
    st.session_state.stickers = stickers
    # NEW: a new number for each video, so the checkboxes start ticked again
    st.session_state.run_id = st.session_state.get("run_id", 0) + 1

    if not stickers:
        st.info("No movement found. Try a lower 'every Nth frame' value in Settings.")
        st.stop()

    # NEW: a little celebration when the stickers are ready. Only on the run
    # straight after "Make stickers", not every time a checkbox is clicked
    confetti()

# NEW: show the saved stickers on every re-run, not just straight after the button
stickers = st.session_state.get("stickers")
if stickers:
    st.subheader(f"{len(stickers)} stickers, in the order they appeared")

    # NEW: human-in-the-loop review. The pipeline can't catch every mistake
    # (stray blobs, bits of background), so the user unticks bad stickers and
    # only the ticked ones go into the zip and the GIF
    st.caption("Untick any sticker you don't want, like a stray blob or a bit of "
               "background. Only ticked stickers go into the downloads.")

    # NEW: one checkbox key per sticker. The run_id makes the keys different
    # for each video, so a new video starts with every box ticked
    keys = [f"keep_{st.session_state.run_id}_{i}" for i in range(len(stickers))]
    for key in keys:
        st.session_state.setdefault(key, True)        # ticked by default

    # NEW: the Tick all / Untick all buttons call this before the page re-runs
    def set_all(keys, value):
        for key in keys:
            st.session_state[key] = value

    # NEW: the ticked stickers, worked out before drawing the page so the
    # counter can go at the top
    keep = [(t, sticker) for key, (t, sticker) in zip(keys, stickers)
            if st.session_state[key]]

    # NEW: Tick all / Untick all buttons and a "Keeping X of Y" counter
    col_all, col_none, col_count = st.columns([1, 1, 4])
    col_all.button("Tick all", on_click=set_all, args=(keys, True))
    col_none.button("Untick all", on_click=set_all, args=(keys, False))
    col_count.markdown(f"Keeping **{len(keep)} of {len(stickers)}** stickers")

    # NEW: the sticker grid, 4 per row, each sticker with its checkbox underneath
    for row_start in range(0, len(stickers), 4):
        cols = st.columns(4)
        for col, i in zip(cols, range(row_start, min(row_start + 4, len(stickers)))):
            t, sticker = stickers[i]
            with col:
                # CHANGED: pass in whether it's ticked, so unticked ones fade out
                st.markdown(sticker_figure(sticker, t, i, st.session_state[keys[i]]),
                            unsafe_allow_html=True)
                st.checkbox(f"Keep ({t:.1f} s)", key=keys[i])

    # NEW: nothing to download if every box is unticked
    if not keep:
        # NEW: say why the preview is empty
        preview_slot.markdown(preview_placeholder("Tick a sticker to see the stop-motion"),
                              unsafe_allow_html=True)
        st.warning("Tick at least one sticker to download.")
        st.stop()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i, (t, sticker) in enumerate(keep):           # CHANGED: only the ticked stickers
            ok, png = cv2.imencode(".png", sticker)          # PNG keeps transparency
            zf.writestr(f"sticker_{i:03d}_{t:.1f}s.png", png.tobytes())

    col_zip, col_gif = st.columns(2)
    with col_zip:
        st.download_button("Download all stickers (.zip)", zip_buffer.getvalue(),
                           file_name="stickers.zip", mime="application/zip")
    with col_gif:
        gif = make_gif(keep)                               # CHANGED: only the ticked stickers
        if gif:
            st.download_button("Download stop-motion (.gif)", gif,
                               file_name="stickers.gif", mime="image/gif")
            # NEW: preview the GIF so you can check it without downloading.
            # Shown as HTML, because st.image() resizes the GIF and only keeps
            # its first frame. Styled like the stickers, but without the tilt.
            # CHANGED: goes in the slot beside the title, not under the button
            b64_gif = base64.b64encode(gif).decode()
            preview_slot.markdown(f'<figure class="sticker" style="--tilt:0deg">'
                                  f'<img src="data:image/gif;base64,{b64_gif}" '
                                  f'alt="Stop-motion preview" '
                                  f'style="width:100%; max-width:260px">'
                                  f'<figcaption>Stop-motion preview</figcaption></figure>',
                                  unsafe_allow_html=True)