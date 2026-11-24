"""
app.py - the interface.
    Run with:   streamlit run app.py

    Upload a video -> extract_stickers() from stickers.py -> show the stickers
    in time order on a "cutting mat" -> download them all as transparent PNGs.

Styling: colours live in .streamlit/config.toml, everything else in CSS below.
NEW: styles/responsive.css makes the page fit smaller windows and phones.
"""

import base64
import io
import random
import tempfile
import time
import zipfile

import cv2
import numpy as np
import streamlit as st

from pathlib import Path
from stickers import DUPLICATE_OVERLAP, extract_stickers, make_gif

st.set_page_config(page_title="Motion Stickers!", page_icon="✂️", layout="wide")

# Load the page styling from styles/style.css
# NEW: then styles/responsive.css, which is loaded second so its rules for
# smaller screens win over the normal ones
styles = Path(__file__).parent / "styles"
css = (styles / "style.css").read_text(encoding="utf-8")
css += "\n" + (styles / "responsive.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sticker_figure(sticker, t, i, kept=True):
    """Turn one BGRA sticker into an HTML <figure> for the sticker sheet.
    Unticked stickers get the "unticked" class so the CSS can fade them out.
    NEW: clicking the sticker opens it full size; clicking again closes it."""
    # NEW: full-size copy for the enlarged view, made before shrinking
    ok, big_png = cv2.imencode(".png", sticker)
    big_b64 = base64.b64encode(big_png.tobytes()).decode()
    big_h, big_w = sticker.shape[:2]                # full size, for the caption

    h, w = sticker.shape[:2]
    if w > 400:                                   # shrink for display only
        sticker = cv2.resize(sticker, (400, int(h * 400 / w)))
    ok, png = cv2.imencode(".png", sticker)
    b64 = base64.b64encode(png.tobytes()).decode()
    tilt = [-3, 2, -1.5, 3, -2, 1][i % 6]         # varied tilt, like hand-placed stickers
    faded = "" if kept else " unticked"           # NEW: faded look for unticked stickers

    # NEW: click to enlarge. <details> opens and closes on click with no
    # JavaScript. When it's open, the CSS turns .zoom-big into a full-screen
    # dark overlay, and clicking the overlay closes it again
    return (f'<details class="zoom"><summary>'
            f'<figure class="sticker{faded}" style="--tilt:{tilt}deg">'
            f'<img src="data:image/png;base64,{b64}" alt="Sticker at {t:.1f} seconds">'
            f'<figcaption>{t:.1f} s</figcaption></figure>'
            f'<span class="zoom-big">'
            f'<img src="data:image/png;base64,{big_b64}" alt="Sticker at {t:.1f} seconds, full size">'
            f'<span class="zoom-caption">{t:.1f} s · {big_w} × {big_h} px original · click anywhere to close</span>'
            f'</span></summary></details>')


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


def preview_placeholder(message, label="Stop-motion preview", tilt=2, tape="left"):
    """NEW: a dashed "empty sticker" for a preview spot that has nothing in it
    yet (no GIF, or no video uploaded), so the page isn't just a blank gap.
    tape is which top corner gets a strip of washi tape ("left" or "right")."""
    return (f'<figure class="sticker" style="--tilt:{tilt}deg">'
            '<div style="position:relative; border:2px dashed #f5c542; border-radius:16px; '
            'width:100%; max-width:320px; height:190px; margin:auto; padding:12px; '
            'display:flex; align-items:center; justify-content:center; '
            'text-align:center">'
            f'<span class="tape {tape}"></span>'            # NEW: washi tape
            f'<span style="opacity:0.85">{message}</span></div>'
            f'<figcaption>{label}</figcaption></figure>')


# NEW: fun messages shown on the progress bar while the stickers are made.
# They change every couple of seconds, so a long video doesn't look stuck
LOADING_MESSAGES = [
    "Sharpening the scissors ✂️",
    "Hunting for movement 🔍",
    "Learning what the empty room looks like 🏠",
    "Spotting things that move 👀",
    "Tidying up the edges 🧹",
    "Shooing away shadows 🌑",
    "Peeling off the stickers ✨",
    "Arranging them on the mat 🗂️",
]
MESSAGE_SECONDS = 2.0                     # how long each message stays up


def progress_with_messages(bar):
    """NEW: wrap the progress bar so it shows a fun message as well as the
    progress. extract_stickers() calls this with the fraction done (0 to 1),
    exactly like it called bar.progress before."""
    start = time.monotonic()

    def update(fraction, *args, **kwargs):
        step = int((time.monotonic() - start) / MESSAGE_SECONDS)
        message = LOADING_MESSAGES[step % len(LOADING_MESSAGES)]
        bar.progress(min(max(float(fraction), 0.0), 1.0), text=message)

    return update


def video_info(video_path):
    """NEW: how many frames the video has and how long it is in seconds,
    for the stats panel. Some videos don't report a frame rate, so guard
    against dividing by zero."""
    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    seconds = frames / fps if fps > 0 else 0.0
    return frames, seconds


def stats_panel(stats):
    """NEW: a row of yellow "stat stickers" showing what happened on this run.
    The styling is in styles/style.css (.stats and .stat)."""
    cards = [
        (stats["stickers"], "stickers found"),
        (f'{stats["seconds"]:.1f} s', "of video"),
        (f'{stats["frames"]:,}', "frames checked"),
        (f'{stats["took"]:.1f} s', "to make them"),
    ]
    tilts = [-2, 1.5, -1, 2]                # hand-placed look, like the stickers
    html = "".join(f'<div class="stat" style="--tilt:{tilt}deg">'
                   f'<b>{value}</b><span>{label}</span></div>'
                   for (value, label), tilt in zip(cards, tilts))
    return f'<div class="stats">{html}</div>'


# NEW: the three steps at the top of the page. They're "live": the step
# you're on is highlighted, finished steps get a tick, and later steps are
# faded. The styling is in styles/style.css (.step.current/.done/.todo)
STEPS = ["Upload your footage", "We find what moves", "Peel off the stickers"]


def steps_html(current, working=False):
    """NEW: HTML for the steps row. current is the step you're on (1, 2 or 3).
    working=True makes the current step pulse while the video is processed."""
    spans = []
    for number, text in enumerate(STEPS, start=1):
        if number < current:
            state, badge = "done", "✓"           # finished: tick instead of number
        elif number == current:
            state, badge = "current", number
            if working:
                state += " working"              # pulses while processing
        else:
            state, badge = "todo", number
        spans.append(f'<span class="step {state}"><b>{badge}</b>{text}</span>')
    return f'<div class="steps">{"".join(spans)}</div>'


# NEW: example stickers shown before any real ones are made, so the bottom
# of the page isn't empty and new users can see what they'll get.
# (emoji, time) pairs, made to look like things a home camera might catch
EXAMPLE_STICKERS = [("🚶", 0.5), ("🐕", 2.0), ("📦", 3.5), ("🐈", 5.0), ("🚲", 7.5)]


def example_stickers_html(message="Upload a clip and your stickers land here ✂️"):
    """NEW: a row of faded example stickers with a line explaining them.
    The styling is in styles/style.css (.examples and .example-sticker)."""
    tilts = [-4, 3, -2, 4, -3]
    stickers = "".join(f'<div class="example-sticker" style="--tilt:{tilt}deg">'
                       f'<span class="emoji">{emoji}</span><em>{t:.1f} s</em></div>'
                       for (emoji, t), tilt in zip(EXAMPLE_STICKERS, tilts))
    return ('<div class="examples">'
            f'<p>{message}</p>'
            f'<div class="example-row">{stickers}</div></div>')


# NEW: timeline strip. A bar the length of the video with a dot where each
# sticker was cut. Hover a dot to see its time; unticked stickers are faded.
# The styling is in styles/style.css (.timeline, .track, .dot)
def timeline_html(stickers, kept_flags, video_seconds):
    """Build the timeline bar. video_seconds can be 0 if the video didn't
    report its length, so fall back to just past the last sticker."""
    last = max(t for t, _ in stickers)
    length = video_seconds if video_seconds > 0 else max(last * 1.05, 1.0)
    dots = []
    for i, ((t, _), kept) in enumerate(zip(stickers, kept_flags)):
        left = min(t / length, 1.0) * 100              # % along the bar
        faded = "" if kept else " unticked"
        dots.append(f'<span class="dot{faded}" style="left:{left:.2f}%" '
                    f'title="Sticker {i + 1} · {t:.1f} s"><em>{t:.1f} s</em></span>')
    return ('<div class="timeline">'
            '<span class="tl-end">0 s</span>'
            f'<div class="track">{"".join(dots)}</div>'
            f'<span class="tl-end">{length:.1f} s</span></div>')


# NEW: printable sticker sheet. Lays the kept stickers out on one A4 page with
# OpenCV: alpha-blends each sticker onto a cream "cutting mat" background and
# draws a grey cut line around it, found by growing the sticker's mask
# (dilation) and tracing its outline (contours)
SHEET_W, SHEET_H = 1240, 1754           # A4 at 150 dpi, in pixels
SHEET_MARGIN = 80
SHEET_TOP = 210                         # room for the title
SHEET_BG = (225, 240, 245)              # cream, in BGR (#f5f0e1)
SHEET_GRID = (205, 222, 230)            # slightly darker cream for the mat grid
SHEET_TEXT = (51, 61, 31)               # dark green, in BGR (#1f3d33)
CUT_LINE = (160, 160, 160)              # grey cut line


def paste_sticker(page, sticker, x, y):
    """Alpha-blend a BGRA sticker onto the BGR page with its top-left at (x, y):
    result = alpha * sticker + (1 - alpha) * page."""
    h, w = sticker.shape[:2]
    alpha = sticker[:, :, 3:4].astype(float) / 255.0
    region = page[y:y + h, x:x + w].astype(float)
    blended = alpha * sticker[:, :, :3] + (1 - alpha) * region
    page[y:y + h, x:x + w] = blended.astype("uint8")


def make_sticker_sheet(keep):
    """Return a PNG (bytes) of all the kept stickers on one printable A4 page."""
    page = np.full((SHEET_H, SHEET_W, 3), SHEET_BG, dtype=np.uint8)
    for gx in range(0, SHEET_W, 50):                    # faint cutting-mat grid
        cv2.line(page, (gx, 0), (gx, SHEET_H), SHEET_GRID, 1)
    for gy in range(0, SHEET_H, 50):
        cv2.line(page, (0, gy), (SHEET_W, gy), SHEET_GRID, 1)

    # Title and a short instruction
    cv2.putText(page, "Motion Stickers!", (SHEET_MARGIN, 120),
                cv2.FONT_HERSHEY_DUPLEX, 2.4, SHEET_TEXT, 4, cv2.LINE_AA)
    cv2.putText(page, f"{len(keep)} stickers - cut along the grey lines",
                (SHEET_MARGIN, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                SHEET_TEXT, 2, cv2.LINE_AA)

    # Grid of cells: 3 across for a few stickers, 4 across for more
    cols = 3 if len(keep) <= 9 else 4
    rows = -(-len(keep) // cols)                        # ceiling division
    cell_w = (SHEET_W - 2 * SHEET_MARGIN) // cols
    cell_h = min(cell_w, (SHEET_H - SHEET_TOP - SHEET_MARGIN) // rows)
    pad, label_h = 24, 34                               # space round each sticker

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for i, (t, sticker) in enumerate(keep):
        row, col = divmod(i, cols)
        # Shrink (or grow) the sticker to fit its cell, keeping its shape
        h, w = sticker.shape[:2]
        scale = min((cell_w - 2 * pad) / w, (cell_h - 2 * pad - label_h) / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(sticker, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Centre it in the cell
        x = SHEET_MARGIN + col * cell_w + (cell_w - new_w) // 2
        y = SHEET_TOP + row * cell_h + pad + (cell_h - 2 * pad - label_h - new_h) // 2

        # Cut line: grow the sticker's mask a little so the line sits just
        # outside it, then draw the outline of the grown mask
        mask = (small[:, :, 3] > 0).astype(np.uint8) * 255
        mask = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        grown = cv2.dilate(mask, kernel)
        contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(page, contours, -1, CUT_LINE, 2, cv2.LINE_AA,
                         offset=(x - 10, y - 10))

        paste_sticker(page, small, x, y)

        # Time label under the sticker
        label = f"{t:.1f} s"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        lx = SHEET_MARGIN + col * cell_w + (cell_w - tw) // 2
        ly = SHEET_TOP + (row + 1) * cell_h - pad
        cv2.putText(page, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    SHEET_TEXT, 2, cv2.LINE_AA)

    ok, png = cv2.imencode(".png", page)
    return png.tobytes()


# NEW: footer details. Check the names and module before submitting
TEAM = ["Patrycja", "Chloe", "Sayo"]
MODULE = "Image Processing · 4th Year · 2026"
REPO_URL = "https://github.com/sayoowolabi/motionstickers"


def render_footer():
    """NEW: footer with a dashed "cut here" line, the team as little name-tag
    stickers, the module, and a GitHub button. Styling is in styles/style.css
    (.footer). Called before every st.stop() too, so it's always at the bottom."""
    tilts = [-3, 2, -1.5, 3]                 # hand-placed look, like the steps
    tags = "".join(f'<span class="name-tag" style="--tilt:{tilts[i % 4]}deg">{name}</span>'
                   for i, name in enumerate(TEAM))
    st.markdown('<div class="footer">'
                '<div class="cut-line"><span>✂️</span></div>'
                f'<div class="footer-names"><span class="made">Made by</span>{tags}</div>'
                f'<div class="footer-info">{MODULE}'
                f'<a class="gh" href="{REPO_URL}" target="_blank">View on GitHub ↗</a>'
                '</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ layout --
# NEW: everything you do goes down the left (title, steps, upload, button),
# and the right has two previews side by side: your clip (the input) and the
# stop-motion (the output). vertical_alignment centres the previews against
# the left-hand column, so there's no big gap under the title
# CHANGED: [6, 5] gives the left column a little more room, so all three
# steps fit on one row
# NEW: the columns go in a container with a key. Streamlit gives it the CSS
# class .st-key-top-layout, which responsive.css uses to put the previews
# under the title when the window is narrow
with st.container(key="top-layout"):
    col_left, col_right = st.columns([6, 5], gap="large", vertical_alignment="center")
with col_left:
    st.title("Motion Stickers!")
    # CHANGED: the steps go in a slot, so they can be updated while the
    # video is being processed further down the script
    steps_slot = st.empty()
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])

    # NEW: which step are we on? Stickers ready -> 3, video uploaded -> 2,
    # nothing yet -> 1
    if st.session_state.get("stickers"):
        current_step = 3
    elif uploaded:
        current_step = 2
    else:
        current_step = 1
    steps_slot.markdown(steps_html(current_step), unsafe_allow_html=True)
    # The button sits right under the upload box
    make = uploaded and st.button("Make stickers")

# The clip gets the bigger share, the stop-motion preview the smaller
# vertical_alignment="bottom" lines up the bottoms of the two previews, so
# the "Your clip" and "Stop-motion preview" badges sit at the same height
# NEW: in a keyed container too (.st-key-previews in responsive.css)
with col_right.container(key="previews"):
    col_clip, col_gif = st.columns([3, 2], gap="medium", vertical_alignment="bottom")

# The GIF isn't made until further down the script, so keep an empty slot
# here and fill it in once the GIF exists. Until then, show the placeholder
preview_slot = col_gif.empty()
preview_slot.markdown(preview_placeholder("Your stop-motion will appear here ✂️",
                                          tape="right"),
                      unsafe_allow_html=True)

if not uploaded:
    # NEW: placeholder for the clip until a video is uploaded, matching the
    # stop-motion one next to it (tilted the other way)
    col_clip.markdown(preview_placeholder("Your clip will appear here 🎬",
                                          label="Your clip", tilt=-2, tape="left"),
                      unsafe_allow_html=True)
else:
    with col_clip:
        # NEW: washi tape on the video's top-left corner, like the placeholder
        # had. st.video can't hold extra HTML, so the tape goes in a
        # zero-height box just above the video and hangs down over its corner.
        # z-index keeps it on top of the video
        st.markdown('<div style="position:relative; height:0; z-index:5">'
                    '<span class="tape left" style="top:14px"></span></div>',
                    unsafe_allow_html=True)
        # getvalue() gives the bytes without using up the upload, so it can
        # still be saved to a temp file below
        st.video(uploaded.getvalue())
        # Video styled like a cut-out stuck on the mat: rounded corners, a
        # deep shadow, a yellow dashed "cut here" line around it, and a
        # slight tilt. It lifts and straightens on hover like the stickers.
        # Comes with a yellow "Your clip" label to match "Stop-motion preview"
        # CHANGED: max-height is now 320px or 45% of the window height,
        # whichever is smaller, so it fits on short screens too
        st.markdown('<style>video{max-height:min(320px, 45vh); width:auto !important; '
                    'max-width:100%; display:block; margin:12px auto 0; '
                    'background:#000; border-radius:14px; '
                    'box-shadow:0 14px 28px rgba(0,0,0,0.45); '
                    'outline:2px dashed #f5c542; outline-offset:6px; '
                    'transform:rotate(-2deg); transition:transform 0.25s ease}'
                    'video:hover{transform:rotate(0deg) translateY(-4px) scale(1.02)}'
                    '</style>'
                    '<div style="text-align:center; margin-top:18px">'
                    '<span style="background:#f5c542; color:#1f3d33; font-weight:600; '
                    'font-size:14px; padding:4px 14px; border-radius:999px">'
                    'Your clip</span></div>',
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

if make:                                  # CHANGED: the button is now drawn at the top
    # OpenCV needs a real file path, so save the upload to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded.getvalue())   # CHANGED: getvalue(), as the preview reads it too
        video_path = tmp.name

    # NEW: step 2 pulses while the video is processed
    steps_slot.markdown(steps_html(2, working=True), unsafe_allow_html=True)
    bar = st.progress(0.0, text=LOADING_MESSAGES[0])
    started = time.perf_counter()        # NEW: time the run for the stats panel
    try:
        # CHANGED: progress_with_messages(bar) instead of bar.progress, so the
        # bar shows the fun loading messages
        stickers = extract_stickers(video_path, every_n, progress_with_messages(bar),
                                    overlap,
                                    night=AUTO_ON_OFF[night_choice],
                                    shadows=AUTO_ON_OFF[shadow_choice])
    except ValueError as err:
        st.session_state.stickers = []   # NEW: clear old stickers so they don't show under the error
        st.session_state.stats = None    # NEW: and the old stats
        steps_slot.markdown(steps_html(2), unsafe_allow_html=True)   # NEW: stop pulsing
        st.error(f"{err} Try an MP4 file.")
        render_footer()                  # NEW: keep the footer on the page
        st.stop()
    bar.empty()
    # NEW: save the stickers so they're still there when a checkbox is ticked.
    # Streamlit re-runs the whole script on every click, and the button above
    # is only True on the run straight after it's pressed
    st.session_state.stickers = stickers
    # NEW: a new number for each video, so the checkboxes start ticked again
    st.session_state.run_id = st.session_state.get("run_id", 0) + 1

    # NEW: numbers for the stats panel, saved like the stickers so they
    # stay on screen when a checkbox is clicked
    frames, seconds = video_info(video_path)
    st.session_state.stats = {"stickers": len(stickers), "seconds": seconds,
                              "frames": frames,
                              "took": time.perf_counter() - started}

    if not stickers:
        steps_slot.markdown(steps_html(2), unsafe_allow_html=True)   # NEW: stop pulsing
        st.info("No movement found. Try a lower 'every Nth frame' value in Settings.")
        render_footer()                  # NEW: keep the footer on the page
        st.stop()

    # NEW: a little celebration when the stickers are ready. Only on the run
    # straight after "Make stickers", not every time a checkbox is clicked
    confetti()
    # NEW: all done, so step 3 is highlighted and steps 1 and 2 get ticks
    steps_slot.markdown(steps_html(3), unsafe_allow_html=True)

# NEW: show the saved stickers on every re-run, not just straight after the button
stickers = st.session_state.get("stickers")

# NEW: no stickers yet, so show the faded examples instead
if not stickers:
    # The line under the examples changes once a video is uploaded
    message = ("Click <b>Make stickers</b> and they'll land here ✂️" if uploaded
               else "Upload a clip and your stickers land here ✂️")
    st.markdown(example_stickers_html(message), unsafe_allow_html=True)
if stickers:
    # anchor=False hides the little link icon Streamlit puts next to headings
    st.subheader(f"{len(stickers)} stickers, in the order they appeared", anchor=False)

    # NEW: the stats panel, if this run saved some
    if st.session_state.get("stats"):
        st.markdown(stats_panel(st.session_state.stats), unsafe_allow_html=True)
    # NEW: a slot for the timeline strip, filled in once we know which
    # stickers are ticked
    timeline_slot = st.empty()

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

    # NEW: fill in the timeline strip
    video_seconds = (st.session_state.get("stats") or {}).get("seconds", 0)
    timeline_slot.markdown(timeline_html(stickers, [st.session_state[k] for k in keys],
                                         video_seconds),
                           unsafe_allow_html=True)

    # NEW: Tick all / Untick all buttons and a "Keeping X of Y" counter.
    # CHANGED: narrow columns with a small gap keep the buttons side by side,
    # and the counter is a yellow pill to match the other badges
    # NEW: keyed container (.st-key-tick-row) so the buttons only take the
    # width they need and wrap on small screens
    with st.container(key="tick-row"):
        col_all, col_none, col_count, _ = st.columns([1, 1, 2, 4], gap="small",
                                                      vertical_alignment="center")
    col_all.button("Tick all", on_click=set_all, args=(keys, True))
    col_none.button("Untick all", on_click=set_all, args=(keys, False))
    col_count.markdown(
        '<span style="background:#f5c542; color:#1f3d33; font-weight:600; '
        'font-size:15px; padding:6px 16px; border-radius:999px; white-space:nowrap">'
        f'Keeping {len(keep)} of {len(stickers)} stickers</span>',
        unsafe_allow_html=True)

    # NEW: the sticker grid, 4 per row, each sticker with its checkbox underneath
    # CHANGED: in a keyed container (.st-key-sticker-grid), so on smaller
    # screens each row of 4 wraps to 2 + 2, or 1 per line on a phone
    grid = st.container(key="sticker-grid")
    for row_start in range(0, len(stickers), 4):
        cols = grid.columns(4)
        for col, i in zip(cols, range(row_start, min(row_start + 4, len(stickers)))):
            t, sticker = stickers[i]
            with col:
                # CHANGED: pass in whether it's ticked, so unticked ones fade out
                st.markdown(sticker_figure(sticker, t, i, st.session_state[keys[i]]),
                            unsafe_allow_html=True)
                st.checkbox(f"Keep ({t:.1f} s)", key=keys[i])
                # NEW: download just this one sticker as a transparent PNG
                ok, one_png = cv2.imencode(".png", sticker)
                st.download_button("⬇ PNG", one_png.tobytes(),
                                   file_name=f"sticker_{i + 1:02d}_{t:.1f}s.png",
                                   mime="image/png",
                                   key=f"dl_{st.session_state.run_id}_{i}")

    # NEW: nothing to download if every box is unticked
    if not keep:
        # NEW: say why the preview is empty
        preview_slot.markdown(preview_placeholder("Tick a sticker to see the stop-motion",
                                                  tape="right"),
                              unsafe_allow_html=True)
        st.warning("Tick at least one sticker to download.")
        render_footer()                  # NEW: keep the footer on the page
        st.stop()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i, (t, sticker) in enumerate(keep):           # CHANGED: only the ticked stickers
            ok, png = cv2.imencode(".png", sticker)          # PNG keeps transparency
            zf.writestr(f"sticker_{i:03d}_{t:.1f}s.png", png.tobytes())

    # CHANGED: three download buttons now, with the printable sheet in the middle
    # NEW: keyed container (.st-key-downloads) so they stack on small screens
    with st.container(key="downloads"):
        col_zip, col_sheet, col_gif = st.columns(3)
    with col_zip:
        st.download_button("Download all stickers (.zip)", zip_buffer.getvalue(),
                           file_name="stickers.zip", mime="application/zip")
    with col_sheet:
        # NEW: printable A4 sticker sheet of the ticked stickers
        st.download_button("Download printable sheet (.png)", make_sticker_sheet(keep),
                           file_name="sticker_sheet.png", mime="image/png")
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
            # NEW: the GIF sits inside the same dashed yellow box as the
            # placeholder it replaces (same size, same tilt, same washi tape),
            # centred and shrunk to fit, so the spot doesn't jump around
            preview_slot.markdown(f'<figure class="sticker" style="--tilt:2deg">'
                                  f'<div style="position:relative; border:2px dashed #f5c542; '
                                  f'border-radius:16px; width:100%; max-width:320px; '
                                  f'height:190px; margin:auto; padding:12px; '
                                  f'display:flex; align-items:center; justify-content:center">'
                                  f'<span class="tape right"></span>'
                                  f'<img src="data:image/gif;base64,{b64_gif}" '
                                  f'alt="Stop-motion preview" '
                                  f'style="max-width:100%; max-height:100%; object-fit:contain">'
                                  f'</div>'
                                  f'<figcaption>Stop-motion preview</figcaption></figure>',
                                  unsafe_allow_html=True)

# NEW: footer at the bottom of the page
render_footer()