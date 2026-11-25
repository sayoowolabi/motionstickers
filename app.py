"""
app.py - the user interface.
Run with: streamlit run app.py

Users upload a video. The extracted stickers are then run through
extract_stickers() in stickers.py and shown in chronological order
on a 'cutting mat'. The extracted stickers can then be downloaded
in transparent PNG format.

This file's role within the project:
The image processing (removal of shadows, night mode, cutting out
the moving objects, background subtraction) occurs in stickers.py.
This file forms the user interface around this. It does the following:
1. Receives the user's video and user inputs from the sidebar,
2. Runs the image processing pipeline and provides feedback,
3. Allows the user to view the results and uncheck any unwanted
stickers (there is a human-in-the-loop as the image processing
pipeline is not perfect),
4. Outputs the selected stickers as a printable A4 sticker sheet,
a stop-motion GIF and as a zip containing all extracted stickers
in transparent PNG format.

Image processing that occurs in this file:
- Paste sticker onto sticker sheet using alpha compositing (paste_sticker) [33]
- Cut line around sticker - produced by morphological dilation of the
  mask of the sticker followed by contour detection (make_sticker_sheet) [7], [34]
- Resampling of stickers based on area to prevent aliasing (jagged
  edges caused by simple resizing) when resizing stickers [18], [23]

Styling:
Colours are defined in .streamlit/config.toml. All other styling is
defined in styles/style.css. styles/responsive.css ensures the page
can be viewed on mobile devices and smaller screens.

References:
Numbers in [ ] refer to the References listed in README.md.

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

# Streamlit [20] builds the web page from this script
st.set_page_config(page_title="Motion Stickers!", page_icon="✂️", layout="wide")

# Load the page styling from styles/style.css, then styles/responsive.css,
# which is loaded second so its rules for smaller screens win over the
# normal ones
styles = Path(__file__).parent / "styles"
css = (styles / "style.css").read_text(encoding="utf-8")
css += "\n" + (styles / "responsive.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sticker_figure(sticker, t, i, kept=True):
    """Convert a single BGRA sticker to an HTML &lt;figure&gt; in the sticker page.
        Untaxed stickers have the class 'unticked' which allows CSS to fade them out.
        Clicking a sticker will display it at full size. Clicking again will close it."""
    # Stickers are stored in BGRA format (blue, green, red, then alpha). The
    # alpha value represents transparency and is zero where the background
    # has been removed from the sticker. PNG is used because it supports
    # transparency and is lossless. JPEG does not support transparency and is
    # lossy, so it would not work here.
    # The image is then base64-encoded so it can be embedded directly in the
    # HTML as a data: URI, so there is no need for file storage.
    # A full-size copy is created before the image is resized

    ok, big_png = cv2.imencode(".png", sticker)
    big_b64 = base64.b64encode(big_png.tobytes()).decode()
    big_h, big_w = sticker.shape[:2]                # full size, for the caption

    h, w = sticker.shape[:2]
    if w > 400:                                   # shrink for display only
        sticker = cv2.resize(sticker, (400, int(h * 400 / w)))
    ok, png = cv2.imencode(".png", sticker)
    b64 = base64.b64encode(png.tobytes()).decode()
    tilt = [-3, 2, -1.5, 3, -2, 1][i % 6]         # varied tilt, like hand-placed stickers
    faded = "" if kept else " unticked"           # Faded look for unticked stickers

    # Click to enlarge: implemented without JavaScript by using a <details>
    # element, which opens and closes on click. When it is open, the CSS
    # makes the .zoom-big element fill the screen with a dark overlay, and
    # clicking anywhere on the overlay closes the view again

    return (f'<details class="zoom"><summary>'
            f'<figure class="sticker{faded}" style="--tilt:{tilt}deg">'
            f'<img src="data:image/png;base64,{b64}" alt="Sticker at {t:.1f} seconds">'
            f'<figcaption>{t:.1f} s</figcaption></figure>'
            f'<span class="zoom-big">'
            f'<img src="data:image/png;base64,{big_b64}" alt="Sticker at {t:.1f} seconds, full size">'
            f'<span class="zoom-caption">{t:.1f} s · {big_w} × {big_h} px original · click anywhere to close</span>'
            f'</span></summary></details>')


# Confetti colours, matching the yellow buttons plus a few bright extras
CONFETTI_COLOURS = ["#f5c542", "#ff6b6b", "#4ecdc4", "#ffffff", "#ff9ff3", "#54a0ff"]


def confetti(pieces=120, slowest=8.0, fastest=5.0):

    """Confetti that falls down the screen when the stickers are loaded.
    Uses only HTML and CSS (no JavaScript), so it can also be used offline,
    with no internet required.
    slowest/fastest are the number of seconds a piece of confetti takes to
    fall. The higher these values are, the slower the confetti falls."""

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
    """Dashed "empty sticker" for a preview space with no content yet (no
    video uploaded, or no GIF made yet), instead of an empty space on the page.
    tape: which top corner gets a strip of washi tape ("left" or "right")."""

    return (f'<figure class="sticker" style="--tilt:{tilt}deg">'
            '<div style="position:relative; border:2px dashed #f5c542; border-radius:16px; '
            'width:100%; max-width:320px; height:190px; margin:auto; padding:12px; '
            'display:flex; align-items:center; justify-content:center; '
            'text-align:center">'
            f'<span class="tape {tape}"></span>'            # Washi tape
            f'<span style="opacity:0.85">{message}</span></div>'
            f'<figcaption>{label}</figcaption></figure>')


# Fun messages shown on the progress bar while the stickers are made.
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

    """Wrapper around the progress bar that also displays a fun message.
        Called by extract_stickers() with the amount completed (0 to 1),
        just like before when it called bar.progress."""
    
    start = time.monotonic()     # monotonic clock never jumps backwards [19]

    def update(fraction, *args, **kwargs):
        step = int((time.monotonic() - start) / MESSAGE_SECONDS)
        message = LOADING_MESSAGES[step % len(LOADING_MESSAGES)]
        bar.progress(min(max(float(fraction), 0.0), 1.0), text=message)

    return update


def video_info(video_path):

    """Number of frames in the video and duration in seconds for use in
        stats panel. Some videos may not report frame rate so account for
        possible divide by zero."""

    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    seconds = frames / fps if fps > 0 else 0.0
    return frames, seconds


def stats_panel(stats):

    """A row of yellow "stat stickers" showing what happened on this run.
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


# The three steps at the top of the page. These are "live" in that the
# current step is highlighted, completed steps are annotated with a tick
# and future steps are faded. The styles are in styles/style.css (.step.current/.done/.todo)
STEPS = ["Upload your footage", "We find what moves", "Peel off the stickers"]


def steps_html(current, working=False):

    """HTML for the steps row. current is the step you're on (1, 2 or 3).
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


# Example stickers that are displayed before the actual stickers are
# created, to avoid blank space at the bottom of the page and to show new
# users the type of stickers that will be created.
# A list of (emoji, time) tuples that mimic objects a home camera might see

EXAMPLE_STICKERS = [("🚶", 0.5), ("🐕", 2.0), ("📦", 3.5), ("🐈", 5.0), ("🚲", 7.5)]


def example_stickers_html(message="Upload a clip and your stickers land here ✂️"):
   
    """A row of faded example stickers with a line explaining them.
    The styling is in styles/style.css (.examples and .example-sticker)."""

    tilts = [-4, 3, -2, 4, -3]
    stickers = "".join(f'<div class="example-sticker" style="--tilt:{tilt}deg">'
                       f'<span class="emoji">{emoji}</span><em>{t:.1f} s</em></div>'
                       for (emoji, t), tilt in zip(EXAMPLE_STICKERS, tilts))
    return ('<div class="examples">'
            f'<p>{message}</p>'
            f'<div class="example-row">{stickers}</div></div>')


# Timeline strip - bar matching length of video with a dot for each
# sticker cut from it. Hover dots to see their times. Stickers not
# with a tick are faded. Styles in styles/style.css (.timeline, .track, .dot)

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

# AI USE: AI assistance (ChatGPT) was used for this section because:
# 1. The stickers have soft transparent edges, so they needed alpha
#    compositing rather than a plain copy onto the page.
# 2. The cut line combines dilation and contour tracing, and the 10px
#    padding meant the contour had to be offset to line up correctly.
# Reviewed and tested by the team.

# Sticker sheet that can be printed out.
# Uses OpenCV to alpha-blend the selected stickers onto a cream "cutting"
# background and to draw a grey cut line around each sticker. The cut line is
# created by performing dilation on the sticker mask to grow it and then detecting the sticker's contours.

SHEET_W, SHEET_H = 1240, 1754           # A4 at 150 dpi, in pixels (8.27 x 11.69 in x 150)
SHEET_MARGIN = 80
SHEET_TOP = 210                         # room for the title
SHEET_BG = (225, 240, 245)              # cream, in BGR (#f5f0e1)
SHEET_GRID = (205, 222, 230)            # slightly darker cream for the mat grid
SHEET_TEXT = (51, 61, 31)               # dark green, in BGR (#1f3d33)
CUT_LINE = (160, 160, 160)              # grey cut line


def paste_sticker(page, sticker, x, y):
    
    """Blending a BGRA sticker over a BGR page at position (x, y) for top left:
        result = alpha * sticker + (1 - alpha) * page.

        This is the Porter and Duff [33] "over" operator. If alpha == 1 then the
        sticker completely covers the page. If alpha == 0 then the page is visible.
        If in between then there is a soft edge as there is blending between the two
        rather than a jagged edge."""

    h, w = sticker.shape[:2]
    # Scale alpha from 0-255 to 0-1. Slicing 3:4 (not just 3) keeps it as an
    # (h, w, 1) array, so NumPy broadcasting applies it to all three colour
    # channels at once with no loop
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

    # Title and a short instruction, drawn with cv2.putText [24]

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

        # INTER_AREA averages the old pixels that fall inside each new pixel.
        # This is the preferred interpolation for shrinking images, since it
        # prevents the aliasing (moire or jagged edges) that bilinear or
        # nearest-neighbour interpolation can cause [18], [23]

        small = cv2.resize(sticker, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Centre it in the cell
        x = SHEET_MARGIN + col * cell_w + (cell_w - new_w) // 2
        y = SHEET_TOP + row * cell_h + pad + (cell_h - 2 * pad - label_h - new_h) // 2

        # Cut line: grow the sticker's mask a little so the line sits just
        # outside it, then draw the outline of the grown mask
        # Binary mask from the alpha channel: 255 where the sticker is, 0 elsewhere

        mask = (small[:, :, 3] > 0).astype(np.uint8) * 255

        # Pad with 10 px of empty border first, so the dilation has room to
        # grow outwards and the outline isn't clipped at the image edge

        mask = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

        # Morphological dilation [7]: a pixel becomes white if any pixel
        # under the 15x15 elliptical kernel is white. This grows the shape
        # evenly outwards by about 7 px. Using an ellipse (instead of a
        # square) keeps the corners of the cut line rounded, like a real
        # die-cut sticker

        grown = cv2.dilate(mask, kernel)

        # findContours uses the border-following algorithm of Suzuki and
        # Abe [34] to trace the border of the white area. CHAIN_APPROX_SIMPLE
        # compresses straight segments to their end points (reducing memory
        # usage), and RETR_EXTERNAL only returns the outer shape (no holes)

        contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # The contour is in the padded mask's coordinates, so offset it back
        # by the 10 px padding to line it up with the sticker on the page

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


# Footer details
TEAM = ["Patrycja", "Chloe", "Sayo"]
MODULE = "IDSP 1701 Image Processing · 4th Year · 2026"
REPO_URL = "https://github.com/sayoowolabi/motionstickers"


def render_footer():
    """Footer with a dashed "cut here" line, the team as little name-tag
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


# ------------------------------------------------------------------ layout ------------------------------------------------------------------ 

# The left column contains all the actions (title, steps, upload, button).
# The right side contains two side-by-side previews: the input (your clip)
# and the output (the stop-motion).
# vertical_alignment centres the previews vertically against the left
# column, so there is no large gap below the title.
# [6, 5] gives the left column some extra space, so all three steps fit
# onto a single row.
# The two columns are inside a container with a key. Streamlit gives the
# container the CSS class .st-key-top-layout, which responsive.css uses to
# move the previews underneath the title on small screens

with st.container(key="top-layout"):
    col_left, col_right = st.columns([6, 5], gap="large", vertical_alignment="center")
with col_left:
    st.title("Motion Stickers!")

    # The steps go in a slot, so they can be updated while the
    # video is being processed further down the script
    
    steps_slot = st.empty()
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])

    # Which step are we on? Stickers ready -> 3, video uploaded -> 2,
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

# The clip gets the larger share of the space and the stop-motion preview
# the smaller share.
# Setting vertical_alignment="bottom" aligns the bottoms of the two
# previews, so the "Your clip" and "Stop-motion preview" labels sit at the same height.
# Also inside a keyed container (.st-key-previews in responsive.css)

with col_right.container(key="previews"):
    col_clip, col_gif = st.columns([3, 2], gap="medium", vertical_alignment="bottom")

# The GIF isn't made until further down the script, so keep an empty slot
# here and fill it in once the GIF exists. Until then, show the placeholder

preview_slot = col_gif.empty()
preview_slot.markdown(preview_placeholder("Your stop-motion will appear here ✂️",
                                          tape="right"),
                      unsafe_allow_html=True)

if not uploaded:
    # Placeholder for the clip until a video is uploaded, matching the
    # stop-motion one next to it (tilted the other way)
    col_clip.markdown(preview_placeholder("Your clip will appear here 🎬",
                                          label="Your clip", tilt=-2, tape="left"),
                      unsafe_allow_html=True)
else:
    with col_clip:

        # Washi tape on the top-left corner of the video, like the placeholder has.
        # st.video doesn't allow additional HTML, so the tape goes in a
        # zero-height box directly above the video and hangs down over its
        # corner. z-index makes the tape appear on top of the video

        st.markdown('<div style="position:relative; height:0; z-index:5">'
                    '<span class="tape left" style="top:14px"></span></div>',
                    unsafe_allow_html=True)
        
        # getvalue() gives the bytes without using up the upload, so it can
        # still be saved to a temp file below

        st.video(uploaded.getvalue())

        # Video that appears as a cut-out on the mat, with rounded corners, a
        # drop shadow, a slight tilt and a yellow dashed "cut here" frame. When
        # hovered over, it straightens and lifts slightly (the same as the
        # stickers). Also includes a yellow "Your clip" label to match the
        # "Stop-motion preview" one. The height of the video is limited to
        # 320px or 45% of the window height (whichever is smaller), so it
        # also fits on smaller screens

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

    # Motion detection examines every frame (to keep the background model
    # up to date), so this slider only affects how often a sticker is made,
    # not how accurate the motion detection is

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

# AI USE: AI assistance (ChatGPT) was used for this section because:
# 1. Streamlit re-runs the whole script on every click, so the stickers,
#    stats and checkbox ticks had to be kept in st.session_state.
# 2. Some parts (steps, timeline, GIF preview) are drawn before their
#    content exists, so st.empty() slots are filled in later.
# Reviewed and tested by the team.

if make:                                  # the button itself is drawn at the top

    # OpenCV needs a real file path, so save the upload to a temp file.
    # (Streamlit only holds the upload in memory, and cv2.VideoCapture can't
    # read from memory)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded.getvalue())   # getvalue(), as the preview reads it too
        video_path = tmp.name

    # Step 2 pulses while the video is processed

    steps_slot.markdown(steps_html(2, working=True), unsafe_allow_html=True)
    bar = st.progress(0.0, text=LOADING_MESSAGES[0])
    started = time.perf_counter()        # Time the run for the stats panel [19]
    try:
        # progress_with_messages(bar) rather than bar.progress, so the
        # bar shows the fun loading messages
        stickers = extract_stickers(video_path, every_n, progress_with_messages(bar),
                                    overlap,
                                    night=AUTO_ON_OFF[night_choice],
                                    shadows=AUTO_ON_OFF[shadow_choice])
        
    # extract_stickers raises ValueError when OpenCV can't open or decode
    # the video (e.g. an unsupported codec), so show a friendly message
    # instead of a crash

    except ValueError as err:
        st.session_state.stickers = []   # Clear old stickers so they don't show under the error
        st.session_state.stats = None    # and the old stats
        steps_slot.markdown(steps_html(2), unsafe_allow_html=True)   # Stop pulsing
        st.error(f"{err} Try an MP4 file.")
        render_footer()                  # Keep the footer on the page
        st.stop()
    bar.empty()

    # Save the stickers so they're still there when a checkbox is ticked.
    # Streamlit re-runs the whole script on every click, and the button above
    # is only True on the run straight after it's pressed

    st.session_state.stickers = stickers

    # A new number for each video, so the checkboxes start ticked again
    st.session_state.run_id = st.session_state.get("run_id", 0) + 1

    # Numbers for the stats panel, saved like the stickers so they
    # stay on screen when a checkbox is clicked

    frames, seconds = video_info(video_path)
    st.session_state.stats = {"stickers": len(stickers), "seconds": seconds,
                              "frames": frames,
                              "took": time.perf_counter() - started}

    if not stickers:
        steps_slot.markdown(steps_html(2), unsafe_allow_html=True)   # Stop pulsing
        st.info("No movement found. Try a lower 'every Nth frame' value in Settings.")
        render_footer()                  # Keep the footer on the page
        st.stop()

    # A little celebration when the stickers are ready. Only on the run
    # straight after "Make stickers", not every time a checkbox is clicked

    confetti()
    # All done, so step 3 is highlighted and steps 1 and 2 get ticks
    steps_slot.markdown(steps_html(3), unsafe_allow_html=True)

# Show the saved stickers on every re-run, not just straight after the button
stickers = st.session_state.get("stickers")

# No stickers yet, so show the faded examples instead
if not stickers:
    # The line under the examples changes once a video is uploaded
    message = ("Click <b>Make stickers</b> and they'll land here ✂️" if uploaded
               else "Upload a clip and your stickers land here ✂️")
    st.markdown(example_stickers_html(message), unsafe_allow_html=True)
if stickers:
    # anchor=False hides the little link icon Streamlit puts next to headings
    st.subheader(f"{len(stickers)} stickers, in the order they appeared", anchor=False)

    # The stats panel, if this run saved some
    if st.session_state.get("stats"):
        st.markdown(stats_panel(st.session_state.stats), unsafe_allow_html=True)

    # A slot for the timeline strip, filled in once we know which
    # stickers are ticked

    timeline_slot = st.empty()

    # Human-in-the-loop review. The pipeline can't catch every mistake
    # (e.g. background fragments or stray blobs), so the user can untick
    # bad stickers. Only ticked stickers go into the GIF and the zip

    st.caption("Untick any sticker you don't want, like a stray blob or a bit of "
               "background. Only ticked stickers go into the downloads.")

    # One checkbox key per sticker. The run_id makes the keys different
    # for each video, so a new video starts with every box ticked

    keys = [f"keep_{st.session_state.run_id}_{i}" for i in range(len(stickers))]
    for key in keys:
        st.session_state.setdefault(key, True)        # ticked by default

    # The Tick all / Untick all buttons call this before the page re-runs
    def set_all(keys, value):
        for key in keys:
            st.session_state[key] = value

    # The ticked stickers, worked out before drawing the page so the
    # counter can go at the top

    keep = [(t, sticker) for key, (t, sticker) in zip(keys, stickers)
            if st.session_state[key]]

    # Fill in the timeline strip
    video_seconds = (st.session_state.get("stats") or {}).get("seconds", 0)
    timeline_slot.markdown(timeline_html(stickers, [st.session_state[k] for k in keys],
                                         video_seconds),
                           unsafe_allow_html=True)

    # Tick all / Untick all buttons and a "Keeping X of Y" counter.
    # Narrow columns with minimal spacing keep the buttons next to each
    # other, and the counter is a yellow pill to match the other badges.
    # Keyed container (.st-key-tick-row), so the buttons only take up the
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

    # The sticker grid, 4 per row, each sticker with its checkbox underneath
    # In a keyed container (.st-key-sticker-grid), so on smaller
    # screens each row of 4 wraps to 2 + 2, or 1 per line on a phone

    grid = st.container(key="sticker-grid")
    for row_start in range(0, len(stickers), 4):
        cols = grid.columns(4)
        for col, i in zip(cols, range(row_start, min(row_start + 4, len(stickers)))):
            t, sticker = stickers[i]
            with col:
                # Pass in whether it's ticked, so unticked ones fade out
                st.markdown(sticker_figure(sticker, t, i, st.session_state[keys[i]]),
                            unsafe_allow_html=True)
                st.checkbox(f"Keep ({t:.1f} s)", key=keys[i])
                # Download just this one sticker as a transparent PNG
                ok, one_png = cv2.imencode(".png", sticker)
                st.download_button("⬇ PNG", one_png.tobytes(),
                                   file_name=f"sticker_{i + 1:02d}_{t:.1f}s.png",
                                   mime="image/png",
                                   key=f"dl_{st.session_state.run_id}_{i}")

    # Nothing to download if every box is unticked
    if not keep:
        # Say why the preview is empty
        preview_slot.markdown(preview_placeholder("Tick a sticker to see the stop-motion",
                                                  tape="right"),
                              unsafe_allow_html=True)
        st.warning("Tick at least one sticker to download.")
        render_footer()                  # Keep the footer on the page
        st.stop()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i, (t, sticker) in enumerate(keep):           # Only the ticked stickers
            ok, png = cv2.imencode(".png", sticker)          # PNG keeps transparency
            zf.writestr(f"sticker_{i:03d}_{t:.1f}s.png", png.tobytes())

    # Three download buttons, with the printable sheet in the middle.
    # Keyed container (.st-key-downloads) so they stack on small screens
    with st.container(key="downloads"):
        col_zip, col_sheet, col_gif = st.columns(3)
    with col_zip:
        st.download_button("Download all stickers (.zip)", zip_buffer.getvalue(),
                           file_name="stickers.zip", mime="application/zip")
    with col_sheet:
        # Printable A4 sticker sheet of the ticked stickers
        st.download_button("Download printable sheet (.png)", make_sticker_sheet(keep),
                           file_name="sticker_sheet.png", mime="image/png")
    with col_gif:
        gif = make_gif(keep)                               # Only the ticked stickers [9]
        if gif:
            st.download_button("Download stop-motion (.gif)", gif,
                               file_name="stickers.gif", mime="image/gif")
            
            # Preview the GIF without needing to download it.
            # Displayed as HTML, because st.image() resizes the GIF and only
            # displays its first frame.
            # Positioned to the right of the title, not below the button

            b64_gif = base64.b64encode(gif).decode()

            # The GIF sits in the same dashed yellow box as the placeholder
            # it replaces (same size, same tilt, same washi tape), centred
            # and shrunk to fit, so the preview doesn't jump around the page

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

# Footer at the bottom of the page
render_footer()