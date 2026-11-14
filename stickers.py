"""
stickers.py - the image-processing side of the project.

The interface (app.py) only calls extract_stickers(). Everything that turns a
video into sticker cutouts lives here, so this file can be developed, tested
and submitted independently of the interface.

CURRENT STATE: steps 1-2 (Sayo), 3-4 (Pati) and 5-6 (Chloe) are in.
Step 2 now uses a running-average background instead of comparing
with the previous frame, to reduce ghosting.

Week 10 changes (Sayo, steps 1-2):
  - Selective background update: the background only learns at full speed
    where nothing is moving, so people who stand still no longer fade out
    and don't leave a trail when they move on.
  - The starting background is the median of frames spread across the whole
    clip, not the first frame, so someone already in frame 1 is no longer
    baked into the background as a ghost.
  - CLAHE (night contrast) is switched on or off once per clip instead of
    per frame, so clips that hover around the brightness limit don't flicker.

KNOWN LIMITATIONS:
  - If something stays in the same place for more than half of the clip,
    the median treats it as background. If it then leaves, the empty spot
    shows up as motion until the slow update absorbs it.
  - A permanent change (door left open, light switched on) takes about
    1 / (BACKGROUND_ALPHA * MOVING_ALPHA_FACTOR) frames to be absorbed,
    because it counts as "moving" and is learned at the slow rate.
"""

import io

import cv2
import numpy as np


# Learning rate for the running-average background (step 2).
# The background "remembers" roughly the last 1 / alpha frames:
#   small alpha -> slow to adapt (stable, but lighting changes and objects
#                  that stop moving take longer to fade into the background)
#   large alpha -> fast to adapt (but a slow-moving object starts to blend
#                  into the background, leaving a hollow or trailing mask)
# 0.01 was chosen from the alpha sweep in evaluate.py on the synthetic test
# clip: 0.005 and 0.01 scored the same (mean IoU ~0.83), while 0.02 and
# above let the moving object blend into the background and leave a trail
# behind it (mean IoU 0.56-0.64). 0.01 adapts twice as fast as 0.005
# (~100 frames, about 4 s at 25 fps), so it was preferred.
# Re-check this on real footage once it has been hand-marked.
BACKGROUND_ALPHA = 0.01

# Selective update: pixels marked as moving are still blended in, but this
# many times slower (0.1 x 0.01 = 0.001, roughly the last 1000 frames).
# The selective update in Piccardi (2004, Section 2.1, eq. 3, from Koller
# et al., 1994) doesn't update moving pixels at all. We use a slow rate
# instead because, with no update, anything moved and left there (a door
# left open, a light switched on, someone in frame 1 who walks away) would
# be flagged as motion forever: a permanent ghost (Cucchiara et al., 2003).
MOVING_ALPHA_FACTOR = 0.1

# Frames below this mean brightness count as night footage and get CLAHE.
NIGHT_BRIGHTNESS = 70

# How many frames, spread across the clip, make the starting background.
BACKGROUND_SAMPLES = 25


def make_sticker(frame, motion_mask):
    """
    Turn one frame into a sticker (BGRA image, transparent background),
    or return None if nothing is moving.

    Parameters:
        frame:       the original colour frame (BGR)
        motion_mask: the frame's motion mask from background_motion_mask(),
                     or None if there is no background yet

    Our pipeline, built up from the research:
      1. Preprocess: greyscale, blur, night-frame check,
         CLAHE on night frames.
      2. Motion mask: compare with a running-average background,
         then threshold.
      3. Clean the mask: morphological opening/closing,
         keep the largest contour, fill it in.
      4. Enhance the object: contrast (CLAHE) and sharpening (unsharp mask).
      5. Cut it out: use the mask as the alpha channel, crop to the
         bounding box, add a white outline by dilating the mask.
    """

    # Steps 1 and 2 (preprocessing and the motion mask) are done once per
    # frame in extract_stickers(), because the selective background update
    # needs the motion mask on every frame, not just sticker frames.
    if motion_mask is None:
        return None

    if cv2.countNonZero(motion_mask) < 500:
        # If the motion is too small, return None
        return None

     # Step 3 (Pati): clean the mask into one solid shape
    cleaned = clean_mask(motion_mask)
    if cleaned is None:
        return None

    # Step 4 (Pati): make the object clearer before cutting it out
    enhanced = enhance_object(frame)

    # Step 5 (Chloe): cut the object out as a transparent sticker
    return cut_out_sticker(enhanced, cleaned)

def preprocess_frame(frame, use_clahe=None):
    """
    Preprocess a video frame before motion detection.

    Converts the frame to greyscale and applies Gaussian blur to reduce
    noise. If it is night footage, CLAHE is applied to improve the
    contrast and make objects easier to distinguish from the background.

    Parameters:
        frame:     The original colour video frame.
        use_clahe: True/False = the decision already made for the whole
                   clip (see estimate_background()). None = decide from
                   this frame's own brightness (the old per-frame rule,
                   which can flicker on clips near NIGHT_BRIGHTNESS).

    Returns:
        The preprocessed greyscale frame.
    """
    #Converts each video frame from colour to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Applies a small Gaussian blur to remove tiny pixel-level changes
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #No clip-wide decision given, so check this frame's average brightness
    if use_clahe is None:
        use_clahe = np.mean(gray) < NIGHT_BRIGHTNESS

    if use_clahe:
        #increases the contrast of dark footage to better distinguish between objects
        gray = night_contrast(gray)

    return gray


def night_contrast(gray):
    """CLAHE on a greyscale frame, used for night footage."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def estimate_background(video_path, samples=BACKGROUND_SAMPLES):
    """
    Build the starting background before the main pass over the video.

    Reads `samples` frames spread evenly across the whole clip and takes
    the median of each pixel. Someone walking through the scene only
    covers a given pixel in a few of those frames, so the median picks the
    empty scene, even if they are already there in frame 1. This replaces
    starting from the first frame, which baked anyone in it into the
    background as a ghost (Cucchiara et al., 2003). Unlike a warm-up
    period, no frames are skipped, so motion at the very start of a short
    clip still makes stickers. Median background: Piccardi (2004), Section
    2.2 (temporal median filter, which works even on sub-sampled frames);
    taking the median of ~25 frames spread over the clip: LearnOpenCV (2019).
    It only works if each pixel shows the empty scene in at least half of
    the samples.

    The same frames are used to decide once, for the whole clip, whether
    it is night footage, so CLAHE doesn't switch on and off between frames
    whose brightness hovers around NIGHT_BRIGHTNESS.

    Returns:
        (background as float32, use_clahe), or (None, None) if no frames
        could be read. Then extract_stickers() falls back to the old
        behaviour (first frame as background, CLAHE decided per frame).
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #Evenly spaced frame numbers from the first to the last frame
    wanted = set(np.linspace(0, max(total - 1, 0), samples).astype(int).tolist())

    #Reads through the clip in order instead of jumping to each frame
    #(cap.set): on the 4K test clip jumping took 23-44 s, reading in order
    #21 s, and it doesn't depend on jumping landing on the exact frame.
    #grab() moves to the next frame; retrieve() is only called on the ones
    #we keep, which skips converting all the others
    grays = []
    index = 0
    while cap.grab():
        if index in wanted:
            ok, frame = cap.retrieve()
            if ok:
                #CLAHE is decided below, once all the samples are in
                grays.append(preprocess_frame(frame, use_clahe=False))
        index += 1
    cap.release()

    if not grays:
        return None, None

    #Night clip if the typical (median) sample is dark
    use_clahe = bool(np.median([np.mean(g) for g in grays]) < NIGHT_BRIGHTNESS)
    if use_clahe:
        grays = [night_contrast(g) for g in grays]

    #Median of each pixel across the samples (axis 0 = the sample number)
    background = np.median(np.stack(grays), axis=0).astype(np.float32)
    return background, use_clahe

def create_motion_mask(current_gray, prev_gray):
    """
    Create a binary mask showing areas that differ between two greyscale
    images.

    The current frame is compared with a reference image using absolute
    difference. The reference can be the previous frame (frame differencing)
    or the running-average background (see background_motion_mask). Pixels
    with a large enough change are turned white (255), while unchanged
    pixels remain black (0).
    """

    #Calculates the absolute difference between the current frame and the reference
    frame_difference = cv2.absdiff(prev_gray, current_gray)

    #Applies a binary threshold to the frame difference to create a mask of moving areas
    _, motion_mask = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)

    return motion_mask


def update_background(gray, background, alpha=BACKGROUND_ALPHA, motion_mask=None):
    """
    Update the running-average background model with a new frame.

    New background = (1 - alpha) * old background + alpha * current frame

    This should be called on EVERY frame (not just the frames we make
    stickers from) so the background stays an accurate picture of the scene.

    Selective update: if the frame's motion mask is given, only the pixels
    where nothing is moving are updated at the normal rate. Moving pixels
    are updated MOVING_ALPHA_FACTOR times slower. Without this, a person
    who stands still is slowly blended into the background, so they fade
    out of the mask, and when they walk off, the copy of them left in the
    background shows up as a trail. Selective update: Koller et al. (1994),
    as described in Piccardi (2004), Section 2.1.

    Parameters:
        gray:        the current frame after preprocess_frame()
        background:  the current background model (float32), or None on
                     the first frame
        alpha:       learning rate, how quickly the background adapts
        motion_mask: the raw mask from background_motion_mask() for this
                     frame, or None to update every pixel the same (the
                     old "blind" update)

    Returns:
        The updated background model (float32).
    """
    #On the first frame there is no background yet, so the frame itself
    #becomes the starting background. accumulateWeighted needs a float
    #image to store the decimal values of the average, hence float32
    if background is None:
        return gray.astype(np.float32)

    if motion_mask is None:
        #Blind update: blends the whole frame into the background (in place)
        cv2.accumulateWeighted(gray, background, alpha)
        return background

    #The threshold misses the soft, blurred edges of a moving object, so the
    #mask is grown a little to keep those edges out of the background too.
    #The raw mask is used rather than clean_mask(), which keeps only the
    #largest shape and would let a second, smaller person fade out
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    moving = cv2.dilate(motion_mask, grow)

    #accumulateWeighted only changes the pixels where the mask is non-zero,
    #so it is run twice: full rate on still pixels, slow rate on moving ones
    cv2.accumulateWeighted(gray, background, alpha, mask=cv2.bitwise_not(moving))
    cv2.accumulateWeighted(gray, background, alpha * MOVING_ALPHA_FACTOR, mask=moving)
    return background


def background_motion_mask(gray, background):
    """
    Create a motion mask by comparing a frame with the background model.

    The background is stored as float32, so it is converted back to a
    normal 8-bit image first, then the existing create_motion_mask() does
    the difference and threshold.
    """
    #Converts the float background back to 0-255 whole numbers
    background_8bit = cv2.convertScaleAbs(background)
    return create_motion_mask(gray, background_8bit)



def clean_mask(mask, min_area=500):
    """
    Step 3 (Pati): clean the motion mask into one solid shape.

    The raw motion mask has two problems: tiny white specks from camera
    noise, and a hollow, broken outline around the moving object (frame
    differencing only picks up the edges that moved). This removes the
    specks, joins the outline up and fills it in.

    Parameters:
        mask:     black/white motion mask from create_motion_mask()
        min_area: smallest shape (in pixels) that counts as a real object

    Returns:
        A filled black/white mask the same size as the input, or None
        if there is no usable object.
    """
    #Opening = erode then dilate. A small kernel deletes specks of noise
    #but leaves the bigger object shape alone
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small)

    #Closing = dilate then erode. A big kernel bridges the gaps in the
    #broken outline so it becomes one connected shape
    big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, big)

    #Finds the outer outline of every white region
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    #Keeps only the largest region, the moving object
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    #Draws that region filled in (thickness=-1) onto a blank mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=-1)
    return filled

def enhance_object(frame):
    """
    Step 4 (Pati): make the object clearer before it is cut out.

    CCTV footage is often dark, low-contrast and a bit soft. This boosts
    the contrast with CLAHE and then sharpens the edges with an unsharp
    mask, so the sticker looks crisp instead of murky.

    Parameters:
        frame: the original colour frame (BGR)

    Returns:
        The enhanced colour frame, same size as the input.
    """
    #Converts to LAB so brightness (L) is separate from colour (A and B).
    #CLAHE only touches L, so the colours don't get changed
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    #CLAHE boosts contrast in small tiles, so dark areas get brighter
    #without blowing out areas that are already bright
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    contrasted = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    #Unsharp mask: blur a copy, then take the blur away from the original.
    #What's left is the edges, which get boosted (1.5 original - 0.5 blur)
    blurred = cv2.GaussianBlur(contrasted, (0, 0), 2)
    sharpened = cv2.addWeighted(contrasted, 1.5, blurred, -0.5, 0)

    return sharpened

def cut_out_sticker(frame, mask, outline=6, padding=10):
    """
    Step 5 (Chloe): turn a frame + a clean mask into a sticker.

    1. Transparency: the mask becomes the alpha channel, so the object is
       opaque (255) and the background is see-through (0).
    2. Crop: cut the image down to the object's bounding box, plus some
       padding so the outline is not clipped at the edges.
    3. Outline: dilate the mask to make it slightly bigger, then colour
       the extra ring white to create the die-cut sticker border.

    Parameters:
        frame:   original colour frame (BGR)
        mask:    single-channel mask, 255 = object, 0 = background
        outline: thickness of the white border in pixels
        padding: empty space kept around the object when cropping

    Returns:
        A BGRA image (4 channels) ready to save as a transparent PNG.
    """
    # --- 1. Transparency --------------------------------------------------
    # Split the frame into its blue, green and red channels, then merge
    # them back with the mask added as a fourth (alpha) channel.
    b, g, r = cv2.split(frame)
    sticker = cv2.merge([b, g, r, mask])

    # --- 2. Crop ----------------------------------------------------------
    # boundingRect gives the smallest upright box around the white pixels
    x, y, w, h = cv2.boundingRect(mask)
    pad = outline + padding
    frame_h, frame_w = mask.shape
    # Clamp to the frame edges so we never slice outside the image
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + w + pad, frame_w), min(y + h + pad, frame_h)
    sticker = sticker[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2]

    # --- 3. White outline -------------------------------------------------
    # Dilation grows the white region outwards by roughly `outline` pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * outline + 1, 2 * outline + 1))
    grown = cv2.dilate(cropped_mask, kernel)

    # The border ring = pixels in the grown mask but not in the original
    ring = (grown > 0) & (cropped_mask == 0)
    sticker[ring] = (255, 255, 255, 255)   # white and fully opaque

    return sticker


def make_gif(stickers, frame_ms=400):
    """
    Step 6 (Chloe): combine the stickers into one looping animated GIF,
    a "stop-motion" version of what moved.

    Every GIF frame must be the same size, but each sticker has a
    different size, so each one is centred on a transparent canvas as big
    as the largest sticker.

    Parameters:
        stickers: list of (time_in_seconds, BGRA sticker) from extract_stickers
        frame_ms: how long each sticker is shown, in milliseconds

    Returns:
        The GIF file as bytes (ready for a download button), or None.
    """
    from PIL import Image          # Pillow, the Python imaging library

    if not stickers:
        return None

    # Canvas size = the largest width and height of any sticker
    canvas_w = max(s.shape[1] for _, s in stickers)
    canvas_h = max(s.shape[0] for _, s in stickers)

    frames = []
    for _, sticker in stickers:
        # OpenCV stores colours as BGRA; Pillow expects RGBA
        rgba = cv2.cvtColor(sticker, cv2.COLOR_BGRA2RGBA)
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        img = Image.fromarray(rgba)
        offset = ((canvas_w - img.width) // 2, (canvas_h - img.height) // 2)
        canvas.paste(img, offset, img)       # third arg keeps transparency
        frames.append(canvas)

    buffer = io.BytesIO()
    frames[0].save(
        buffer, format="GIF",
        save_all=True,             # save every frame, not just the first
        append_images=frames[1:],  # the rest of the frames, in order
        duration=frame_ms,         # milliseconds per frame
        loop=0,                    # 0 = loop forever
        disposal=2,                # clear each frame before the next one,
                                   # otherwise transparent frames pile up
    )
    return buffer.getvalue()


def extract_stickers(video_path, every_n_frames=15, progress_callback=None):
    """
    Read a video and return a chronological list of (time_in_seconds, sticker).

    every_n_frames: only try to make a sticker from every Nth frame, so the
                    output isn't hundreds of near-identical stickers. This
                    does NOT affect motion detection: the background is
                    updated on every frame regardless.
    progress_callback: optional function(fraction) used by the interface
                       to update a progress bar.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25          # fall back if unknown
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    #Starting background (median of frames across the clip) and the
    #clip-wide night/CLAHE decision. If this fails, background is None and
    #the first frame becomes the background, as before
    background, use_clahe = estimate_background(video_path)

    stickers = []
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break                                   # end of video

        #Preprocess every frame once, since the background needs every frame
        gray = preprocess_frame(frame, use_clahe)

        #Step 2: the motion mask is needed on every frame now, because the
        #selective update uses it to decide which pixels to learn
        motion_mask = None
        if background is not None:
            motion_mask = background_motion_mask(gray, background)

        #Generate stickers only at the interval chosen with the slider
        if index % every_n_frames == 0:
            sticker = make_sticker(frame, motion_mask)
            if sticker is not None:
                stickers.append((index / fps, sticker))

        #Update the background on EVERY frame. This happens after the
        #sticker check so the current frame (and the object in it) isn't
        #blended into the background it is being compared against
        background = update_background(gray, background, motion_mask=motion_mask)
        index += 1
        if progress_callback:
            progress_callback(min(index / total, 1.0))

    cap.release()
    return stickers