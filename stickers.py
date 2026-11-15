"""
stickers.py - the image-processing side of the project.

The interface (app.py) only calls extract_stickers(). Everything that turns a
video into sticker cutouts lives here, so this file can be developed, tested
and submitted independently of the interface.

CURRENT STATE: steps 1-2 (Sayo), 3-4 (Pati) and 5-6 (Chloe) are in.
Step 2 now uses a running-average background instead of comparing
with the previous frame, to reduce ghosting.
The sticker outline and padding (step 5) scale with the object's size.
Near-duplicate stickers (an object standing still) are skipped.

Week 10 changes (Sayo, steps 1-2):
  - Selective background update: the background only learns at full speed
    where nothing is moving, so people who stand still no longer fade out
    and don't leave a trail when they move on.
  - The starting background is the median of frames spread across the whole
    clip, not the first frame, so someone already in frame 1 is no longer
    baked into the background as a ghost.
  - CLAHE (night contrast) is switched on or off once per clip instead of
    per frame, so clips that hover around the brightness limit don't flicker.

Week 10 changes (Pati, steps 1 and 3):
  - Night clips get gamma correction before CLAHE (NIGHT_GAMMA), so the
    dark areas have detail for CLAHE to work with. Uses Sayo's clip-wide
    night decision, so it doesn't flicker either.
  - Shadow removal: an HSV test (Cucchiara et al., 2003) against a colour
    copy of the background removes pixels that only got darker without
    changing colour. Skipped on night clips.

Week 11 changes (Chloe, from the final evaluation):
  - Shadow removal runs on footage that has colour, instead of on footage
    that isn't dark (has_colour()). Infrared clips are often bright enough
    not to count as night but have no colour, so the HSV test was cutting
    pieces out of people there.
  - SHADOW_DARKEST 0.4 -> 0.7, so dark clothing no longer counts as shadow.
  - extract_stickers() takes the app's "Night mode" and "Shadow removal"
    settings (None = automatic).
  - clean_mask() drops shapes smaller than RELATIVE_MIN_SIZE of the biggest
    one, so leftover blobs don't appear as floating pieces in stickers.
  - drop_ghosts(): shapes that don't change between frames are dropped when
    another shape is moving (Cucchiara et al., 2003).
  - Stickers covering more than MAX_OBJECT_FRAC of the frame are skipped
    (the whole scene changed, e.g. the camera adjusting its exposure).

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

# Frames below this mean brightness count as night footage and get gamma
# correction + CLAHE (decided once per clip, see estimate_background()).
NIGHT_BRIGHTNESS = 70

# Night handling (step 1, Pati): gamma < 1 brightens dark areas more than
# light ones, so CLAHE has some detail to work with.
NIGHT_GAMMA = 0.6

# Shadow removal (step 3, Pati), HSV test against the colour background.
# A pixel counts as shadow if it is darker than the background by a
# "shadow-like" amount (between SHADOW_DARKEST and SHADOW_LIGHTEST of the
# background's brightness) while keeping roughly the same colour.
# 0.7 rather than 0.4: at 0.4, dark clothing (much darker than the floor,
# with little colour) also passed as "shadow" and was cut out of people.
# Tested on the burglary clip (Chloe): IoU 0.473 at 0.4, 0.532 at 0.7, and
# 0.7 also had less extra area than no shadow removal at all.
SHADOW_DARKEST = 0.7    # darker than 70% of the background = a real object
SHADOW_LIGHTEST = 0.93  # barely darker (93%+) = just noise
SHADOW_MAX_SAT_CHANGE = 40   # 0-255 scale
SHADOW_MAX_HUE_CHANGE = 10   # 0-179 scale (OpenCV hue)

# Shadow removal compares colour, so it only runs on footage that HAS
# colour: the background's mean HSV saturation must be at least this.
# Infrared/black-and-white CCTV has a saturation of 0 even when it is bright
# enough not to count as night (the dog walk and bunny clips), and a dark
# clip can still have colour (the indoor night clips), so "is it night?"
# was the wrong test.
COLOUR_MIN_SATURATION = 20   # 0-255 scale

# Smallest shape that counts as a real object (step 3, Pati), as a share of
# the frame's area instead of a fixed 500 px, so it means the same thing
# at any resolution: 0.0005 = 0.05% of the frame, which is about 150 px at
# 640x480, 1,000 px at 1080p and 4,100 px at 4K. A fixed 500 px was too
# big for small/distant objects on low-res CCTV (the bunny got deleted)
# and too small to filter noise blobs on 4K footage.
MIN_OBJECT_FRAC = 0.0005
MIN_OBJECT_PIXELS = 50       # never go below this, even on tiny frames
# clean_mask() keeps every shape at least this share of the biggest one, so
# a second object is kept (the dog is 49-76% of its owner's size) but small
# leftover blobs (ghost patches, glare) aren't glued into the sticker.
# Burglary clip: IoU 0.532 -> 0.547, extra 0.391 -> 0.374 (0.5 would
# sometimes drop the dog).
RELATIVE_MIN_SIZE = 0.3

# Ghost check (Cucchiara et al., 2003): a ghost is a patch where the
# background is out of date, so nothing in it changes from frame to frame;
# a real object does. Each shape is compared with the frame GHOST_GAP frames
# earlier: if under GHOST_MIN_MOVING of its pixels changed by more than
# GHOST_DIFF, it counts as still. Still shapes are dropped only when another
# shape in the same frame IS moving, so a person standing still is kept.
GHOST_GAP = 5
GHOST_DIFF = 15
GHOST_MIN_MOVING = 0.05

# A "sticker" covering more than this share of the frame is the whole scene
# changing (the camera adjusting its exposure, lights switching on), not a
# moving object, so it is skipped. On our clips real objects covered at
# most 18% of the frame; the bunny clip's first frame covered 67%.
MAX_OBJECT_FRAC = 0.5

# How many frames, spread across the clip, make the starting background.
BACKGROUND_SAMPLES = 25

# Sticker de-duplication: skip a new sticker when it overlaps the last
# kept sticker by more than this (IoU).
DUPLICATE_OVERLAP = 0.7


def drop_ghosts(cleaned, change):
    """
    Remove shapes that aren't moving (ghosts) when another shape is moving.

    cleaned: mask from clean_mask()
    change:  black/white mask of pixels that changed since GHOST_GAP frames
             ago (frame differencing), or None to skip the check
    Returns the mask with ghost shapes removed (or unchanged).
    """
    if change is None:
        return cleaned
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return cleaned

    moving, still = [], []
    for c in contours:
        shape = np.zeros_like(cleaned)
        cv2.drawContours(shape, [c], -1, 255, thickness=-1)
        area = cv2.countNonZero(shape)
        changed = cv2.countNonZero(cv2.bitwise_and(shape, change))
        (moving if changed >= GHOST_MIN_MOVING * area else still).append(c)

    #Nothing moving (someone standing still) or nothing still: keep it all
    if not moving or not still:
        return cleaned
    kept = np.zeros_like(cleaned)
    cv2.drawContours(kept, moving, -1, 255, thickness=-1)
    return kept


def make_sticker(frame, motion_mask, return_mask=False,
                 colour_background=None, shadows=True, change=None):
    """
    Turn one frame into a sticker (BGRA image, transparent background),
    or return None if nothing is moving.

    With return_mask=True it returns (sticker, cleaned_mask) instead, or
    (None, None), so extract_stickers() can compare masks between stickers.

    Parameters:
        frame:             the original colour frame (BGR)
        motion_mask:       the frame's motion mask from
                           background_motion_mask(), or None if there is
                           no background yet
        colour_background: the colour (BGR) background model, used to
                           remove shadows (step 3). None = no shadow removal
        shadows:           False to skip shadow removal, e.g. on black-and-
                           white/infrared footage, which has no colour for
                           the HSV test to use (see has_colour())

    Our pipeline, built up from the research:
      1. Preprocess: greyscale, blur, night-clip check,
         gamma correction + CLAHE on night clips.
      2. Motion mask: compare with a running-average background,
         then threshold.
      3. Clean the mask: remove shadows (HSV test against the colour
         background), morphological opening/closing, keep every shape
         above the minimum size, fill them in.
      4. Enhance the object: contrast (CLAHE) and sharpening (unsharp mask).
      5. Cut it out: use the mask as the alpha channel, crop to the
         bounding box, add a white outline by dilating the mask.
    """

       # Steps 1 and 2 (preprocessing and the motion mask) are done once per
    # frame in extract_stickers(), because the selective background update
    # needs the motion mask on every frame, not just sticker frames.
    if motion_mask is None:
        return (None, None) if return_mask else None

    nothing = (None, None) if return_mask else None

    # If the motion is too small, return None. The minimum scales with the
    # frame size (see min_object_area()) instead of a fixed 500 px
    if cv2.countNonZero(motion_mask) < min_object_area(motion_mask):
        return nothing

    # Step 3 (Pati): remove shadows, then clean the mask into solid shapes.
    # Shadow removal is skipped on footage with no colour (infrared/black
    # and white), where the hue and saturation checks would just be noise.
    if colour_background is not None and shadows:
        cleaned = clean_mask(motion_mask, frame=frame,
                             background=colour_background)
    else:
        cleaned = clean_mask(motion_mask)
    if cleaned is None:
        return nothing

    # Ghost check: drop shapes that aren't moving if something else is
    cleaned = drop_ghosts(cleaned, change)

    # Too big to be one object: the whole scene changed, so no sticker
    if cv2.countNonZero(cleaned) > MAX_OBJECT_FRAC * cleaned.size:
        return nothing

    # Step 4 (Pati): make the object clearer before cutting it out
    enhanced = enhance_object(frame)

    # Step 5 (Chloe): cut the object out as a transparent sticker
    sticker = cut_out_sticker(enhanced, cleaned)
    return (sticker, cleaned) if return_mask else sticker


def brightness(frame):
    """
    Average brightness of a colour frame (0-255), measured on the
    greyscale image, the same way the night check in the pipeline does it,
    so tune_night.py prints numbers you can compare with NIGHT_BRIGHTNESS.
    """
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


def gamma_correct(gray, gamma=None):
    """
    Brighten a dark image with gamma correction.

    Each pixel becomes 255 * (pixel / 255) ** gamma. With gamma < 1 the
    dark values are lifted a lot and the bright values only a little, so
    detail comes out of the shadows without blowing out lights. A lookup
    table makes it fast (256 values worked out once, not per pixel).
    """
    if gamma is None:
        gamma = NIGHT_GAMMA
    table = (((np.arange(256) / 255.0) ** gamma) * 255).astype("uint8")
    return cv2.LUT(gray, table)


def preprocess_frame(frame, use_clahe=None):
    """
    Preprocess a video frame before motion detection.

    Converts the frame to greyscale and applies Gaussian blur to reduce
    noise. If it is night footage, gamma correction brightens the dark
    areas and CLAHE improves the contrast (see night_contrast()), so
    objects are easier to distinguish from the background.

    Parameters:
        frame:     The original colour video frame.
        use_clahe: True/False = the night decision already made for the
                   whole clip (see estimate_background()). None = decide
                   from this frame's own brightness (the old per-frame
                   rule, which can flicker on clips near NIGHT_BRIGHTNESS).

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
        #brightens dark footage, then increases the contrast to better
        #distinguish between objects
        gray = night_contrast(gray)

    return gray


def night_contrast(gray):
    """
    Night enhancement on a greyscale frame: gamma correction first, so the
    dark areas have some detail, then CLAHE to boost the local contrast.
    Used by both preprocess_frame() and estimate_background(), so the
    frames and the background are always enhanced the same way.
    """
    gray = gamma_correct(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def estimate_background(video_path, samples=BACKGROUND_SAMPLES,
                        return_colour=False, use_clahe=None):
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

    With return_colour=True it also builds a colour (BGR) version of the
    background from the same frames, for shadow removal (step 3), and
    returns three values instead of two. Off by default so existing
    callers (e.g. the evaluation scripts) keep working unchanged.

    Returns:
        (background as float32, use_clahe), or (None, None) if no frames
        could be read. Then extract_stickers() falls back to the old
        behaviour (first frame as background, CLAHE decided per frame).
        With return_colour=True: (background, use_clahe, colour_background).
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
    colours = []
    index = 0
    while cap.grab():
        if index in wanted:
            ok, frame = cap.retrieve()
            if ok:
                #CLAHE is decided below, once all the samples are in
                grays.append(preprocess_frame(frame, use_clahe=False))
                if return_colour:
                    colours.append(frame)
        index += 1
    cap.release()

    if not grays:
        return (None, None, None) if return_colour else (None, None)

    #Night clip if the typical (median) sample is dark, unless the caller
    #(e.g. the app's "Night mode" setting) has already decided
    if use_clahe is None:
        use_clahe = bool(np.median([np.mean(g) for g in grays]) < NIGHT_BRIGHTNESS)
    if use_clahe:
        grays = [night_contrast(g) for g in grays]

    #Median of each pixel across the samples (axis 0 = the sample number)
    background = np.median(np.stack(grays), axis=0).astype(np.float32)
    if not return_colour:
        return background, use_clahe

    #Same median, per colour channel, for the shadow test
    colour_background = np.median(np.stack(colours), axis=0).astype(np.float32)
    return background, use_clahe, colour_background

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
    Update background model for current frame

    Arguments:
    gray: current frame after preprocess_frame()
    background: current background model (float32). None on first frame.
    alpha: learning rate (how quickly background “learns”)
    motion_mask: raw mask returned from background_motion_mask() for this
    frame. None updates every pixel equally (old “blind” update)

    Returns:
    Updated background model (float32)

    New background = (1 - alpha) * old background + alpha * current frame

    This function should be called on EVERY frame (not just to create stickers)
    to ensure the model accurately represents the scene.

    Selective update: If motion mask is provided, only non-moving pixels are updated
    at normal speed while moving pixels are updated MOVING_ALPHA_FACTOR times
    more slowly. Otherwise, someone who stays in one spot would gradually become
    part of the background, causing their mask to change to show they are no longer
    in the frame. When they move away, because there will still be a copy of them in
    the background, it will appear to follow them (create a trail). Selective update
    based on Koller et al. (1994), outlined in Piccardi (2004), Section 2.1.
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
    #The raw mask is used rather than clean_mask(), which drops small
    #shapes and shadows, so those pixels would be learned at the full rate
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



def has_colour(colour_background):
    """
    True if the footage has enough colour for shadow removal: the colour
    background's mean HSV saturation is at least COLOUR_MIN_SATURATION.
    Black-and-white and infrared footage has a saturation of about 0.
    """
    if colour_background is None:
        return False
    hsv = cv2.cvtColor(cv2.convertScaleAbs(colour_background), cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) >= COLOUR_MIN_SATURATION


def shadow_mask(frame, background):
    """
    Step 3 (Pati): find pixels that changed only because of a shadow.

    Uses the HSV test from Cucchiara et al. (2003). A shadow makes a
    surface darker but doesn't change its colour much, so in HSV the
    brightness (V) drops by a limited amount while hue (H) and
    saturation (S) stay about the same. A real object covering the
    surface usually changes the colour as well.

    The frame is compared with the colour background model (what the
    surface looks like with nothing on it), as in Cucchiara et al. With a
    background there is only one direction to check: a shadow makes the
    frame darker than the background. (The earlier version compared with
    the previous frame, which only caught the edges of a shadow, so it
    had to check both darker and lighter.)

    Parameters:
        frame:      the current colour frame (BGR)
        background: the colour background model (BGR, uint8 or float32)

    Returns:
        A black/white mask, 255 where the change looks like a shadow.
    """
    #The background model is float32, so it is turned back into 0-255
    background = cv2.convertScaleAbs(background)

    #A light blur so single noisy pixels don't decide the result
    cur = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    old = cv2.cvtColor(cv2.GaussianBlur(background, (5, 5), 0), cv2.COLOR_BGR2HSV)
    cur = cur.astype("float32")
    old = old.astype("float32")

    #Hue is an angle (0-179 in OpenCV), so 175 and 2 are only 7 apart
    hue_diff = np.abs(cur[:, :, 0] - old[:, :, 0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    same_hue = hue_diff <= SHADOW_MAX_HUE_CHANGE
    same_sat = np.abs(cur[:, :, 1] - old[:, :, 1]) <= SHADOW_MAX_SAT_CHANGE
    same_colour = same_hue & same_sat

    #Brightness ratio frame / background, +1 so we never divide by zero.
    #A shadow is darker than the background, but not too much darker
    ratio = (cur[:, :, 2] + 1) / (old[:, :, 2] + 1)
    darkened = (ratio >= SHADOW_DARKEST) & (ratio <= SHADOW_LIGHTEST)

    shadow = same_colour & darkened
    return shadow.astype("uint8") * 255


def min_object_area(mask, frac=None):
    """
    Smallest shape (in pixels) that counts as a real object for a frame
    of this size: MIN_OBJECT_FRAC of the frame's area, but at least
    MIN_OBJECT_PIXELS. Scales with resolution, like the sticker outlines.
    """
    if frac is None:
        frac = MIN_OBJECT_FRAC
    h, w = mask.shape[:2]
    return max(MIN_OBJECT_PIXELS, round(h * w * frac))


def clean_mask(mask, min_area=None, frame=None, background=None):
    """
    Step 3 (Pati): clean the motion mask into solid shapes.

    The raw motion mask has three problems: shadows that move with the
    object, tiny white specks from camera noise, and a hollow, broken
    outline around the moving object (frame differencing only picks up
    the edges that moved). This takes the shadows out, removes the specks,
    joins the outline up and fills it in.

    Every shape above the minimum size is kept, not just the largest, so
    two things moving at once (a person and their dog) both stay in. They
    end up in the same sticker, since a sticker is cut from the whole mask.

    Parameters:
        mask:       black/white motion mask from create_motion_mask()
        min_area:   smallest shape (in pixels) that counts as a real
                    object. None = scale it with the frame size
                    (see min_object_area())
        frame:      optional current colour frame, for shadow removal
        background: optional colour background model, for shadow removal
                    (if either is missing, shadows are left in)

    Returns:
        A filled black/white mask the same size as the input, or None
        if there is no usable object.
    """
    #Removes shadow pixels from the mask before any morphology, so the
    #closing step can't join a shadow onto the object
    if frame is not None and background is not None:
        shadows = shadow_mask(frame, background)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(shadows))

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

    #Keeps every region big enough to be a real object, not just the
    #largest one, so a second moving thing (the dog) isn't dropped
    if min_area is None:
        min_area = min_object_area(mask)
    objects = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not objects:
        return None

    #Keeps only shapes that are a reasonable size next to the biggest one,
    #so a second object (the dog next to its owner) is kept but small
    #leftover blobs (a ghost patch, window glare) don't end up in the sticker
    largest = max(cv2.contourArea(c) for c in objects)
    objects = [c for c in objects if cv2.contourArea(c) >= RELATIVE_MIN_SIZE * largest]

    #Draws those regions filled in (thickness=-1) onto a blank mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, objects, -1, 255, thickness=-1)
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

def outline_size(mask, outline_frac=0.04, padding_frac=0.03,
                 min_outline=2, min_padding=2):
    """
    Work out the outline and padding (in pixels) for one sticker, scaled to
    the size of the object rather than fixed.

    A fixed 6-pixel outline looks thick on a small, low-resolution object
    but almost disappears on a 4K one, because the app shrinks big stickers
    to fit its sticker sheet (columns about 200 px wide). Scaling both with
    the object's longest side keeps stickers looking the same whatever the
    video resolution.

    Parameters:
        mask:         single-channel mask, 255 = object, 0 = background
        outline_frac: outline thickness as a share of the object's longest
                      side (0.04 = 4%)
        padding_frac: extra empty space around the outline, same idea
        min_outline, min_padding: smallest values in pixels, so tiny
                      objects still get a visible border

    Returns:
        (outline, padding) as whole numbers of pixels.
    """
    _, _, w, h = cv2.boundingRect(mask)
    longest = max(w, h)
    outline = max(min_outline, round(longest * outline_frac))
    padding = max(min_padding, round(longest * padding_frac))
    return outline, padding

def cut_out_sticker(frame, mask, outline=None, padding=None):
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
        outline: thickness of the white border in pixels. Left as None,
                 it is scaled to the object's size (see outline_size)
        padding: empty space kept around the object when cropping. Left
                 as None, it is scaled the same way
    Returns:
        A BGRA image (4 channels) ready to save as a transparent PNG.
    """

    # --- 0. Outline and padding sizes -------------------------------------
    # Scaled to the object unless fixed values were passed in
    auto_outline, auto_padding = outline_size(mask)
    if outline is None:
        outline = auto_outline
    if padding is None:
        padding = auto_padding

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

def mask_overlap(mask_a, mask_b):
    """
    How much two masks overlap, as intersection over union (IoU):
    the area both cover, divided by the area either covers.
    1 = the same shape in the same place, 0 = no overlap at all.

    Used to spot near-duplicate stickers: if an object hasn't moved, its
    mask barely changes from one sticker to the next.
    """
    inter = cv2.countNonZero(cv2.bitwise_and(mask_a, mask_b))
    union = cv2.countNonZero(cv2.bitwise_or(mask_a, mask_b))
    if union == 0:
        return 0.0
    return inter / union


def extract_stickers(video_path, every_n_frames=15, progress_callback=None,
                     duplicate_overlap=DUPLICATE_OVERLAP, night=None,
                     shadows=None):
    """
    Read a video and return a chronological list of (time_in_seconds, sticker).

    every_n_frames: only try to make a sticker from every Nth frame, so the
                    output isn't hundreds of near-identical stickers. This
                    does NOT affect motion detection: the background is
                    updated on every frame regardless.
    progress_callback: optional function(fraction) used by the interface
                       to update a progress bar.
    duplicate_overlap: skip a new sticker if its mask overlaps the last
                       KEPT sticker's mask by more than this (IoU, 0 to 1),
                       so an object standing still doesn't give dozens of
                       near-identical stickers. 1.0 keeps every sticker.
    night:   True/False = the user's "Night mode" choice (gamma + CLAHE on
             or off). None = decide from the clip's brightness.
    shadows: True/False = the user's "Shadow removal" choice. None = on if
             the footage has colour (see has_colour()).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # fall back if unknown
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    #Starting background (median of frames across the clip) and the
    #clip-wide night/CLAHE decision. If this fails, background is None and
    #the first frame becomes the background, as before
    background, use_clahe, colour_background = estimate_background(
        video_path, return_colour=True, use_clahe=night)

    #Shadow removal: the user's choice, or on if the footage has colour
    if shadows is None:
        shadows = has_colour(colour_background)

    stickers = []
    last_kept_mask = None  # mask of the last sticker we kept
    index = 0
    recent = []                # the last GHOST_GAP greyscale frames

    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of video

        #Preprocess every frame once, since the background needs every frame
        gray = preprocess_frame(frame, use_clahe)

        #Step 2: the motion mask is needed on every frame now, because the
        #selective update uses it to decide which pixels to learn
        motion_mask = None
        if background is not None:
            motion_mask = background_motion_mask(gray, background)

        # Generate stickers only at the interval chosen with the slider
        if index % every_n_frames == 0:
            #Pixels that changed since GHOST_GAP frames ago, for the ghost check
            change = None
            if len(recent) == GHOST_GAP:
                diff = cv2.absdiff(gray, recent[0])
                _, change = cv2.threshold(diff, GHOST_DIFF, 255, cv2.THRESH_BINARY)
            sticker, mask = make_sticker(frame, motion_mask, return_mask=True,
                                         colour_background=colour_background,
                                         shadows=shadows, change=change)
            if sticker is not None:
                # Skip near-duplicates: compare with the last sticker we KEPT,
                # not the last one made, so a slow drift still adds up to a
                # new sticker eventually
                duplicate = (last_kept_mask is not None and
                             mask_overlap(mask, last_kept_mask) > duplicate_overlap)
                if not duplicate:
                    stickers.append((index / fps, sticker))
                    last_kept_mask = mask

        #Update the background on EVERY frame. This happens after the
        #sticker check so the current frame (and the object in it) isn't
        #blended into the background it is being compared against
        background = update_background(gray, background, motion_mask=motion_mask)

        #The colour background (for shadow removal) learns the same way,
        #with the same selective update, so the two models stay in step
        colour_background = update_background(frame, colour_background,
                                              motion_mask=motion_mask)
        recent.append(gray)
        if len(recent) > GHOST_GAP:
            recent.pop(0)
        index += 1
        if progress_callback:
            progress_callback(min(index / total, 1.0))

    cap.release()
    return stickers