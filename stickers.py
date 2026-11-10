"""
stickers.py - the image-processing side of the project.

The interface (app.py) only calls extract_stickers(). Everything that turns a
video into sticker cutouts lives here, so this file can be developed, tested
and submitted independently of the interface.

CURRENT STATE: steps 1-2 (Sayo), 3-4 (Pati) and 5-6 (Chloe) are in.
"""

import io

import cv2
import numpy as np


# --- Settings to tune on our footage -------------------------------------
# Night handling (step 1)
NIGHT_BRIGHTNESS = 70   # average brightness (0-255) below this = night frame
NIGHT_GAMMA = 0.6       # gamma < 1 brightens dark areas more than light ones

# Shadow removal (step 3), HSV test
# A pixel counts as shadow if it got darker by a "shadow-like" amount
# (between SHADOW_DARKEST and SHADOW_LIGHTEST of its old brightness)
# while keeping roughly the same colour (hue and saturation).
SHADOW_DARKEST = 0.4    # darker than 40% of before = a real object, not shadow
SHADOW_LIGHTEST = 0.93  # barely darker (93%+) = just noise
SHADOW_MAX_SAT_CHANGE = 40   # 0-255 scale
SHADOW_MAX_HUE_CHANGE = 10   # 0-179 scale (OpenCV hue)


def make_sticker(frame, prev_gray, prev_frame=None, night=None):
    """
    Turn one frame into a sticker (BGRA image, transparent background),
    or return None if nothing is moving.

    prev_frame: the previous colour frame, used to remove shadows (step 3)
    night:      True/False to force night mode, None to check this frame

    TODO - our pipeline, built up from the research:
      1. Preprocess: greyscale, blur, night-frame check,
         CLAHE on night frames.
      2. Motion mask: compare with the previous frame or a
         running-average background, then threshold.
      3. Clean the mask: morphological opening/closing,
         keep the largest contour, fill it in.
      4. Enhance the object: contrast (CLAHE) and sharpening (unsharp mask).
      5. Cut it out: use the mask as the alpha channel, crop to the
         bounding box, add a white outline by dilating the mask.
    """

    if night is None:
        night = is_night(frame)

    gray = preprocess_frame(frame, night)

    if prev_gray is None:
        return None

    motion_mask = create_motion_mask(gray, prev_gray)

    if cv2.countNonZero(motion_mask) < 500:
        # If the motion is too small, return None
        return None

    # Step 3 (Pati): remove shadows, then clean the mask into one solid shape.
    # Shadow removal is skipped at night: dark/infrared frames have almost
    # no colour, so the hue and saturation checks would just be noise.
    if prev_frame is not None and not night:
        cleaned = clean_mask(motion_mask, frame=frame, prev_frame=prev_frame)
    else:
        cleaned = clean_mask(motion_mask)
    if cleaned is None:
        return None

    # Step 4 (Pati): make the object clearer before cutting it out
    enhanced = enhance_object(frame)

    # Step 5 (Chloe): cut the object out as a transparent sticker
    return cut_out_sticker(enhanced, cleaned)


def brightness(frame):
    """Average brightness of a frame (0-255), from the V channel of HSV."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def is_night(frame, threshold=None):
    """True if the frame is dark enough to count as night footage."""
    if threshold is None:
        threshold = NIGHT_BRIGHTNESS
    return brightness(frame) < threshold


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


def preprocess_frame(frame, night=None):
    """
    Preprocess a video frame before motion detection.

    Converts the frame to greyscale and applies Gaussian blur to reduce
    noise. The average brightness is then checked to determine whether
    the frame is dark. If it is a night frame, gamma correction brightens
    it and CLAHE improves the contrast, so objects are easier to
    distinguish from the background.

    Parameters:
        frame: The original colour video frame.
        night: True/False to force night mode (so a whole clip is treated
               the same way), or None to check this frame on its own.

    Returns:
        The preprocessed greyscale frame.
    """
    #Converts each video frame from colour to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Applies a small Gaussian blur to remove tiny pixel-level changes
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #Checks the average brightness of the frame
    if night is None:
        night = is_night(frame)

    if night:
        #brightens the dark areas first, so CLAHE has detail to work with
        gray = gamma_correct(gray)
        #increases the contrast of dark footage to better distinguish between objects
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return gray

def create_motion_mask(current_gray, prev_gray):
    """
    Create a binary mask showing areas that changed between two frames.

    The current frame is compared with the previous frame using absolute
    difference. Pixels with a large enough change are turned white (255),
    while unchanged pixels remain black (0).
    """

    #Calculates the absolute difference between the current and previous frames
    frame_difference = cv2.absdiff(prev_gray, current_gray)

    #Applies a binary threshold to the frame difference to create a mask of moving areas
    _, motion_mask = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)

    return motion_mask



def shadow_mask(frame, prev_frame):
    """
    Step 3 (Pati): find pixels that changed only because of a shadow.

    Uses the HSV test from Cucchiara et al. (2003). A shadow makes a
    surface darker but doesn't change its colour much, so in HSV the
    brightness (V) drops by a limited amount while hue (H) and
    saturation (S) stay about the same. A real object covering the
    surface usually changes the colour as well.

    It checks both ways round, because with frame differencing a shadow
    shows up where it arrives (now darker) and where it left (now lighter).

    Parameters:
        frame:      the current colour frame (BGR)
        prev_frame: the colour frame it is being compared with (BGR)

    Returns:
        A black/white mask, 255 where the change looks like a shadow.
    """
    #A light blur so single noisy pixels don't decide the result
    cur = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    old = cv2.cvtColor(cv2.GaussianBlur(prev_frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    cur = cur.astype("float32")
    old = old.astype("float32")

    #Hue is an angle (0-179 in OpenCV), so 175 and 2 are only 7 apart
    hue_diff = np.abs(cur[:, :, 0] - old[:, :, 0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    same_hue = hue_diff <= SHADOW_MAX_HUE_CHANGE
    same_sat = np.abs(cur[:, :, 1] - old[:, :, 1]) <= SHADOW_MAX_SAT_CHANGE
    same_colour = same_hue & same_sat

    def darkened(dark, light):
        #Brightness ratio, +1 so we never divide by zero
        ratio = (dark[:, :, 2] + 1) / (light[:, :, 2] + 1)
        return (ratio >= SHADOW_DARKEST) & (ratio <= SHADOW_LIGHTEST)

    #Shadow arrived (current darker) or shadow left (previous darker)
    shadow = same_colour & (darkened(cur, old) | darkened(old, cur))
    return shadow.astype("uint8") * 255


def clean_mask(mask, min_area=500, frame=None, prev_frame=None):
    """
    Step 3 (Pati): clean the motion mask into one solid shape.

    The raw motion mask has three problems: shadows that move with the
    object, tiny white specks from camera noise, and a hollow, broken
    outline around the moving object (frame differencing only picks up
    the edges that moved). This takes the shadows out, removes the specks,
    joins the outline up and fills it in.

    Parameters:
        mask:       black/white motion mask from create_motion_mask()
        min_area:   smallest shape (in pixels) that counts as a real object
        frame:      optional current colour frame, for shadow removal
        prev_frame: optional previous colour frame, for shadow removal
                    (if either is missing, shadows are left in)

    Returns:
        A filled black/white mask the same size as the input, or None
        if there is no usable object.
    """
    #Removes shadow pixels from the mask before any morphology, so the
    #closing step can't join a shadow onto the object
    if frame is not None and prev_frame is not None:
        shadows = shadow_mask(frame, prev_frame)
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

    every_n_frames: only process every Nth frame, so the output isn't
                    hundreds of near-identical stickers.
    progress_callback: optional function(fraction) used by the interface
                       to update a progress bar.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25          # fall back if unknown
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    stickers = []
    prev_gray = None
    prev_frame = None
    night = None
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break                                   # end of video

        if index % every_n_frames == 0:
            #Decides day/night once, from the first frame, and keeps it for
            #the whole clip. Deciding per frame could flip a frame near the
            #threshold, and then the whole image would count as "motion"
            if night is None:
                night = is_night(frame)

            sticker = make_sticker(frame, prev_gray, prev_frame, night)
            if sticker is not None:
                stickers.append((index / fps, sticker))

            #compare preprocessed previous frame VS preprocessed current frame
            prev_gray = preprocess_frame(frame, night)
            prev_frame = frame
        index += 1
        if progress_callback:
            progress_callback(min(index / total, 1.0))

    cap.release()
    return stickers