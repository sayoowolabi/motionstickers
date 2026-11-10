"""
stickers.py - the image-processing side of the project.

The interface (app.py) only calls extract_stickers(). Everything that turns a
video into sticker cutouts lives here, so this file can be developed, tested
and submitted independently of the interface.

CURRENT STATE: steps 1-2 (Sayo), 3-4 (Pati) and 5-6 (Chloe) are in.
Step 2 now uses a running-average background instead of comparing
with the previous frame, to reduce ghosting.

KNOWN LIMITATIONS (not handled yet, to be looked at next):
  - If something is already in the very first frame, it is baked into the
    starting background and leaves a "ghost" until the background catches
    up (about 1 / BACKGROUND_ALPHA frames). A warm-up period before making
    stickers could fix this, but it would miss motion at the very start
    of short clips.
  - preprocess_frame() only applies CLAHE when a frame's mean brightness is
    below 70. If a clip hovers around 70, CLAHE switches on and off between
    frames, and the whole frame can look different from the averaged
    background. Needs checking on night footage.
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


def make_sticker(frame, gray, background):
    """
    Turn one frame into a sticker (BGRA image, transparent background),
    or return None if nothing is moving.

    Parameters:
        frame:      the original colour frame (BGR)
        gray:       the same frame after preprocess_frame()
        background: the running-average background from update_background()
                    (float32), or None if there isn't one yet

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

    # Step 1 (preprocessing) is now done once per frame in
    # extract_stickers(), so the frame isn't preprocessed twice.
    if background is None:
        return None

    # Step 2 (Sayo): compare the frame with the background model
    motion_mask = background_motion_mask(gray, background)

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

def preprocess_frame(frame):
    """
    Preprocess a video frame before motion detection.

    Converts the frame to greyscale and applies Gaussian blur to reduce
    noise. The average brightness is then checked to determine whether
    the frame is dark. If it is a night frame, CLAHE is applied to
    improve the contrast and make objects easier to distinguish from
    the background.

    Parameters:
        frame: The original colour video frame.

    Returns:
        The preprocessed greyscale frame.
    """
    #Converts each video frame from colour to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Applies a small Gaussian blur to remove tiny pixel-level changes
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #Calculates the average brightness of the frame.
    if np.mean(gray) < 70:
        #increases the contrast of dark footage to better distinguish between objects
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return gray

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


def update_background(gray, background, alpha=BACKGROUND_ALPHA):
    """
    Update the running-average background model with a new frame.

    New background = (1 - alpha) * old background + alpha * current frame

    This should be called on EVERY frame (not just the frames we make
    stickers from) so the background stays an accurate picture of the scene.

    Parameters:
        gray:       the current frame after preprocess_frame()
        background: the current background model (float32), or None on
                    the first frame
        alpha:      learning rate, how quickly the background adapts

    Returns:
        The updated background model (float32).
    """
    #On the first frame there is no background yet, so the frame itself
    #becomes the starting background. accumulateWeighted needs a float
    #image to store the decimal values of the average, hence float32
    if background is None:
        return gray.astype(np.float32)

    #Blends the current frame into the background (updates it in place)
    cv2.accumulateWeighted(gray, background, alpha)
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

    stickers = []
    background = None
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break                                   # end of video

        #Preprocess every frame once, since the background needs every frame
        gray = preprocess_frame(frame)

        #Generate stickers only at the interval chosen with the slider
        if index % every_n_frames == 0:
            sticker = make_sticker(frame, gray, background)
            if sticker is not None:
                stickers.append((index / fps, sticker))

        #Update the background on EVERY frame. This happens after the
        #sticker check so the current frame (and the object in it) isn't
        #blended into the background it is being compared against
        background = update_background(gray, background)
        index += 1
        if progress_callback:
            progress_callback(min(index / total, 1.0))

    cap.release()
    return stickers