"""
stickers.py - the image-processing side of the project.

The interface (app.py) only calls extract_stickers(). Everything that turns a
video into sticker cutouts lives here, so this file can be developed, tested
and submitted independently of the interface.

CURRENT STATE: steps 1-2 (Sayo) and 5-6 (Chloe) are in. Step 3 uses a
TEMPORARY stand-in, temp_clean_mask(), until Pati's version is merged.
Step 4 (enhancement) is not in yet.
"""

import io

import cv2
import numpy as np


def make_sticker(frame, prev_gray):
    """
    Turn one frame into a sticker (BGRA image, transparent background),
    or return None if nothing is moving.

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

    gray = preprocess_frame(frame)

    if prev_gray is None:
        return None

    motion_mask = create_motion_mask(gray, prev_gray)

    if cv2.countNonZero(motion_mask) < 500:
        # If the motion is too small, return None
        return None

    # Step 3 (TEMPORARY stand-in until Pati's clean_mask() is merged)
    clean_mask = temp_clean_mask(motion_mask)
    if clean_mask is None:
        return None

    # Step 5 (Chloe): cut the object out as a transparent sticker
    return cut_out_sticker(frame, clean_mask)

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

def temp_clean_mask(mask):
    """
    TEMPORARY - replace with Pati's clean_mask() (step 3) once it is merged.

    Frame differencing gives a hollow, broken outline of the moving object.
    This joins the outline up and keeps only the biggest moving region, so
    step 5 has a solid shape to cut out.

    Returns a filled black/white mask the same size as the input, or None
    if there is no usable object.
    """
    # Closing = dilate then erode. A big kernel bridges the gaps in the
    # outline so it becomes one connected shape.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the outer outline of every white region
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep only the largest region - small ones are usually noise
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return None

    # Draw that region filled in (thickness=-1) onto a blank mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=-1)
    return filled


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
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break                                   # end of video

        if index % every_n_frames == 0:
            sticker = make_sticker(frame, prev_gray)
            if sticker is not None:
                stickers.append((index / fps, sticker))

            #compare preprocessed previous frame VS preprocessed current frame
            prev_gray = preprocess_frame(frame)
        index += 1
        if progress_callback:
            progress_callback(min(index / total, 1.0))

    cap.release()
    return stickers