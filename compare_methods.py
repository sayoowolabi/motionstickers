"""
compare_methods.py - side-by-side comparison of alternative methods for
the blog's comparison section. Not part of the app.

Run it from the project folder (next to stickers.py):

    python compare_methods.py --make-test            # make a controlled test video
    python compare_methods.py footage/test.mp4       # compare methods on a video
    python compare_methods.py footage/day_cat_01.mp4 --frame 120

It saves labelled comparison images into  output/compare/  and prints how
long each method takes, so the blog can show both what each method looks
like and what it costs.

Comparisons:
  1. Motion mask:  frame differencing (current)  vs  running-average background
  2. Mask cleanup: raw mask  vs  largest contour filled  vs  convex hull filled
  3. Cutout:       alpha channel  vs  bitwise_and  vs  feathered alpha  vs  GrabCut
  4. Outline:      dilation ring  vs  drawContours line
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from stickers import preprocess_frame, create_motion_mask, temp_clean_mask

OUT = Path("output/compare")


# ----------------------------------------------------------------- helpers --
def on_checkerboard(img):
    """Show transparency: draw a BGRA image over a grey checkerboard."""
    if img.ndim == 2:                                  # mask -> grey image
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:                              # no transparency
        return img
    h, w = img.shape[:2]
    squares = (np.indices((h, w)).sum(axis=0) // 10) % 2
    board = cv2.cvtColor((squares * 60 + 160).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    alpha = img[:, :, 3:4] / 255.0
    return (img[:, :, :3] * alpha + board * (1 - alpha)).astype(np.uint8)


def panel(images, labels, height=240):
    """Resize images to one height, add a label above each, join them."""
    tiles = []
    for img, text in zip(images, labels):
        img = on_checkerboard(img)
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, int(w * height / h)), height),
                         interpolation=cv2.INTER_NEAREST)
        bar = np.full((34, img.shape[1], 3), 35, np.uint8)
        cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        gap = np.full((height + 34, 12, 3), 255, np.uint8)
        tiles += [np.vstack([bar, img]), gap]
    return np.hstack(tiles[:-1])


def timed(fn, *args, repeats=10):
    """Run fn several times and return (result, average milliseconds)."""
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn(*args)
    return result, (time.perf_counter() - start) * 1000 / repeats


def crop_box(mask, pad=16):
    """Bounding box of the mask plus padding, clamped to the image."""
    x, y, w, h = cv2.boundingRect(mask)
    H, W = mask.shape
    return max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)


# -------------------------------------------- 1. motion mask alternatives --
def mask_frame_difference(frames, i, gap):
    """Current method: compare with the frame `gap` frames earlier."""
    return create_motion_mask(preprocess_frame(frames[i]),
                              preprocess_frame(frames[i - gap]))


def mask_running_average(frames, i, alpha=0.05):
    """Compare with a slowly-updated average of all earlier frames."""
    avg = preprocess_frame(frames[0]).astype("float32")
    for f in frames[1:i]:
        cv2.accumulateWeighted(preprocess_frame(f), avg, alpha)
    background = cv2.convertScaleAbs(avg)
    return create_motion_mask(preprocess_frame(frames[i]), background)


# ------------------------------------------- 2. mask cleanup alternatives --
def clean_convex_hull(mask):
    """Close gaps, then fill the convex hull of the largest region."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [hull], -1, 255, thickness=-1)
    return filled


# ------------------------------------------------ 3. cutout alternatives --
def cutout_alpha(frame, mask):
    """Mask becomes the alpha channel: true transparency."""
    b, g, r = cv2.split(frame)
    return cv2.merge([b, g, r, mask])


def cutout_bitwise(frame, mask):
    """Keep only the masked pixels; background turns black, not clear."""
    return cv2.bitwise_and(frame, frame, mask=mask)


def cutout_feathered(frame, mask):
    """Blur the alpha channel so the edge fades out softly."""
    b, g, r = cv2.split(frame)
    return cv2.merge([b, g, r, cv2.GaussianBlur(mask, (15, 15), 0)])


def cutout_grabcut(frame, mask):
    """
    Refine the motion mask with GrabCut, which models the colours of the
    object and background and re-draws the boundary along real edges.
    The motion mask is only a starting guess: inside = probably object,
    outside = probably background.
    """
    x1, y1, x2, y2 = crop_box(mask, pad=30)
    roi, roi_mask = frame[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    gc = np.where(roi_mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[cv2.erode(roi_mask, np.ones((15, 15), np.uint8)) > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(roi, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    refined = np.zeros_like(mask)
    refined[y1:y2, x1:x2] = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0)
    return cutout_alpha(frame, refined)


# ----------------------------------------------- 4. outline alternatives --
def outline_dilation(frame, mask, width=6):
    """Grow the mask; colour the new ring white (outside the object)."""
    sticker = cutout_alpha(frame, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    ring = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
    sticker[ring] = (255, 255, 255, 255)
    return sticker


def outline_draw_contours(frame, mask, width=6):
    """Draw a thick white line along the contour (covers the edge itself)."""
    sticker = cutout_alpha(frame, mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sticker, contours, -1, (255, 255, 255, 255), thickness=2 * width)
    return sticker


# ------------------------------------------------------- controlled test --
def make_test_video(path, frames=120, size=(480, 360)):
    """
    A 'cat' (body, head, ears, legs, tail) walks in from off-screen across
    a textured room. Because we drew it, we know exactly what the correct
    sticker should look like. The legs and tail make the shape non-convex,
    which shows the difference between contour and convex hull filling.
    """
    W, H = size
    rng = np.random.default_rng(0)
    room = cv2.GaussianBlur(rng.integers(120, 230, (H, W, 3)).astype(np.uint8), (31, 31), 0)
    cv2.rectangle(room, (0, 270), (W, H), (150, 170, 190), -1)          # floor
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    colour = (40, 60, 100)
    for i in range(frames):
        f = room.copy()
        x = -120 + i * 5                                                # starts off-screen
        cv2.ellipse(f, (x, 230), (55, 28), 0, 0, 360, colour, -1)       # body
        cv2.circle(f, (x + 58, 198), 22, colour, -1)                    # head
        for ex in (x + 46, x + 66):                                     # ears
            cv2.fillConvexPoly(f, np.array([[ex - 8, 185], [ex + 8, 185], [ex, 168]]), colour)
        step = 6 if (i // 4) % 2 else -6                                # walking legs
        for lx, s in ((x - 35, step), (x - 15, -step), (x + 15, step), (x + 35, -step)):
            cv2.line(f, (lx, 245), (lx + s, 272), colour, 9)
        cv2.ellipse(f, (x - 65, 205), (25, 10), -40, 0, 360, colour, -1)  # tail
        out.write(f)
    out.release()
    print(f"Test video saved to {path}")


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="video to test on")
    ap.add_argument("--frame", type=int, default=None, help="frame number to compare")
    ap.add_argument("--gap", type=int, default=15, help="frames between compared frames")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="running-average learning rate (try 0.005-0.2)")
    ap.add_argument("--make-test", action="store_true", help="create a test video")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.make_test:
        make_test_video(OUT.parent / "test.mp4")
        args.video = args.video or str(OUT.parent / "test.mp4")
    if not args.video:
        ap.error("give a video path, or use --make-test")

    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) <= args.gap:
        raise SystemExit("Video is too short.")
    i = args.frame if args.frame is not None else int(len(frames) * 0.6)
    frame = frames[i]
    times = []

    # 1. Motion mask
    diff, t1 = timed(mask_frame_difference, frames, i, args.gap, repeats=3)
    avg, t2 = timed(mask_running_average, frames, i, args.alpha, repeats=1)
    times += [("Motion: frame difference", t1), ("Motion: running average", t2)]
    cv2.imwrite(str(OUT / "1_motion.png"), panel(
        [frame, diff, avg], ["Original frame", "Frame difference", f"Running average ({args.alpha})"]))

    # 2. Mask cleanup (from the current frame-difference mask)
    largest, t3 = timed(temp_clean_mask, diff)
    hull, t4 = timed(clean_convex_hull, diff)
    if largest is None:
        raise SystemExit("No moving object found at this frame - try another --frame.")
    times += [("Clean: largest contour", t3), ("Clean: convex hull", t4)]
    cv2.imwrite(str(OUT / "2_cleanup.png"), panel(
        [diff, largest, hull], ["Raw mask", "Largest contour", "Convex hull"]))

    # 3. Cutout (all use the same cleaned mask, cropped to the object)
    x1, y1, x2, y2 = crop_box(largest)
    crop = lambda img: img[y1:y2, x1:x2]
    results = []
    for name, fn, reps in [("Alpha channel", cutout_alpha, 10),
                           ("bitwise_and", cutout_bitwise, 10),
                           ("Feathered alpha", cutout_feathered, 10),
                           ("GrabCut", cutout_grabcut, 1)]:
        img, t = timed(fn, frame, largest, repeats=reps)
        results.append(crop(img))
        times.append((f"Cutout: {name}", t))
    cv2.imwrite(str(OUT / "3_cutout.png"), panel(
        results, ["Alpha channel", "bitwise_and", "Feathered", "GrabCut"]))

    # 4. Outline
    dil, t5 = timed(outline_dilation, frame, largest)
    line, t6 = timed(outline_draw_contours, frame, largest)
    times += [("Outline: dilation ring", t5), ("Outline: drawContours", t6)]
    cv2.imwrite(str(OUT / "4_outline.png"), panel(
        [crop(dil), crop(line)], ["Dilation ring", "drawContours line"]))

    print(f"\nCompared frame {i} of {len(frames)}. Images saved to {OUT}/\n")
    print(f"{'Method':32} {'Time (ms)':>10}")
    for name, t in times:
        print(f"{name:32} {t:10.2f}")


if __name__ == "__main__":
    main()
