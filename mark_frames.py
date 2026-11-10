"""
Ground-truth marking tool for evaluate.py (Chloe).

Saves a few frames from a clip, then lets you trace the moving object in
each one by clicking around its outline. Each traced frame is saved as a
black/white mask (white = object) - the "perfect answer" that the
pipeline's masks are scored against.

Run with:
    python mark_frames.py footage/donny.mov --count 10

Controls in the window:
    left click  add a point on the object's outline
    z           undo the last point
    r           reset this frame
    space       add another shape (e.g. a second object or a gap in the legs)
    enter       save this frame's mask and go to the next one
    s           skip this frame (nothing moving in it)
    q           quit

Files are saved to ground_truth/<clip name>/:
    frame_00045.png       the original frame
    frame_00045_mask.png  your hand-marked mask
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def pick_frame_indexes(total, count, skip_start):
    """Spread `count` frames evenly through the clip, skipping the first few
    (the running-average background needs time to settle)."""
    start = min(skip_start, total - 1)
    return np.linspace(start, total - 1, count).astype(int).tolist()


def mark_one_frame(frame, window):
    """Let the user trace polygons on one frame. Returns the mask, or None
    if skipped, or the string "quit"."""
    shapes = [[]]          # list of polygons, each a list of (x, y) points

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            shapes[-1].append((x, y))

    cv2.setMouseCallback(window, on_click)

    while True:
        view = frame.copy()
        for poly in shapes:
            if len(poly) > 1:
                cv2.polylines(view, [np.array(poly)], len(poly) > 2,
                              (0, 255, 0), 1)
            for p in poly:
                cv2.circle(view, p, 2, (0, 0, 255), -1)
        cv2.imshow(window, view)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("z") and shapes[-1]:
            shapes[-1].pop()
        elif key == ord("r"):
            shapes[:] = [[]]
        elif key == ord(" ") and len(shapes[-1]) > 2:
            shapes.append([])
        elif key == ord("s"):
            return None
        elif key == ord("q"):
            return "quit"
        elif key in (13, 10):                       # enter
            mask = np.zeros(frame.shape[:2], np.uint8)
            for poly in shapes:
                if len(poly) > 2:
                    cv2.fillPoly(mask, [np.array(poly)], 255)
            return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--count", type=int, default=10,
                        help="how many frames to mark (default 10)")
    parser.add_argument("--skip-start", type=int, default=50,
                        help="ignore the first N frames (default 50)")
    parser.add_argument("--out", default="ground_truth")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir = Path(args.out) / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    window = "Trace the moving object - enter=save, s=skip, q=quit"
    cv2.namedWindow(window)

    # Read frames in order rather than jumping with cap.set(), because
    # jumping isn't frame-accurate for some video formats, and the frame
    # numbers must match the ones evaluate.py reads.
    wanted = set(pick_frame_indexes(total, args.count, args.skip_start))
    index = -1
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index not in wanted:
            continue
        wanted.discard(index)
        mask = mark_one_frame(frame, window)
        if isinstance(mask, str):
            break
        if mask is None:
            continue
        cv2.imwrite(str(out_dir / f"frame_{index:05d}.png"), frame)
        cv2.imwrite(str(out_dir / f"frame_{index:05d}_mask.png"), mask)
        print(f"saved frame {index}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()