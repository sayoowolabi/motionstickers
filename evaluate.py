"""
Evaluation script (Chloe): score the pipeline's masks against hand-marked
ground truth, so the team can compare versions with numbers instead of
judging stickers by eye.

For every hand-marked frame (made with mark_frames.py) it runs each motion
detection method through Pati's clean_mask() and reports:
    IoU       overlap with the hand-marked mask, 0 (none) to 1 (perfect)
    time (ms) time per frame for motion mask + cleaning

Run with:
    python evaluate.py footage/donny.mov
    python evaluate.py footage/donny.mov --every-n 15 --csv results.csv
    python evaluate.py footage/donny.mov --save-images comparisons

Ground truth is read from ground_truth/<clip name>/frame_XXXXX_mask.png.

Adding a method: write a class with update(gray) -> mask (called once per
frame, in order) and add it to make_methods(). When Sayo's running-average
background is merged, add it here and compare it with the stand-in.
"""

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from stickers import clean_mask, create_motion_mask, preprocess_frame


# ---------------------------------------------------------------- methods

class FrameDifferencing:
    """The current pipeline: compare each frame with the frame exactly
    N frames earlier, the same gap extract_stickers() uses."""

    def __init__(self, every_n):
        self.every_n = every_n
        self.history = deque(maxlen=every_n)   # the last N greyscale frames

    def update(self, gray):
        mask = None
        if len(self.history) == self.every_n:
            # history[0] is the frame from exactly N frames ago
            mask = create_motion_mask(gray, self.history[0])
        self.history.append(gray)
        return mask


class RunningAverageStandIn:
    """TEMPORARY stand-in for Sayo's step 2 fix, so the evaluation can be
    built and tested now. Compares each frame with a slowly updated average
    of the scene (cv2.accumulateWeighted), updated every frame. Replace with
    Sayo's version once merged."""

    def __init__(self, rate=0.005, threshold=25):
        self.rate = rate
        self.threshold = threshold
        self.background = None

    def update(self, gray):
        if self.background is None:
            self.background = gray.astype(np.float32)
            return None
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        cv2.accumulateWeighted(gray, self.background, self.rate)
        return mask


def make_methods(every_n):
    return {
        f"Frame differencing (every {every_n})": FrameDifferencing(every_n),
        "Running average (stand-in)": RunningAverageStandIn(),
    }


# ---------------------------------------------------------------- scoring

def iou(predicted, truth):
    """Intersection over union of two black/white masks."""
    if predicted is None:
        predicted = np.zeros_like(truth)
    inter = cv2.countNonZero(cv2.bitwise_and(predicted, truth))
    union = cv2.countNonZero(cv2.bitwise_or(predicted, truth))
    if union == 0:
        return 1.0          # both empty: nothing moving, and nothing found
    return inter / union


def load_ground_truth(video_path, gt_root):
    """Return {frame_index: mask} for one clip."""
    folder = Path(gt_root) / Path(video_path).stem
    truth = {}
    for path in sorted(folder.glob("frame_*_mask.png")):
        index = int(path.stem.split("_")[1])
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        truth[index] = (mask > 127).astype(np.uint8) * 255
    return truth


def evaluate_clip(video_path, truth, every_n, save_dir=None):
    """Run every method over the clip, scoring the marked frames."""
    methods = make_methods(every_n)
    scores = {name: [] for name in methods}
    times = {name: [] for name in methods}
    warmed_up = set()

    cap = cv2.VideoCapture(str(video_path))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or index > max(truth):
            break
        gray = preprocess_frame(frame)

        for name, method in methods.items():
            start = time.perf_counter()
            raw = method.update(gray)
            cleaned = clean_mask(raw) if raw is not None else None
            elapsed = (time.perf_counter() - start) * 1000

            if index in truth:
                scores[name].append(iou(cleaned, truth[index]))
                # skip the first timed run: one-off warm-up cost
                if name in warmed_up:
                    times[name].append(elapsed)
                warmed_up.add(name)
                if save_dir is not None:
                    save_comparison(save_dir, video_path, index, name,
                                    frame, cleaned, truth[index])
        index += 1
    cap.release()
    return scores, times


def save_comparison(save_dir, video_path, index, name, frame, predicted, truth):
    """Save an image: green = correct, red = extra (e.g. ghost),
    blue = missed. Useful for the blog and report."""
    if predicted is None:
        predicted = np.zeros_like(truth)
    view = frame.copy()
    overlay = view.copy()
    overlay[(predicted > 0) & (truth > 0)] = (0, 200, 0)
    overlay[(predicted > 0) & (truth == 0)] = (0, 0, 255)
    overlay[(predicted == 0) & (truth > 0)] = (255, 0, 0)
    view = cv2.addWeighted(overlay, 0.5, view, 0.5, 0)
    safe = "".join(c if c.isalnum() else "_" for c in name)
    out = Path(save_dir) / Path(video_path).stem
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"frame_{index:05d}_{safe}.png"), view)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--every-n", type=int, default=15,
                        help="frame interval used by the current pipeline")
    parser.add_argument("--gt", default="ground_truth")
    parser.add_argument("--csv", help="also save results to this CSV file")
    parser.add_argument("--save-images", metavar="FOLDER",
                        help="save colour-coded comparison images here")
    args = parser.parse_args()

    rows = []
    for video in args.videos:
        truth = load_ground_truth(video, args.gt)
        if not truth:
            print(f"No ground truth for {video} - run mark_frames.py first.")
            continue
        scores, times = evaluate_clip(video, truth, args.every_n, args.save_images)
        for name in scores:
            rows.append({
                "clip": Path(video).stem,
                "method": name,
                "frames": len(scores[name]),
                "mean_iou": round(float(np.mean(scores[name])), 3),
                "min_iou": round(float(np.min(scores[name])), 3),
                "ms_per_frame": round(float(np.mean(times[name])), 2) if times[name] else None,
            })

    print(f"\n{'Clip':<14}{'Method':<34}{'Frames':>7}{'Mean IoU':>10}"
          f"{'Min IoU':>9}{'ms/frame':>10}")
    for r in rows:
        print(f"{r['clip']:<14}{r['method']:<34}{r['frames']:>7}"
              f"{r['mean_iou']:>10}{r['min_iou']:>9}{str(r['ms_per_frame']):>10}")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.csv}")


if __name__ == "__main__":
    main()