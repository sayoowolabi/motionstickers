"""
Evaluation script (Chloe): score the pipeline's masks against hand-marked
ground truth, so the team can compare versions with numbers instead of
judging stickers by eye.

For every hand-marked frame (made with mark_frames.py) it runs each motion
detection method through Pati's clean_mask() and reports:
    IoU       overlap with the hand-marked mask, 0 (none) to 1 (perfect)
    extra     share of the predicted mask that is NOT the object, 0 is best.
              This is where ghosting shows up (the red area in the images)
    time (ms) time per frame for motion mask + cleaning

Run with:
    python evaluate.py footage/donny.mov
    python evaluate.py footage/donny.mov --every-n 15 --csv results.csv
    python evaluate.py footage/donny.mov --save-images comparisons
    python evaluate.py footage/donny.mov --every-n 1 5 15 30
    python evaluate.py footage/donny.mov --alphas 0.005 0.01 0.02 0.05 0.1
    python evaluate.py footage/dusk.mp4 --clahe per-frame   # old CLAHE rule

Running-average rows (one set per alpha):
    Blind, 1st frame      the old pipeline (week 9)
    Selective, 1st frame  + selective update only
    Blind, median         + starting background from estimate_background() only
    Selective, median         both (Sayo's version)
    App: + shadows            Selective, median + Pati's shadow removal,
                              exactly what the app runs now (on night clips
                              shadow removal is skipped, like in the app)
    python evaluate.py footage/donny.mov --per-frame per_frame.csv

Ground truth is read from ground_truth/<clip name>/frame_XXXXX_mask.png.
For a test with a known answer, compare_methods.py --make-test writes
ground truth for its synthetic video automatically.

Adding a method: write a class with update(gray) -> mask (called once per
frame, in order) and add it to make_methods().
"""

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from stickers import (BACKGROUND_ALPHA, NIGHT_BRIGHTNESS, background_motion_mask,
                      clean_mask, create_motion_mask, estimate_background,
                      preprocess_frame, update_background)


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


class RunningAverage:
    """Compare each frame with the running-average background, then update
    the background with that frame. Uses the same functions as
    extract_stickers() in stickers.py, so this scores the code that
    actually runs in the app. The background is updated on every frame,
    so the result does not depend on every_n.

    start:     None = the first frame becomes the background (old), or a
               starting background from estimate_background() (new)
    selective: False = every pixel updated the same (old "blind" update),
               True = moving pixels updated more slowly (new)"""

    def __init__(self, alpha=BACKGROUND_ALPHA, start=None, selective=False):
        self.alpha = alpha
        self.selective = selective
        #copy, so methods sharing the same start don't update each other's
        self.background = None if start is None else start.copy()

    def update(self, gray):
        #Same order as extract_stickers(): compare first, then update
        mask = None
        if self.background is not None:
            mask = background_motion_mask(gray, self.background)
        self.background = update_background(gray, self.background, self.alpha,
                                            mask if self.selective else None)
        return mask


class AppPipeline(RunningAverage):
    """What the app runs now: selective update from the median start, plus
    Pati's shadow removal against a colour copy of the background (skipped
    on night clips, as in make_sticker()). Unlike the other methods it also
    needs the colour frame, so update() returns (raw mask, cleaned mask)."""

    def __init__(self, alpha, start, colour_start, night):
        super().__init__(alpha, start, selective=True)
        self.colour = colour_start.copy()
        self.night = night

    def update(self, gray, frame):
        #Same order as extract_stickers(): compare and clean first, using
        #the backgrounds from before this frame, then update both
        mask = background_motion_mask(gray, self.background)
        if self.night:
            cleaned = clean_mask(mask)
        else:
            cleaned = clean_mask(mask, frame=frame, background=self.colour)
        self.background = update_background(gray, self.background, self.alpha, mask)
        self.colour = update_background(frame, self.colour, self.alpha, mask)
        return mask, cleaned


def make_methods(every_n, alphas=(BACKGROUND_ALPHA,), start=None,
                 colour_start=None, night=False):
    """The old methods plus the current pipeline. Each change is also scored
    on its own (2 starts x 2 update rules), so its effect can be seen."""
    methods = {f"Frame differencing (N={every_n})": FrameDifferencing(every_n)}
    #Four running-average entries per learning rate, so they can be compared
    for alpha in alphas:
        methods[f"Blind, 1st frame (a={alpha})"] = RunningAverage(alpha)
        methods[f"Selective, 1st frame (a={alpha})"] = RunningAverage(alpha, None, True)
        if start is not None:
            methods[f"Blind, median (a={alpha})"] = RunningAverage(alpha, start)
            methods[f"Selective, median (a={alpha})"] = RunningAverage(alpha, start, True)
            if colour_start is not None:
                methods[f"App: + shadows (a={alpha})"] = AppPipeline(
                    alpha, start, colour_start, night)
    return methods


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


def extra_fraction(predicted, truth):
    """Share of the predicted mask that lies outside the hand-marked object,
    from 0 (all of it is the object) to 1 (none of it is). Ghost regions
    count as extra, so this measures ghosting directly. IoU on its own
    mixes up extra pixels and missed pixels."""
    if predicted is None:
        return 0.0          # nothing predicted, so nothing extra
    total = cv2.countNonZero(predicted)
    if total == 0:
        return 0.0
    outside = cv2.countNonZero(cv2.bitwise_and(predicted, cv2.bitwise_not(truth)))
    return outside / total


def load_ground_truth(video_path, gt_root):
    """Return {frame_index: mask} for one clip."""
    folder = Path(gt_root) / Path(video_path).stem
    truth = {}
    for path in sorted(folder.glob("frame_*_mask.png")):
        index = int(path.stem.split("_")[1])
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        truth[index] = (mask > 127).astype(np.uint8) * 255
    return truth


def evaluate_clip(video_path, truth, every_n, alphas, save_dir=None,
                  clahe="clip"):
    """Run every method over the clip, scoring the marked frames.

    clahe: "clip" = decide night contrast once for the clip, like the app
           does now; "per-frame" = the old rule, to measure the flicker."""
    #Starting background + clip-wide CLAHE decision, same as extract_stickers().
    #Done once per clip and shared, so it is timed separately
    start_time = time.perf_counter()
    start, use_clahe, colour_start = estimate_background(str(video_path),
                                                         return_colour=True)
    start_ms = (time.perf_counter() - start_time) * 1000
    if clahe == "per-frame":
        use_clahe = None
    report_clahe(video_path, truth, use_clahe, start_ms)

    methods = make_methods(every_n, alphas, start, colour_start, bool(use_clahe))
    per_frame = []
    scores = {name: [] for name in methods}
    extras = {name: [] for name in methods}
    times = {name: [] for name in methods}
    warmed_up = set()

    cap = cv2.VideoCapture(str(video_path))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or index > max(truth):
            break
        gray = preprocess_frame(frame, use_clahe)

        for name, method in methods.items():
            start = time.perf_counter()
            if isinstance(method, AppPipeline):
                raw, cleaned = method.update(gray, frame)
            else:
                raw = method.update(gray)
                cleaned = clean_mask(raw) if raw is not None else None
            elapsed = (time.perf_counter() - start) * 1000

            if index in truth:
                frame_iou = iou(cleaned, truth[index])
                frame_extra = extra_fraction(cleaned, truth[index])
                scores[name].append(frame_iou)
                extras[name].append(frame_extra)
                # skip the first timed run: one-off warm-up cost
                if name in warmed_up:
                    times[name].append(elapsed)
                warmed_up.add(name)
                #One row per marked frame, so a 0.0 can be traced to its frame
                per_frame.append({
                    "clip": Path(video_path).stem,
                    "every_n": every_n,
                    "method": name,
                    "frame": index,
                    "iou": round(frame_iou, 3),
                    "extra": round(frame_extra, 3),
                    "found_object": cleaned is not None,
                    "ms": round(elapsed, 2),
                })
                if save_dir is not None:
                    save_comparison(save_dir, video_path, index, name,
                                    frame, cleaned, truth[index])
        index += 1
    cap.release()
    return scores, extras, times, per_frame


def report_clahe(video_path, truth, use_clahe, start_ms):
    """CLAHE flicker check: print the clip's brightness range and how often
    the old per-frame rule (mean < NIGHT_BRIGHTNESS) would switch CLAHE on
    or off over the scored part of the clip. Many switches = flicker."""
    cap = cv2.VideoCapture(str(video_path))
    means = []
    while len(means) <= max(truth):
        ok, frame = cap.read()
        if not ok:
            break
        means.append(np.mean(preprocess_frame(frame, use_clahe=False)))
    cap.release()
    night = [m < NIGHT_BRIGHTNESS for m in means]
    switches = sum(a != b for a, b in zip(night, night[1:]))
    decision = {True: "on", False: "off", None: "per frame"}[use_clahe]
    print(f"{Path(video_path).stem}: brightness {min(means):.0f}-{max(means):.0f} "
          f"(mean {np.mean(means):.0f}), per-frame rule would switch CLAHE "
          f"{switches} times; CLAHE used: {decision}; "
          f"starting background took {start_ms:.0f} ms")


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
    parser.add_argument("--every-n", type=int, nargs="+", default=[15],
                        help="frame interval(s) to test, e.g. 1 5 15 30")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[BACKGROUND_ALPHA],
                        help="running-average learning rate(s) to test")
    parser.add_argument("--clahe", choices=["clip", "per-frame"], default="clip",
                        help="decide night contrast once per clip (the app) "
                             "or per frame (the old rule, to measure flicker)")
    parser.add_argument("--gt", default="ground_truth")
    parser.add_argument("--csv", help="also save results to this CSV file")
    parser.add_argument("--save-images", metavar="FOLDER",
                        help="save colour-coded comparison images here")
    parser.add_argument("--per-frame", metavar="CSV",
                        help="also save every marked frame's score to this CSV")
    args = parser.parse_args()

    rows = []
    frame_rows = []
    for video in args.videos:
        truth = load_ground_truth(video, args.gt)
        if not truth:
            print(f"No ground truth for {video} - run mark_frames.py first.")
            continue
        #Run the whole clip once per frame interval. The running-average
        #rows should come out the same for every N; frame differencing won't
        for every_n in args.every_n:
            scores, extras, times, per_frame = evaluate_clip(
                video, truth, every_n, args.alphas, args.save_images,
                args.clahe)
            frame_rows.extend(per_frame)
            for name in scores:
                rows.append({
                    "clip": Path(video).stem,
                    "every_n": every_n,
                    "method": name,
                    "frames": len(scores[name]),
                    "mean_iou": round(float(np.mean(scores[name])), 3),
                    "min_iou": round(float(np.min(scores[name])), 3),
                    "mean_extra": round(float(np.mean(extras[name])), 3),
                    "ms_per_frame": round(float(np.mean(times[name])), 2) if times[name] else None,
                })

    print(f"\n{'Clip':<14}{'N':>4}  {'Method':<32}{'Frames':>7}{'Mean IoU':>10}"
          f"{'Min IoU':>9}{'Extra':>8}{'ms/frame':>10}")
    for r in rows:
        print(f"{r['clip']:<14}{r['every_n']:>4}  {r['method']:<32}{r['frames']:>7}"
              f"{r['mean_iou']:>10}{r['min_iou']:>9}{r['mean_extra']:>8}"
              f"{str(r['ms_per_frame']):>10}")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.csv}")

    if args.per_frame and frame_rows:
        #Sorted by clip, frame number, then method, so both methods for
        #the same frame sit next to each other
        frame_rows.sort(key=lambda r: (r["clip"], r["every_n"], r["frame"], r["method"]))
        with open(args.per_frame, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=frame_rows[0].keys())
            writer.writeheader()
            writer.writerows(frame_rows)

        #Point out the worst frames so they're easy to find
        worst = sorted(frame_rows, key=lambda r: r["iou"])[:3]
        print(f"\nSaved {args.per_frame}. Lowest-scoring frames:")
        for r in worst:
            print(f"  {r['clip']} frame {r['frame']:>5}  {r['method']:<32} IoU {r['iou']}")


if __name__ == "__main__":
    main()