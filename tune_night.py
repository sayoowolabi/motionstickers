"""
tune_night.py (Pati): help pick the night brightness threshold and gamma.

Prints the average brightness of frames from each clip, so you can see
where day footage ends and night footage starts, and saves a side-by-side
image of one frame with different gamma values to compare by eye.

Run with:
    python tune_night.py footage/day.mov footage/night.mov
    python tune_night.py footage/night.mov --gammas 0.4 0.5 0.6 0.8 --save gamma_test.png

Then change NIGHT_BRIGHTNESS and NIGHT_GAMMA at the top of stickers.py.
"""

import argparse

import cv2
import numpy as np

from stickers import NIGHT_BRIGHTNESS, brightness, gamma_correct


def clip_brightness(path, every_n=15):
    """Return (list of brightness values, a middle frame) for one clip."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open {path}")
    values, frames, index = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % every_n == 0:
            values.append(brightness(frame))
            frames.append(frame)
        index += 1
    cap.release()
    return values, frames[len(frames) // 2] if frames else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--every-n", type=int, default=15)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    parser.add_argument("--save", default="gamma_test.png",
                        help="where to save the gamma comparison (last clip)")
    args = parser.parse_args()

    print(f"Current NIGHT_BRIGHTNESS = {NIGHT_BRIGHTNESS}\n")
    last_frame = None
    for path in args.videos:
        values, middle = clip_brightness(path, args.every_n)
        if not values:
            print(f"{path}: no frames read")
            continue
        night = sum(v < NIGHT_BRIGHTNESS for v in values)
        print(f"{path}")
        print(f"  brightness  min {min(values):.0f}  mean {np.mean(values):.0f}"
              f"  max {max(values):.0f}")
        print(f"  {night}/{len(values)} sampled frames count as night\n")
        last_frame = middle

    if last_frame is not None:
        gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        panels = [gray] + [gamma_correct(gray, g) for g in args.gammas]
        labels = ["original"] + [f"gamma {g}" for g in args.gammas]
        for panel, label in zip(panels, labels):
            cv2.putText(panel, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, 255, 2)
        cv2.imwrite(args.save, np.hstack(panels))
        print(f"Saved gamma comparison to {args.save}")


if __name__ == "__main__":
    main()