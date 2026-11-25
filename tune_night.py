"""
tune_night.py (Pati): A tool to tune the night parameters in stickers.py
(NIGHT_BRIGHTNESS and NIGHT_GAMMA)

For each clip this program:
  1. Prints out the mean, minimum and maximum brightness of the sampled
     frames. It also reports if the clip will be labelled DAY or NIGHT,
     as in the estimate_background() function in stickers.py. The
     program samples an even number of BACKGROUND_SAMPLES frames across the
     length of the clip and then uses the median brightness of these
     frames against the value of NIGHT_BRIGHTNESS to make this decision.
     Testing this program on both a day and a night clip should give an
     appropriate threshold.

  2. Generates an image from the middle of both the day and night clips
     using different values of gamma. The top row is only using gamma
     and the bottom row uses gamma + CLAHE. CLAHE is also used on night
     clips in the current processing pipeline (see night_contrast()
     in stickers.py). Pick the gamma that gives you the best visibility
     of the object in the bottom row.

Usage:
    python tune_night.py footage/day.mov footage/night.mov
    python tune_night.py footage/night.mov --gammas 0.4 0.5 0.6 0.8 --out-dir gamma_tests

Once complete update the values for NIGHT_BRIGHTNESS and NIGHT_GAMMA at
the top of stickers.py.

Rationale for this script:
    Manually selecting these two parameters in the stickers using
    the eye would be purely a matter of luck. This program measures the
     actual brightness of the test clips and allows for a side-by-side
     view of the results for each value of gamma. The parameters in
     stickers.py are selected based on our test footage.

References: Numbers in [ ] refer to the References in README.md.
  - Gamma (power-law) transformation: s = c * r ** gamma [1], [28]
  - CLAHE via cv2.createCLAHE(clipLimit, tileGridSize) [2]
  - Median brightness sampled across a clip as background [3], [5]
  - Using argparse for command-line arguments [26], [27]
"""

import argparse
import os

import cv2
import numpy as np


# Use the same functions and settings as in the pipeline, so the numbers
# reported here agree exactly with what stickers.py does with the video.
# If the pipeline changes, this script changes with it automatically

from stickers import (BACKGROUND_SAMPLES, NIGHT_BRIGHTNESS, brightness,
                      gamma_correct, preprocess_frame)


# Each box in the image comparison is at most this width. Using a 4K video
# with multiple gamma settings would otherwise create an image that is
# 20,000 px wide

PANEL_WIDTH = 640

# AI USE: AI assistance (ChatGPT) was used for this function because:
# 1. It collects three different sets of frames (report, night decision,
#    middle frame) in a single pass, using grab()/retrieve() so only the
#    frames that are needed get decoded.
# 2. It has to copy the pipeline's night decision exactly (same frames,
#    same preprocessing, same median), or it would predict the wrong result.
# Reviewed and tested by the team.

def clip_brightness(path, every_n=15):
    """
    Measure brightness of a single video in one sweep.

    Inputs:
    path:    path to the video
    every_n: every this many frames are used in min/mean/max values

    Returns a dict with:
    values:  brightness (0-255) of every Nth frame
    median:  median brightness of the BACKGROUND_SAMPLES frames used by
            the pipeline for night decision
    night:   whether the pipeline will classify this video as night
    middle:  middle frame of video (used in gamma comparison)

    Returns None if no frames were read.
    """
    
    # Open the video [21]. If it can't be opened (wrong path, unsupported
    # format etc.), stop with a clear error message, rather than silently
    # reading nothing and returning None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open {path}")

    # The number of frames is known up front, so we can work out which
    # frames we need before reading, instead of storing every sampled frame
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # The same evenly spaced frames that estimate_background() in
    # stickers.py samples for its night decision. Spreading them across the
    # whole clip, and taking the median later, means a person walking
    # through a few frames doesn't change the result [3], [5]

    decision_frames = set(np.linspace(0, max(total - 1, 0),
                                      BACKGROUND_SAMPLES).astype(int).tolist())
    middle_index = total // 2

    values, decision_values, middle = [], [], None
    index = 0

    # grab() moves to the next frame without decoding it, and retrieve()
    # only decodes the frames we actually need. Decoding is the slow part,
    # so this is much faster than calling read() on every frame. It is the
    # same technique Sayo uses in estimate_background()

    while cap.grab():
        wanted_for_report = index % every_n == 0
        wanted_for_decision = index in decision_frames
        is_middle = index == middle_index

        if wanted_for_report or wanted_for_decision or is_middle:
            ok, frame = cap.retrieve()
            if ok:
                if wanted_for_report:
                    values.append(brightness(frame))
                if wanted_for_decision:
                    # Measured exactly like the pipeline: greyscale + blur,
                    # no night enhancement yet, then the mean
                    decision_values.append(
                        float(np.mean(preprocess_frame(frame, use_clahe=False))))
                if is_middle:
                    middle = frame
        index += 1

    # Frees the video file once we're done with it
    cap.release()

    if not values:
        return None

    # Some formats report the frame count wrongly, so the decision frames
    # may never be reached; in that case the median of the report frames is
    # used instead. The median (not the mean) is used because it ignores a
    # few unusually bright or dark frames, e.g. a light being switched on
   
    median = float(np.median(decision_values)) if decision_values else float(np.median(values))
    return {
        "values": values,
        "median": median,
        "night": median < NIGHT_BRIGHTNESS,
        "middle": middle,
    }


def gamma_then_clahe(gray, gamma):

    """
    Similar to the night enhancement in the current pipeline (in stickers.py as
    night_contrast()). This version instead uses a gamma value chosen by the
    user rather than NIGHT_GAMMA. A gamma correction is applied first before a
    CLAHE is applied using the same parameters listed in [1], [2].

    CLAHE or Contrast Limited Adaptive Histogram Equalisation divides the image
    into 8x8 tiles. Within each tile the intensity values are spread out to
    enhance local detail instead of global image detail. The parameter clipLimit
    is set to 2.0 which limits the amount of contrast enhanced and prevents the
    noise in dark areas from being over enhanced. Gamma correction is applied
    first in order to provide more enhanced detail to CLAHE.
    """

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gamma_correct(gray, gamma))


def label(panel, text):

    """
    Write a label in the top-left corner of a panel [24]. The text is drawn
    twice: first thick and black (thickness 5) as an outline, then thin and
    white (thickness 2) on top, so it can be read on both light and dark
    panels.
    """

    cv2.putText(panel, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 5,
                cv2.LINE_AA)
    cv2.putText(panel, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2,
                cv2.LINE_AA)
    return panel


def gamma_comparison(frame, gammas):

    """
    Create the comparison image for a single frame (two rows of panels):
    Top row: original image, then gamma only for each value
    Bottom row: original image, then gamma + CLAHE for each value 
    (this is what is actually applied to night clips via the pipeline)
    """

    # Greyscale [22], because the motion detection in stickers.py works on
    # greyscale frames

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Shrinks big frames (keeping the aspect ratio) so the final image
    # stays a sensible size. INTER_AREA averages the pixels being merged,
    # which is the method meant for shrinking as it avoids aliasing [18], [23]

    h, w = gray.shape
    if w > PANEL_WIDTH:
        scale = PANEL_WIDTH / w
        gray = cv2.resize(gray, (PANEL_WIDTH, int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    # Gamma correction [1], [28]: s = 255 * (r / 255) ** gamma.
    # With gamma < 1, dark pixels are brightened a lot and light pixels only
    # slightly (e.g. with gamma 0.5, a pixel of 25 becomes 80, but 200
    # only becomes 226). This brings out detail in the dark areas without
    # washing out the bright ones

    top = [label(gray.copy(), "original")]
    bottom = [label(gray.copy(), "original")]
    for g in gammas:
        top.append(label(gamma_correct(gray, g), f"gamma {g}"))
        bottom.append(label(gamma_then_clahe(gray, g), f"gamma {g} + CLAHE"))

    # axis=1 joins the panels left to right into a row, then axis=0 puts
    # the two rows on top of each other [25]. This only works because all
    # the panels are the same size

    rows = [np.concatenate(top, axis=1), np.concatenate(bottom, axis=1)]
    return np.concatenate(rows, axis=0)


def main():

    # --- Command-line arguments --------------------------------------------
    # argparse [26] turns the words typed after "python tune_night.py" into
    # variables, and builds a --help message from the docstring above


    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # nargs="+" = one or more video paths, collected into a list [27]
    parser.add_argument("videos", nargs="+",
                        help="one or more video clips to measure")
    parser.add_argument("--every-n", type=int, default=15,
                        help="measure one frame out of every N (default 15)")
    # Also nargs="+", so several gamma values can be given at once
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.4, 0.6, 0.8],
                        help="gamma values to compare (below 1 = brighter)")
    parser.add_argument("--out-dir", default=".",
                        help="folder for the gamma comparison images "
                             "(one per clip, named gamma_<clip>.png)")
    args = parser.parse_args()

    # Creates the output folder if it doesn't exist yet
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Current NIGHT_BRIGHTNESS = {NIGHT_BRIGHTNESS}\n")
    for path in args.videos:

        # --- 1. Brightness report ------------------------------------------

        result = clip_brightness(path, args.every_n)
        if result is None:
            print(f"{path}: no frames read\n")
            continue

        values = result["values"]

        # Counts the sampled frames darker than the threshold
        # (True counts as 1 and False as 0 when summed)

        dark = sum(v < NIGHT_BRIGHTNESS for v in values)

        print(f"{path}")
        print(f"  brightness  min {min(values):.0f}  mean {np.mean(values):.0f}"
              f"  max {max(values):.0f}")
        print(f"  {dark}/{len(values)} sampled frames are below the threshold")
        print(f"  clip median {result['median']:.0f} -> pipeline will treat "
              f"this clip as {'NIGHT' if result['night'] else 'DAY'}")

        # --- 2. Gamma comparison image -------------------------------------

        if result["middle"] is not None:
            # Names the image after the clip, e.g. gamma_dogwalk.png
            name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(args.out_dir, f"gamma_{name}.png")
            cv2.imwrite(out_path, gamma_comparison(result["middle"], args.gammas))
            print(f"  saved gamma comparison to {out_path}")
        print()


# Only runs when the file is run directly ("python tune_night.py ..."),
# not when another file imports it
if __name__ == "__main__":
    main()