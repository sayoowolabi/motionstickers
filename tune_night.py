"""
tune_night.py (Pati): a helper script for choosing the night settings in
stickers.py (NIGHT_BRIGHTNESS and NIGHT_GAMMA).

What it does, for each clip:
  1. Reports on brightness by printing the average, darkest and brightest
     brightness of the sampled frames. It also reports whether the video
     will be categorised as DAY or NIGHT, exactly as in the function
     estimate_background() in stickers.py: it evenly samples
     BACKGROUND_SAMPLES frames across the entire clip and compares their
     median brightness to NIGHT_BRIGHTNESS. Testing this script on both a
     night and a day video helps establish a reasonable cut-off.

  2. Compares gamma by saving one frame from the middle of each video as an image. 
     The top image uses gamma alone while the bottom uses gamma + CLAHE (how night frames 
     are processed in the script see night_contrast() in stickers.py). 
     Select the gamma that best makes the subject most clearly visible in the lower image.

Run with:
    python tune_night.py footage/day.mov footage/night.mov
    python tune_night.py footage/night.mov --gammas 0.4 0.5 0.6 0.8 --out-dir gamma_tests

After running, edit both NIGHT_BRIGHTNESS and NIGHT_GAMMA at the top of stickers.py.

References (GeeksforGeeks tutorials this script is based on):
  - Reading a video frame by frame with cv2.VideoCapture:
    https://www.geeksforgeeks.org/python/python-program-extract-frames-using-opencv/
  - Gamma (power-law) transformation, s = c * r ** gamma:
    https://www.geeksforgeeks.org/python/python-intensity-transformation-operations-on-images/
  - CLAHE with cv2.createCLAHE(clipLimit, tileGridSize):
    https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/
  - Converting a frame to greyscale with cv2.cvtColor:
    https://www.geeksforgeeks.org/python/python-opencv-cv2-cvtcolor-method/
  - Shrinking an image with cv2.resize and INTER_AREA:
    https://www.geeksforgeeks.org/python/image-resizing-using-opencv-python/
  - Writing labels on an image with cv2.putText:
    https://www.geeksforgeeks.org/python/python-opencv-cv2-puttext-method/
  - Joining images side by side / on top of each other (np.concatenate):
    https://www.geeksforgeeks.org/python/how-to-display-multiple-images-in-one-window-using-opencv-python/
  - Command-line arguments with argparse, and nargs for lists:
    https://www.geeksforgeeks.org/python/command-line-option-and-argument-parsing-using-argparse-in-python/
    https://www.geeksforgeeks.org/python/how-to-pass-a-list-as-a-command-line-argument-with-argparse/
"""

import argparse
import os

import cv2
import numpy as np


# Use the same functions and settings as in the pipeline so the output
# numbers here agree with what stickers.py does with the video

from stickers import (BACKGROUND_SAMPLES, NIGHT_BRIGHTNESS, brightness,
                      gamma_correct, preprocess_frame)


# Each box in the image comparison is at most this width. Using a 4K video
# with multiple gamma settings would otherwise create an image that is
# 20,000 px wide

PANEL_WIDTH = 640


def clip_brightness(path, every_n=15):
    """
    Measure the brightness of one video clip in a single pass.

    Parameters:
        path:    path to the video file
        every_n: measure one frame out of every this many for the
                 min/mean/max report

    Returns:
        A dictionary with:
          values:  brightness (0-255) of every Nth frame
          median:  median brightness of the BACKGROUND_SAMPLES frames the
                   pipeline uses for its night decision
          night:   True if the pipeline will treat this clip as night
          middle:  the middle frame of the clip (for the gamma comparison)
        or None if no frames could be read.
    """
    
    # Open the video.
    # If the video could not be opened (wrong path, wrong format, etc.)
    # we exit with a clear error message rather than silently reading nothing and returning None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open {path}")

    # The number of frames is known up front, so we can work out which
    # frames we need before reading, instead of storing every sampled frame
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # The same evenly spaced frames that estimate_background() in
    # stickers.py samples for its night decision
    decision_frames = set(np.linspace(0, max(total - 1, 0),
                                      BACKGROUND_SAMPLES).astype(int).tolist())
    middle_index = total // 2

    values, decision_values, middle = [], [], None
    index = 0

    # grab() gets the next image without decoding it, while retrieve() only decodes when we need it.
    # This is the same technique that Sayo uses in estimate_background()
    # This is also much faster than using read() on every frame

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
    # may never be reached; then the median of the report frames is used
    median = float(np.median(decision_values)) if decision_values else float(np.median(values))
    return {
        "values": values,
        "median": median,
        "night": median < NIGHT_BRIGHTNESS,
        "middle": middle,
    }


def gamma_then_clahe(gray, gamma):
    """
    Like the night enhancement that is used in the pipeline (in
    stickers.py as night_contrast()), but using a user-chosen gamma
    instead of NIGHT_GAMMA. First apply gamma correction, then apply
    CLAHE using the same settings.
    """

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gamma_correct(gray, gamma))


def label(panel, text):
    """
    Render text in the top-left corner of a panel with a thick black outline first,
    then white text to ensure it is visible on both light and dark panels.
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

    # Greyscale, because the motion detection in stickers.py works on
    # greyscale frames

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Shrinks big frames (keeping the aspect ratio) so the final image
    # stays a sensible size. INTER_AREA is the method meant for shrinking

    h, w = gray.shape
    if w > PANEL_WIDTH:
        scale = PANEL_WIDTH / w
        gray = cv2.resize(gray, (PANEL_WIDTH, int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    # Using gamma < 1 increases the dark pixels significantly and the
    # light ones only slightly (s = 255 * (r / 255) ** gamma). This brings
    # out details from the dark pixels without over saturating the light ones

    top = [label(gray.copy(), "original")]
    bottom = [label(gray.copy(), "original")]
    for g in gammas:
        top.append(label(gamma_correct(gray, g), f"gamma {g}"))
        bottom.append(label(gamma_then_clahe(gray, g), f"gamma {g} + CLAHE"))

    # axis=1 joins the panels left to right into a row, then axis=0 puts
    # the two rows on top of each other (all panels are the same size)

    rows = [np.concatenate(top, axis=1), np.concatenate(bottom, axis=1)]
    return np.concatenate(rows, axis=0)


def main():
    # --- Command-line arguments --------------------------------------------
    # argparse turns the words typed after "python tune_night.py" into
    # variables, and builds a --help message from the docstring above

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # nargs="+" = one or more video paths, collected into a list
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