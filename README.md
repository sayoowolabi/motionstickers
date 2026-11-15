# [House Alarm!] — Motion Sticker Extraction

An image-processing project for IDSP 1701 (Image Processing) at TU Dublin.

Upload fixed-camera footage, such as indoor CCTV, and the system detects
whatever is moving and cuts it out as a "sticker", like the iPhone sticker
feature, producing a timeline of cutouts in the order they appeared.

The project uses classical image-processing techniques only (no machine learning).

## Team
- Patrycja Palka
- Folasayo Owolabi
- Chloe Vegiga


## How it's supposed to work
1. **Preprocessing:** greyscale, blur, contrast enhancement for night footage
2. **Motion detection:** compare each frame with a running-average
   background (`cv2.accumulateWeighted`, updated every frame) to find what changed.
   The background starts as the median of frames across the clip and is only updated at full speed where nothing is moving (selective update)
3. **Mask cleanup:** thresholding and morphology
4. **Cutout:** keep the moving object, make the background transparent,
   add a white sticker outline (scaled to the object's size, so it shows on 4K footage too)
5. **Stickers:** skip near-duplicate stickers (an object standing still), then
   download them as a zip or a stop-motion GIF

## Setup
Requires Python 3.14.

```
git clone [repo URL]
cd housealarm
py -3.14 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running
```
streamlit run app.py
```
Upload a video in the browser window that opens, then click **Make stickers**.

Settings in the sidebar:
- **Check every Nth frame** (default 15): how often a sticker is made. Lower
  gives more stickers. Movement is still tracked on every frame.
- **Skip near-duplicates** (default 0.70): a sticker is skipped if its shape
  overlaps the last kept sticker by more than this. Lower skips more; 1.0 keeps
  every sticker.

## Test footage
Videos are not stored in this repo because of their size. Upload them into a `footage/` folder in the
project root.

## How to evaluate
`evaluate.py` scores the pipeline against frames marked by hand, so changes can
be compared with numbers instead of judging stickers by eye.

### 1. Mark the frames (ground truth)
```
python mark_frames.py footage/burglary.mp4 --count 10
```
This picks `--count` frames spread through the clip (skipping the first 50) and
opens each one in a window. Click around the object's outline, then:

| Key | Action |
| --- | --- |
| left click | add a point to the outline |
| `z` | undo the last point |
| `r` | start this frame again |
| space | start another shape (a second person or animal) |
| enter | save this frame |
| `s` | skip this frame |
| `q` | quit |

Masks are saved to `ground_truth/<clip name>/`. Large videos (such as 4K) are
shown shrunk to fit the screen, but the masks are saved at full size.

**Marking rule (use it for every clip so scores can be compared):** trace every
person or animal in view, moving or still. Don't trace shadows or light (such as
a torch beam). Skip only frames where no person or animal is in view. For clips
that are empty for long stretches, use `--count 20`.

### 2. Score the pipeline
```
python evaluate.py footage/burglary.mp4 footage/dogwalk.mp4 --csv results.csv --per-frame per_frame.csv --save-images comparisons
```
Several clips can be scored at once. Each method is run over the whole clip and
scored on the marked frames:

| Column | Meaning |
| --- | --- |
| Mean IoU | overlap with the hand-marked mask: 1 = perfect, 0 = no overlap |
| Min IoU | the worst marked frame |
| Extra | share of the predicted mask that is not the object (ghosts, shadows): lower is better |
| ms/frame | time per frame for detection and cleanup |

The running-average rows separate Sayo's changes: *Blind, 1st frame* is the
old pipeline, and *Selective, median* is what the app runs now.

| Option | What it does |
| --- | --- |
| `--csv FILE` | save the results table |
| `--per-frame FILE` | save every marked frame's score and print the 3 worst frames |
| `--save-images FOLDER` | save colour-coded images: green = correct, red = extra, blue = missed |
| `--alphas 0.005 0.01 0.02` | compare background learning rates |
| `--every-n 5 15 30` | compare frame intervals |
| `--clahe per-frame` | use the old per-frame night rule, to check for flicker |
| `--gt FOLDER` | read ground truth from another folder |

### 3. Keep each run
Give each run its own names, so a later run doesn't overwrite an earlier one,
e.g. `--csv results_final.csv --per-frame per_frame_final.csv --save-images comparisons_final`.
All `results*.csv`, `per_frame*.csv` and `comparisons*/` files are ignored by git.

## Project structure
- `app.py`: Streamlit interface (upload, display, download)
- `stickers.py`: the image-processing pipeline
- `requirements.txt`: libraries to install
- `evaluate.py`: scores motion-detection methods against ground truth (IoU, extra area, timing)
- `mark_frames.py`: hand-mark ground-truth masks for a clip
- `compare_methods.py`: side-by-side images of alternative methods; `--make-test`
  makes a synthetic clip with exact ground truth

## Status
- [x] Project setup and interface starter
- [x] Motion detection (running-average background)
- [x] Mask cleanup and cutouts
- [ ] Night-time footage handling
- [x] Sticker outline scaling and de-duplication
- [x] Evaluation tools, and six real clips hand-marked and scored (before/after selective background update)
- [ ] Final evaluation after the shadow and night changes

## References
- Streamlit 
- OpenCV