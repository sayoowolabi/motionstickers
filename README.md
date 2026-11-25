# Motion Stickers!

Motion Sticker Extraction: An image-processing project for IDSP 1701 (Image Processing) at TU Dublin.

Upload fixed-camera footage, such as indoor CCTV, and the system detects whatever is moving and cuts it out as a "sticker", like the iPhone sticker feature, producing a timeline of cutouts in the order they appeared.

The project uses classical image-processing techniques only (no machine learning).

## Team
- Patrycja Palka
- Folasayo Owolabi
- Chloe Vegiga


## How it's supposed to work
1. **Preprocessing:** greyscale and Gaussian blur. Night clips also get gamma correction and then CLAHE, to bring out detail in the dark [14, 15]. Whether a clip counts as night is decided once per video, so the enhancement doesn't flicker on and off between frames.
2. **Motion detection:** compare each frame with a running-average background (`cv2.accumulateWeighted`, updated every frame) to find what changed [3, 4]. The background starts as the median of frames spread across the clip [3, 24], and is only updated at full speed where nothing is moving (selective update [3, 23]).
3. **Mask cleanup:** remove shadows, then use morphology to refine the mask (opening to remove noise, closing to fill gaps) [5].
   - **Shadows:** pixels that have only darkened and not changed colour (HSV method, Cucchiara et al., 2003 [1]). Only applied to colour video, as infrared has no colour information.
   - **Size filters:** small shapes are rejected as noise (the minimum scales with the video size), and leftover shapes smaller than 30% of the largest object are rejected. All other objects are kept, so a dog alongside its owner is kept.
   - **Ghost check:** shapes that aren't moving are dropped if another object is moving [1].
   - **Whole-scene changes:** if the "object" covers more than 50% of the frame (such as the camera changing its own exposure), no sticker is made.
4. **Object enhancement:** boost the contrast with CLAHE (on the brightness channel only, so colours don't change) and sharpen with an unsharp mask, so stickers from murky CCTV look crisp.
5. **Cutout:** keep the moving object, make the background transparent, and add a white sticker outline (scaled to the object's size, so it shows on 4K footage too).
6. **Stickers:** skip near-duplicate stickers (an object standing still), then download them as a zip, a printable sheet or a stop-motion GIF [30].

## Setup
Requires Python 3.14.

```
git clone https://github.com/sayoowolabi/motionstickers
cd motionstickers 
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
- **Night mode** (default Auto): gamma correction and CLAHE for dark footage.
  Auto turns it on when the clip is dark; On/Off override that.
- **Shadow removal** (default Auto): removes shadows from the stickers. Auto
  turns it on for colour footage and off for black-and-white or infrared
  footage, where the colour test doesn't work.

Once the stickers are made:
- Untick any sticker you don't want (a stray blob or a bit of background).
  Only ticked stickers go into the downloads. **Tick all** / **Untick all**
  change every box at once.
- Click a sticker to see it full size.
- The timeline under the stats shows where in the video each sticker was cut.
- Download each sticker as a PNG, all of them as a zip, a printable A4 sticker
  sheet, or a stop-motion GIF (previewed beside the title).

## Test footage
Videos are not stored in this repo because of their size, and because the stock  clips are licensed previews. Put them in a `footage/` folder in the project root. The hand-marked frames for each clip go in `ground_truth/<clip name>/`, which is not in the repo either.

We evaluated on six clips, all from fixed cameras except Donny:

| Clip | What it shows |
| --- | --- |
| `donny.mov` | our own 4K footage, filmed handheld (an example of camera movement) |
| `burglary.mp4` | indoor CCTV, daytime: a person walking through a kitchen and living room [9] |
| `dogwalk.mp4` | outdoor infrared night: a person walking a dog past [10] |
| `bunny.mp4` | outdoor infrared night: a rabbit that sits still, then hops away [11] |
| `night_torch.mp4` | indoor night: an intruder with a torch going upstairs [12] |
| `night_shadows.mp4` | indoor night: an intruder casting large shadows on the floor [13] |

## How to evaluate
`evaluate.py` scores the pipeline against frames marked by hand, so changes can be compared with numbers instead of judging stickers by eye.

### 1. Mark the frames (ground truth)
```
python mark_frames.py footage/burglary.mp4 --count 10
```
This picks `--count` frames spread through the clip (skipping the first 50) and opens each one in a window. Click around the object's outline, then:

| Key | Action |
| --- | --- |
| left click | add a point to the outline |
| `z` | undo the last point |
| `r` | start this frame again |
| space | start another shape (a second person or animal) |
| enter | save this frame |
| `s` | skip this frame |
| `q` | quit |

Masks are saved to `ground_truth/<clip name>/`. Large videos (such as 4K) are shown shrunk to fit the screen, but the masks are saved at full size.

**Marking rule (use it for every clip so scores can be compared):** trace every  person or animal in view, moving or still. Don't trace shadows or light (such as a torch beam). Skip only frames where no person or animal is in view. For clips that are empty for long stretches, use `--count 20`.

### 2. Score the pipeline
```
python evaluate.py footage/burglary.mp4 footage/dogwalk.mp4 --csv results.csv --per-frame per_frame.csv --save-images comparisons
```
Several clips can be scored at once. Each method is run over the whole clip and scored on the marked frames:

| Column | Meaning |
| --- | --- |
| Mean IoU | overlap with the hand-marked mask: 1 = perfect, 0 = no overlap |
| Min IoU | the worst marked frame |
| Extra | share of the predicted mask that is not the object (ghosts, shadows): lower is better |
| ms/frame | time per frame for detection and cleanup |

IoU is the overlap measure used in the PASCAL VOC challenge [2].

The running-average rows separate each change: *Blind, 1st frame* is the old pipeline, *Selective, median* adds Sayo's selective update and median start, and *App: + shadows* is what the app runs now (it also adds shadow removal, the ghost check and the whole-scene filter).

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
Save each run in its own folder under `evaluation/`, so a later run doesn't
overwrite an earlier one. Create the folder first:
```
mkdir evaluation\final
python evaluate.py footage/burglary.mp4 --csv evaluation/final/results.csv --per-frame evaluation/final/per_frame.csv --save-images evaluation/final/comparisons
```
Our runs: `before` (old pipeline), `sayo` (selective update + median start), `final` (+ Pati's changes), `shadowfix` (shadow removal on colour footage only) and `alphas` (learning-rate test). The `evaluation/` folder is ignored by git.

## Project structure
- `app.py`: Streamlit interface (upload, display, download)
- `styles/style.css`, `.streamlit/config.toml`: the app's look
- `styles/responsive.css`: makes the page fit smaller windows and phones
- `stickers.py`: the image-processing pipeline
- `requirements.txt`: libraries to install (the versions we tested with)
- `evaluate.py`: scores motion-detection methods against ground truth (IoU, extra area, timing)
- `mark_frames.py`: hand-mark ground-truth masks for a clip
- `compare_methods.py`: side-by-side images of alternative methods, including GrabCut [31, 32]; `--make-test` makes a synthetic clip with exact ground truth
- `tune_night.py`: helps choose the night settings (`NIGHT_BRIGHTNESS`, `NIGHT_GAMMA`): reports each clip's brightness and saves a gamma comparison image

Made locally and not in the repo: `footage/` (videos), `ground_truth/` (marked frames), `output/` (from `compare_methods.py`), and `evaluation/` (results, per-frame scores and comparison images for each run).


## Status
- [x] Project setup and interface starter
- [x] Motion detection (running-average background)
- [x] Mask cleanup and cutouts
- [ ] Night-time footage handling
- [x] Sticker outline scaling and de-duplication
- [x] Evaluation tools, and six real clips hand-marked and scored (before/after selective background update)
- [ ] Final evaluation after the shadow and night changes

## References
1. Cucchiara, R., Grana, C., Piccardi, M. and Prati, A. (2003) 'Detecting moving objects, ghosts, and shadows in video streams', IEEE Transactions on Pattern Analysis and Machine Intelligence, 25(10), pp. 1337–1342.
2. Everingham, M., Van Gool, L., Williams, C.K.I., Winn, J. and Zisserman, A. (2010) 'The Pascal Visual Object Classes (VOC) Challenge', International Journal of Computer Vision, 88(2), pp. 303–338.
3. Piccardi, M. (2004) 'Background subtraction techniques: a review', 2004 IEEE International Conference on Systems, Man and Cybernetics, pp. 3099–3104.
4. OpenCV (no date) Motion Analysis and Object Tracking. Available at: https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html
5. OpenCV (no date) Morphological Transformations. Available at: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
6. OpenCV (no date) Geometric Image Transformations. Available at: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
7. Python Software Foundation (no date) time: Time access and conversions. Available at: https://docs.python.org/3/library/time.html
8. Streamlit (no date) Streamlit documentation. Available at: https://docs.streamlit.io
9. O2O Creative (no date) Security Camera Home Burglary in Progress - 4K stock video. Getty Images, no. 1084342338. Available at: https://www.gettyimages.ie/detail/video/security-camera-home-burglary-in-progress-stock-footage/1084342338
10. WW News (no date) People walking with dog in the early morning, image captured by CCTV surveillance camera. Getty Images, no. 1382583677. Available at: https://www.gettyimages.ie/detail/video/people-walking-with-dog-in-the-early-morning-image-stock-video-footage/1382583677
11. WW News (no date) CCTV Surveillance Camera Point Of View Of Rabbit Jumping Forward At Night - 4K stock video. Getty Images, no. 2044643186. Available at: https://www.gettyimages.ie/detail/video/surveillance-camera-point-of-view-of-rabbit-jumping-stock-footage/2044643186
12. Beckham, R. (no date) Security Camera View Of Hooded Intruder With A Torch Entering A Home And Walking Upstairs Backlit Night - 4K stock video. Getty Images, no. 2200417282. Available at: https://www.gettyimages.ie/detail/video/security-camera-view-of-hooded-intruder-with-a-torch-stock-footage/2200417282
13. tawattiw (no date) Burglar Stealing Valuables from Desk Inside House at Night - 4K stock video. Getty Images, no. 2298084001. Available at: https://www.gettyimages.ie/detail/video/burglar-stealing-valuables-from-desk-inside-house-at-stock-footage/2298084001
14. GeeksforGeeks (no date) Python – Intensity Transformation Operations on Images. Available at: https://www.geeksforgeeks.org/python/python-intensity-transformation-operations-on-images/ (Accessed: October 2026).
15. GeeksforGeeks (no date) CLAHE Histogram Equalization – OpenCV. Available at: https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/ (Accessed: October 2026).
16. GeeksforGeeks (no date) Program to extract frames using OpenCV-Python. Available at: https://www.geeksforgeeks.org/python/python-program-extract-frames-using-opencv/ (Accessed: November 2026).
17. GeeksforGeeks (no date) Python OpenCV | cv2.cvtColor() method. Available at: https://www.geeksforgeeks.org/python/python-opencv-cv2-cvtcolor-method/ (Accessed: October 2026).
18. GeeksforGeeks (no date) Image Resizing using OpenCV | Python. Available at: https://www.geeksforgeeks.org/python/image-resizing-using-opencv-python/ (Accessed: November 2026).
19. GeeksforGeeks (no date) Python OpenCV | cv2.putText() method. Available at: https://www.geeksforgeeks.org/python/python-opencv-cv2-puttext-method/ (Accessed: October 2026).
20. GeeksforGeeks (no date) How to Display Multiple Images in One Window using OpenCV Python? Available at: https://www.geeksforgeeks.org/python/how-to-display-multiple-images-in-one-window-using-opencv-python/ (Accessed: November 2026).
21. GeeksforGeeks (no date) Command-Line Option and Argument Parsing using argparse in Python. Available at: https://www.geeksforgeeks.org/python/command-line-option-and-argument-parsing-using-argparse-in-python/ (Accessed: November 2026).
22. GeeksforGeeks (no date) How to pass a list as a command-line argument with argparse? Available at: https://www.geeksforgeeks.org/python/how-to-pass-a-list-as-a-command-line-argument-with-argparse/ (Accessed: November 2026).

23. Koller, D., Weber, J., Huang, T., Malik, J., Ogasawara, G., Rao, B. and Russell, S. (1994) 'Towards robust automatic traffic scene analysis in real-time', Proceedings of the 12th International Conference on Pattern Recognition, Jerusalem, pp. 126–131. doi:10.1109/ICPR.1994.576243. Cited in Piccardi (2004).
24. LearnOpenCV (2019) Simple Background Estimation in Videos using OpenCV (C++/Python). Available at: https://learnopencv.com/simple-background-estimation-in-videos-using-opencv-c-python/ (Accessed: November 2026).
25. OpenCV (no date) Changing the contrast and brightness of an image! Available at: https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
26. OpenCV (no date) Operations on arrays. Available at: https://docs.opencv.org/4.x/d2/de8/group__core__array.html
27. Streamlit (no date) st.slider. Available at: https://docs.streamlit.io/develop/api-reference/widgets/st.slider
28. Google Fonts (no date) Fredoka. Available at: https://fonts.google.com/specimen/Fredoka
29. Google Fonts (no date) Figtree. Available at: https://fonts.google.com/specimen/Figtree
30. Pillow (no date) Image file formats: GIF. Available at: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
31. Rother, C., Kolmogorov, V. and Blake, A. (2004) '"GrabCut": interactive foreground extraction using iterated graph cuts', ACM Transactions on Graphics, 23(3), pp. 309–314.
32. OpenCV (no date) Interactive Foreground Extraction using GrabCut Algorithm. Available at: https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html
