# Motion Stickers!
 
Motion Sticker Extraction: An image-processing project for IDSP 1701 (Image Processing) at TU Dublin.
 
Upload fixed-camera footage, such as indoor CCTV, and the system detects whatever is moving and cuts it out as a "sticker", like the iPhone sticker feature, producing a timeline of cutouts in the order they appeared.
 
The project uses classical image-processing techniques only (no machine learning).
 
## Team
- Patrycja Palka (blog: https://patrycjapalka.wordpress.com/)
- Folasayo Owolabi (blog: https://learn5093.wordpress.com/)
- Chloe Vegiga (blog: https://c23444496.wordpress.com/)
## How it works
1. **Preprocessing:** greyscale and Gaussian blur. Night clips also get gamma correction and then CLAHE, to bring out detail in the dark [1], [2]. Whether a clip counts as night is decided once per video, so the enhancement doesn't flicker on and off between frames.
2. **Motion detection:** compare each frame with a running-average background (`cv2.accumulateWeighted`, updated every frame) to find what changed [3], [4]. The background starts as the median of frames spread across the clip [3], [5], and is only updated at full speed where nothing is moving (selective update [3], [6]).
3. **Mask cleanup:** remove shadows, then use morphology to refine the mask (opening to remove noise, closing to fill gaps) [7].
   - **Shadows:** pixels that have only darkened and not changed colour (HSV method [8]). Only applied to colour video, as infrared has no colour information.
   - **Size filters:** small shapes are rejected as noise (the minimum scales with the video size), and leftover shapes smaller than 30% of the largest object are rejected. All other objects are kept, so a dog alongside its owner is kept.
   - **Ghost check:** shapes that aren't moving are dropped if another object is moving [8].
   - **Whole-scene changes:** if the "object" covers more than 50% of the frame (such as the camera changing its own exposure), no sticker is made.
4. **Object enhancement:** boost the contrast with CLAHE [2] (on the brightness channel only, so colours don't change) and sharpen with an unsharp mask, so stickers from murky CCTV look crisp.
5. **Cutout:** keep the moving object, make the background transparent, and add a white sticker outline (scaled to the object's size, so it shows on 4K footage too).
6. **Stickers:** skip near-duplicate stickers (an object standing still), then download them as a zip, a printable sheet or a stop-motion GIF [9].
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
Videos are not stored in this repo because of their size, and because the stock clips are licensed previews. Put them in a `footage/` folder in the project root. The hand-marked frames for each clip go in `ground_truth/<clip name>/`, which is not in the repo either.
 
We evaluated on six clips, all from fixed cameras except Donny:
 
| Clip | What it shows |
| --- | --- |
| `donny.mov` | our own 4K footage, filmed handheld (an example of camera movement) |
| `burglary.mp4` | indoor CCTV, daytime: a person walking through a kitchen and living room [10] |
| `dogwalk.mp4` | outdoor infrared night: a person walking a dog past [11] |
| `bunny.mp4` | outdoor infrared night: a rabbit that sits still, then hops away [12] |
| `night_torch.mp4` | indoor night: an intruder with a torch going upstairs [13] |
| `night_shadows.mp4` | indoor night: an intruder casting large shadows on the floor [14] |
 
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
 
**Marking rule (use it for every clip so scores can be compared):** trace every person or animal in view, moving or still. Don't trace shadows or light (such as a torch beam). Skip only frames where no person or animal is in view. For clips that are empty for long stretches, use `--count 20`.
 
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
 
IoU is the overlap measure used in the PASCAL VOC challenge [15].
 
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
Our runs: `before` (old pipeline), `sayo` (selective update + median start), `final` (+ Pati's changes), `shadowfix` (shadow removal on colour footage only), `alphas` (learning-rate test) and `release` (the final app, used for the results below). The `evaluation/` folder is ignored by git.
 
## Results
Mean IoU / Extra on each clip, from the `before` run (the old pipeline) and the `release` run (*App: + shadows*, exactly what the app runs):
 
| Clip | Before | Final |
| --- | --- | --- |
| Burglary | 0.476 / 0.405 | **0.634 / 0.261** |
| Dog walk | 0.413 / 0.183 | **0.553 / 0.067** |
| Bunny | 0.059 / 0.269 | **0.329 / 0.023** |
| Torch | 0.208 / 0.660 | **0.454 / 0.465** |
| Shadows | 0.295 / 0.673 | **0.322 / 0.664** |
| Donny (handheld) | 0.292 / 0.702 | **0.303 / 0.690** |
 
Every clip improved on both numbers. The weakest results are lighting changes (torch light, an opened door), dark moonlight shadows and handheld footage, because background subtraction assumes a fixed camera.
 
## Project structure
- `app.py`: Streamlit interface (upload, display, download)
- `styles/style.css`, `.streamlit/config.toml`: the app's look
- `styles/responsive.css`: makes the page fit smaller windows and phones
- `stickers.py`: the image-processing pipeline
- `requirements.txt`: libraries to install (the versions we tested with)
- `evaluate.py`: scores motion-detection methods against ground truth (IoU, extra area, timing)
- `mark_frames.py`: hand-mark ground-truth masks for a clip
- `compare_methods.py`: side-by-side images of alternative methods, including GrabCut [16], [17]; `--make-test` makes a synthetic clip with exact ground truth
- `tune_night.py`: helps choose the night settings (`NIGHT_BRIGHTNESS`, `NIGHT_GAMMA`): reports each clip's brightness and saves a gamma comparison image
Other documentation and tutorials used while writing the code: OpenCV image transformations [18], Python's `time` module [19], Streamlit [20], GeeksforGeeks tutorials [21]–[27], OpenCV contrast and array operations [28], [29], Streamlit sliders [30], and the Google Fonts Fredoka and Figtree [31], [32].
 
Made locally and not in the repo: `footage/` (videos), `ground_truth/` (marked frames), `output/` (from `compare_methods.py`), and `evaluation/` (results, per-frame scores and comparison images for each run).
 
 
## Status
- [x] Project setup and interface starter
- [x] Motion detection (running-average background, selective update, median start)
- [x] Mask cleanup and cutouts (shadow removal, size filters, ghost check, whole-scene rule)
- [x] Night-time footage handling (gamma and CLAHE, decided once per clip)
- [x] Sticker outline scaling and de-duplication
- [x] Evaluation tools, and six real clips hand-marked and scored
- [x] Final evaluation after the shadow and night changes
- [x] Choosing stickers before download, printable sheet and responsive layout
The detection code in `stickers.py` hasn't changed since the final evaluation, so the results above match the code.
 
## References
References use IEEE style and are numbered in the order they are first cited.
 
[1] GeeksforGeeks, "Python – Intensity transformation operations on images," GeeksforGeeks. [Online]. Available: https://www.geeksforgeeks.org/python/python-intensity-transformation-operations-on-images/ (accessed Oct. 2026).
 
[2] GeeksforGeeks, "CLAHE histogram equalization – OpenCV," GeeksforGeeks. [Online]. Available: https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/ (accessed Oct. 2026).
 
[3] M. Piccardi, "Background subtraction techniques: A review," in *Proc. IEEE Int. Conf. Systems, Man and Cybernetics*, The Hague, Netherlands, 2004, pp. 3099–3104.
 
[4] OpenCV, "Motion analysis and object tracking," OpenCV 4.x Documentation. [Online]. Available: https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html (accessed Nov. 2026).
 
[5] LearnOpenCV, "Simple background estimation in videos using OpenCV (C++/Python)," LearnOpenCV, 2019. [Online]. Available: https://learnopencv.com/simple-background-estimation-in-videos-using-opencv-c-python/ (accessed Nov. 2026).
 
[6] D. Koller, J. Weber, T. Huang, J. Malik, G. Ogasawara, B. Rao, and S. Russell, "Towards robust automatic traffic scene analysis in real-time," in *Proc. 12th Int. Conf. Pattern Recognition*, Jerusalem, Israel, 1994, pp. 126–131, doi: 10.1109/ICPR.1994.576243.
 
[7] OpenCV, "Morphological transformations," OpenCV 4.x Documentation. [Online]. Available: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html (accessed Nov. 2026).
 
[8] R. Cucchiara, C. Grana, M. Piccardi, and A. Prati, "Detecting moving objects, ghosts, and shadows in video streams," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 25, no. 10, pp. 1337–1342, Oct. 2003.
 
[9] Pillow, "Image file formats: GIF," Pillow Documentation. [Online]. Available: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif (accessed Nov. 2026).
 
[10] O2O Creative, "Security camera home burglary in progress," Getty Images, stock footage no. 1084342338. [Online Video]. Available: https://www.gettyimages.ie/detail/video/security-camera-home-burglary-in-progress-stock-footage/1084342338 (accessed Nov. 2026).
 
[11] WW News, "People walking with dog in the early morning, image captured by CCTV surveillance camera," Getty Images, stock footage no. 1382583677. [Online Video]. Available: https://www.gettyimages.ie/detail/video/people-walking-with-dog-in-the-early-morning-image-stock-video-footage/1382583677 (accessed Nov. 2026).
 
[12] WW News, "CCTV surveillance camera point of view of rabbit jumping forward at night," Getty Images, stock footage no. 2044643186. [Online Video]. Available: https://www.gettyimages.ie/detail/video/surveillance-camera-point-of-view-of-rabbit-jumping-stock-footage/2044643186 (accessed Nov. 2026).
 
[13] R. Beckham, "Security camera view of hooded intruder with a torch entering a home and walking upstairs backlit night," Getty Images, stock footage no. 2200417282. [Online Video]. Available: https://www.gettyimages.ie/detail/video/security-camera-view-of-hooded-intruder-with-a-torch-stock-footage/2200417282 (accessed Nov. 2026).
 
[14] tawattiw, "Burglar stealing valuables from desk inside house at night," Getty Images, stock footage no. 2298084001. [Online Video]. Available: https://www.gettyimages.ie/detail/video/burglar-stealing-valuables-from-desk-inside-house-at-stock-footage/2298084001 (accessed Nov. 2026).
 
[15] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, "The PASCAL Visual Object Classes (VOC) challenge," *Int. J. Comput. Vis.*, vol. 88, no. 2, pp. 303–338, Jun. 2010.
 
[16] C. Rother, V. Kolmogorov, and A. Blake, "'GrabCut': Interactive foreground extraction using iterated graph cuts," *ACM Trans. Graph.*, vol. 23, no. 3, pp. 309–314, Aug. 2004.
 
[17] OpenCV, "Interactive foreground extraction using GrabCut algorithm," OpenCV 4.x Documentation. [Online]. Available: https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html (accessed Nov. 2026).
 
[18] OpenCV, "Geometric image transformations," OpenCV 4.x Documentation. [Online]. Available: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html (accessed Nov. 2026).
 
[19] Python Software Foundation, "time — Time access and conversions," Python 3 Documentation. [Online]. Available: https://docs.python.org/3/library/time.html (accessed Nov. 2026).
 
[20] Streamlit, "Streamlit documentation," Streamlit. [Online]. Available: https://docs.streamlit.io (accessed Nov. 2026).
 
[21] GeeksforGeeks, "Program to extract frames using OpenCV-Python," GeeksforGeeks. [Online]. Available: https://www.geeksforgeeks.org/python/python-program-extract-frames-using-opencv/ (accessed Nov. 2026).
 
[22] GeeksforGeeks, "Python OpenCV | cv2.cvtColor() method," GeeksforGeeks. [Online]. Available: https://www.geeksforgeeks.org/python/python-opencv-cv2-cvtcolor-method/ (accessed Oct. 2026).
 
[23] GeeksforGeeks, "Image resizing using OpenCV | Python," GeeksforGeeks. [Online]. Available: https://www.geeksforgeeks.org/python/image-resizing-using-opencv-python/ (accessed Nov. 2026).
 
[24] GeeksforGeeks, "Python OpenCV | cv2.putText() method," GeeksforGeeks. [Online]. Available: https://www.geeksforgeeks.org/python/python-opencv-cv2-puttext-method/ (accessed Oct. 2026).
 
[25] GeeksforGeeks, "How to display multiple images in one window using OpenCV Python?," GeeksforGeeks. [Online]. Available: https://www.geeksforgeeks.org/python/how-to-display-multiple-images-in-one-window-using-opencv-python/ (accessed Nov. 2026).
 
[26] GeeksforGeeks, "Command-line option and argument parsing using argparse in Python," GeeksforGeeks. [Online]. Available: https://www.geeksforgeeks.org/python/command-line-option-and-argument-parsing-using-argparse-in-python/ (accessed Nov. 2026).
 
[27] GeeksforGeeks, "How to pass a list as a command-line argument with argparse?," GeeksforGeeks. [Online]. Available: https://www.geeksforgeeks.org/python/how-to-pass-a-list-as-a-command-line-argument-with-argparse/ (accessed Nov. 2026).
 
[28] OpenCV, "Changing the contrast and brightness of an image!," OpenCV 4.x Documentation. [Online]. Available: https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html (accessed Nov. 2026).
 
[29] OpenCV, "Operations on arrays," OpenCV 4.x Documentation. [Online]. Available: https://docs.opencv.org/4.x/d2/de8/group__core__array.html (accessed Nov. 2026).
 
[30] Streamlit, "st.slider," Streamlit Documentation. [Online]. Available: https://docs.streamlit.io/develop/api-reference/widgets/st.slider (accessed Nov. 2026).
 
[31] Google Fonts, "Fredoka," Google Fonts. [Online]. Available: https://fonts.google.com/specimen/Fredoka (accessed Nov. 2026).
 
[32] Google Fonts, "Figtree," Google Fonts. [Online]. Available: https://fonts.google.com/specimen/Figtree (accessed Nov. 2026).
