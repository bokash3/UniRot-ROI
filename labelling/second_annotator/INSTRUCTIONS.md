# Palm-ROI labelling: instructions for the second annotator

Thank you for helping. You will draw one square, the palm region of interest (ROI), on each of 100 near-infrared
palm images (50 VERA, 50 SCUT-PV). Please work **independently**: do not look at anyone else's labels, at model
outputs, at figures of the paper, and do not discuss the images with the first annotator until you have finished.

## Setup (once)
Python 3.8 or newer with OpenCV and NumPy:  `pip install opencv-python numpy`
If Python is not on the PATH, set it before running, e.g. `set PY=C:\path\to\python.exe`

## Which square?
The standard palm ROI: a square **inside the palm**; its **top edge just below the line joining the finger bases
(the valleys between the fingers)**; its bottom edge above the wrist; its side about the width of the palm.
The square is tilted with the hand, so that "up" points towards the fingers. Use the same rule for every image;
be consistent rather than perfect.

## How to draw (the hand is shown upright, fingers up)
1. Click the **centre** of the square.
2. Click the **middle of the top edge**. This fixes the size and the "up" direction. The square and its crop appear.
3. Fine adjust if needed: `i j k l` move by 1 px (Shift = 5 px); `+ -` size by 1 %; `, .` rotate by 1° (Shift = 5°);
   `z x` change the window size only.
4. `Enter` or `Space` accepts and opens the next image. `r` redoes the clicks, `b` goes back one image.
   `s` skips an image: use it only if the hand is unusable (for example, mostly outside the picture).
5. `q` saves and quits. Labels are saved after every image; run the same file again to continue.

## Steps
1. Double-click `1_practice.bat`: six practice images (three per dataset) to learn the tool; they are not scored.
2. Double-click `2_annotator2.bat`: the 100 images (VERA first, then SCUT-PV), in a fixed random order.
   About 1 to 2 minutes per image; breaks are fine.
3. Send back the two files `data\vera\labels_annotator2.json` and `data\scut\labels_annotator2.json`.

(`3_annotator1_pass2.bat` is for the first annotator only; please do not run it.)
