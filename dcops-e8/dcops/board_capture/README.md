# Board-camera capture for fine-tuning

The classifier is fine-tuned on frames from the E8's own camera. These tools record those frames.

1. Build and flash the demo firmware (`scripts/alif/build_dcops_demo.sh`), then power-cycle the board.
2. Open the clock-synced slideshow on a phone. Slide k is on screen during `[EPOCH + SEC*k, EPOCH + SEC*(k+1))`:
   - deck 1: `slides.json`, 96 training-split photos of 8 classes, 6 s per slide, EPOCH 2026-10-18 00:00 UTC
   - deck 2: `slides2.json`, 150 photos of the 5 demo classes at varied zoom, 4 s per slide, EPOCH 2026-10-18 19:00 UTC (made by `build_deck2.py`)
3. Hold the phone 5-7 cm in front of the camera (the camera faces the same side as the LCD) and run
   `python capture.py clock --deck 2`. It reads the model-input buffer `rgb_image` over J-Link about
   once every 0.7 s and labels each frame from the clock. Frames within 1.2 s of a slide change are skipped.
   `python capture.py background --seconds 120` records frames labelled background.
4. `python build_ft_data.py` drops blank and stale frames (and frames marked `keep=False` after review),
   splits by slide id (every 6th slide held out), adds 180 human-labelled replay crops per class, and
   writes `board_ft.npz`.
5. Upload `board_ft.npz` to the dataset repo and run `dcops/colab/dcops_board_ft.ipynb`
   (`dcops/jobs/finetune_board.py`). It prints board test accuracy before and after.

Always look at a contact sheet of the frames before training. One capture with the camera aimed at the
ceiling produced labels for frames that showed no slide, and training on it made the model worse.
