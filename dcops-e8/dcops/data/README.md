# DC-Ops datasets

## `dc_components/`: data-center component photos

YOLO-format segmentation labels for 16 data-center component classes (`dataset.yaml` lists them).
The train split has 1,918 images and the val split 138. The val split is the one behind the "val"
accuracies in `dcops/docs/RESULTS.md`.

The teacher and student crops are cut from these images by `dcops/jobs/` (`teacher.py`,
`teacher_v2.py`, `student.py`). `dcops/board_capture/` uses the train split for the slideshows and the replay set.

Image sources. All are CC BY 4.0, which requires attribution:

| Prefix of the file name | Source |
|---|---|
| `server_*` | [Server Vision](https://universe.roboflow.com/hunters-workspace-kqclz/server-vision/dataset/5) and [Server Rack](https://universe.roboflow.com/server-rack-detection/server-rack/dataset/12), Roboflow Universe |
| `pc_*` | [PC Ports](https://universe.roboflow.com/ports-gmmmp/pc-ports/dataset/10), Roboflow Universe |
| `ports_*` | [Ports](https://universe.roboflow.com/ym-pffnw/ports-yutnj/dataset/1), Roboflow Universe |
| `fujitsu_*` | [Server Detection](https://universe.roboflow.com/fujitsu-ih8ei/server-detection/dataset/1), Roboflow Universe |

Labels were mapped to the 16 DC-Ops classes. All 2,056 images carry human labels.

## `board_captures/`: frames from the E8's own camera

These are the model-input frames (160x160 RGB, uint8) recorded over J-Link from the board while a
phone showed a clock-synced slideshow. `dcops/board_capture/README.md` explains the method. Each
`.npz` file holds `images`, `labels` (class id, 16 = background) and `slide_ids` (-1 when there is no
slide). Two files also hold `keep`, a hand review that is False where the phone was out of view.

| File | Contents |
|---|---|
| `clock_111029.npz` | deck 1, 397 frames; 233 kept, the rest show the ceiling |
| `clock_113014.npz` | deck 1, 591 frames, all 96 slides |
| `clock_deck2_121751.npz` | deck 2, 509 frames: 313 slide frames, then about 190 room frames relabelled as background |

`python dcops/board_capture/build_ft_data.py` turns these into `board_ft.npz`, which has 1,330
frames: 1,119 for training and 211 for testing, split by slide. That is the set behind the deployed
`v3_board_ft` model.
