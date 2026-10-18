# DCSM-Simplified: live demo run sheet (~2 min)

## Before judges arrive (T-10 min)

- [ ] Demo firmware built with `scripts/alif/build_dcops_demo.sh` (mask 0x13250, cutoff 0.6) and
      flashed with `scripts/alif/flash_e8_hp.sh`.
- [ ] **Power-cycle**: unplug PRG USB about 10 s. The camera hangs after a J-Link reset. If camera
      init fails, LED1 blinks: power-cycle again.
- [ ] Preview live on the LCD, footer shows ms / FPS.
- [ ] Phone: held-out test photo page (5 classes: power shelf, network port, fan, power connector,
      drive bay), brightness max, glare-free angle. These slides were never trained on.
- [ ] Laptop tab: perf table (`DEVPOST.md`) and architecture diagram (`ARCHITECTURE.md`).
- [ ] Quiet spot for voice, or a plan to skip it.
- [ ] Backup video on the laptop: `TODO(record)`.

## Script

| Time | Do | Say |
|---|---|---|
| 0:00 | Hold up the board | "Secure data centers ban phones and cloud. This is an offline component identifier on an Alif E8. Camera in, answer on screen, nothing leaves the device." |
| 0:10 | Point at the phone: drive bay photo, 20–30 cm away | "A MobileNetV2 on the Ethos-U85 runs on every frame. Drive bay, plus its info card. These are held-out photos, never trained on." |
| 0:25 | Swipe: network port, then power shelf | "When it is unsure, it shows nothing. A blank beats a wrong label." |
| 0:40 | Say **"go"** | "Hands are busy in a rack, so voice drives it. 'Go' latches the part and opens its inspection checklist. It only works when vision is confident: vision decides what, voice drives the workflow." |
| 0:50 | Say **"yes"** | "Item 1: 'Identification correct?' Yes." (check mark) |
| 0:55 | Say **"yes"** | "Next item, an LED check from the knowledge base, passes." |
| 1:00 | Say **"no"** | "This one fails: the border turns red. A 'no' on item 1 would reject the identification and go back to scanning: a human correcting the model." |
| 1:05 | Say **"stop"** | "Back to live scanning. The keyword spotter runs on the same NPU: MFCC on the M55 in 3–4 ms, DS-CNN in 0.34 ms." |
| 1:15 | Point at the LCD footer / perf table | "Vision inference is 8 ms on the NPU. The whole frame is 124 ms, 8 FPS, down from 274 ms after we cut per-frame logging. The NPU is 6% of that; camera and display are the bottleneck. Both models, 100% of ops on the U85, one image, 195 KB of MRAM to spare." |
| 1:35 | Show the training diagram | "Our phone detector labelled crops for ConvNeXt teachers, then a MobileNetV2 student. Then we recorded the board camera's own frames against a clock-synced slideshow, no network, and fine-tuned: held-out board accuracy 35.5 to 73 percent." |
| 1:55 | Close | "Offline, NPU-accelerated, voice-driven. Questions?" |

If asked "why only 5 classes": the model has 16 + background. Through the board camera, compute
tray, cooling manifold and DPU were confused, and 8 classes had no board-camera data. We show the
5 the board camera separates reliably, measured on held-out slides: 79% correct, 7% wrong, rest blank.

## Fallbacks

| Problem | Do this |
|---|---|
| Camera frozen or black, or LED1 blinking | Power-cycle (unplug PRG USB, about 10 s). Don't reset with J-Link. If it is still dead, play the backup video and walk through the perf table. |
| Wrong label on a real object | Switch to the phone photo page. Fill the frame with the component and hold it 20–30 cm away. |
| No label shown | Expected when unsure. Hold steady for 1 s, centre the part (the model sees a centre crop), move closer. |
| Labels flicker | Hold steady for 1 s and point at the centre of the frame. |
| Phone glare | Tilt the phone, lower brightness. |
| "go" ignored | Vision is not confident yet: wait for a label, then say "go". |
| Voice doesn't trigger | Speak close to the mic, one word, in a quiet moment. If the room is too loud, say "it's noisy here" and show the KWS accuracy (94.66% test) from `RESULTS.md`. |
| Stuck in the checklist | Say "stop". |
| DC-Ops model misbehaves | Flash the validated 67-class ImageNet build (`build-classify`) and show the board-vs-PC match: keyboard 12.6/13.0%, eraser 9.0/9.0%, TV 8.7/8.1%. Then **power-cycle**. |
| Board won't boot after flash | Don't debug live. Use the second board / backup video. Recovery means writing the MRAM backup through the HE core, which is too slow for a demo slot. |

## Showing perf numbers

- Preferred: the LCD footer (frame ms / FPS) plus the table in `DEVPOST.md` / `RESULTS.md`.
- Measured: vision NPU 7.9–8.0 ms, preprocessing 1.75 ms, frame 255–287 ms (3.5–3.9 FPS),
  KWS MFCC 3.3–4.0 ms + DS-CNN 0.34 ms. After removing per-frame UART logging: frame 124 ms, 8.0 FPS on the LCD footer (was 274 ms, 3.6 FPS): 2.2x faster.
- Power is not measured. Say so if asked.
