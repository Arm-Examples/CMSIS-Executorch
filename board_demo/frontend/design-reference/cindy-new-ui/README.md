# ArmExecuTorchEEG — updated front-end source

## Files
- index.html — structure, person and brain SVGs, 22 electrode labels, controls.
- styles.css — current blue-glow theme, Helvetica Neue font stack, layout and visual states.
- gui.js — playback, human cue animation, cursor, accuracy, waveforms and RMS heat-map rendering.
- hardware-adapter.js — transport-independent input adapter. Actual serial/bridge connection is not implemented.

## Run
Keep the files together and open index.html in your browser. No framework, build step or dependencies are required. Optional: run `python3 -m http.server 8000` in this folder and visit http://localhost:8000. Python is only an optional file server; the GUI is JavaScript.

## Integrate without overwriting existing hardware work
Use this folder as the updated presentation/rendering source. Preserve your existing UART/bridge code and connect it through the API below. Do not replace working hardware code with this adapter blindly.

```js
ArmEEGHardware.begin(); // hardware mode; disables mock predictions and EEG
// After your actual transport is connected:
ArmEEGHardware.connected();
// Forward each complete normalized message:
ArmEEGHardware.receive({type: 'prediction', class_id: 0, ground_truth: 0});
ArmEEGHardware.receive({type: 'ground_truth', label: 0});
ArmEEGHardware.receive({type: 'eeg', channels: {C3: [1,2,1], Cz: [0,1,0], C4: [2,1,2]}});
// On actual disconnect:
ArmEEGHardware.disconnected();
```

Class IDs: 0 left, 1 right. Verify firmware mapping. Supply ground_truth on a prediction ONLY when it is paired with the same EEG window; otherwise omit it and the prediction will not count toward accuracy. Separate ground_truth messages control the human cue and are not used for scoring a potentially delayed prediction. Click Play for local cursor motion. Play/Pause does not command firmware.

All 22 montage names accepted in channels: Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz. Coordinates are schematic BCI IV 2a positions; verify against the actual dataset. Pass consistently scaled numeric samples, oldest first. Last 500 samples per supplied channel are retained; each EEG message is a complete display snapshot, so omitted channels become unavailable.

Waveforms use three columns (8/7/7); missing or constant channels are hidden. Heat-map fields use relative per-frame channel RMS from the same samples as the waveforms. This is illustrative scalp EEG amplitude, not ERD or cortical source localization. Frames hold until replaced; EEG freshness detection is not implemented. Mock mode generates simulated signals in all 22 channels and a first prediction immediately, then every 2.5 seconds. Cursor speed is 156 px/s with compact travel bounds.

The source matches the latest HTML design. Syntax and DOM-stub behavior checks passed; physical hardware and browser visual layout have not been verified here. Helvetica Neue uses installed-system availability with Helvetica/Arial fallbacks.
