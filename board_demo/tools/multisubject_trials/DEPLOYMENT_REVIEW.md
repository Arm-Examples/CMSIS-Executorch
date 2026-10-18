# Proposed physical multi-subject test — not authorized or performed

Review `validation/streaming-image.json` and NUMERICAL_VALIDATION_REPORT.md first.
The unchanged unfused v4 student is numerically validated in FVP on the entire
prepared pool; two desktop ties resolve differently and must not be overwritten.
This procedure requires separate explicit authorization covering application load
and the debugger's actual reset/start scope. No provisioning, mass erase, security,
SE/debug-stub or trained-weight changes are part of it. Stop on failure; no retries.

## Verified candidate

Project: `artifacts/multisubject-v4/firmware/project/`.
CMSIS target: DevKit-E8, M55_HP. Application vector base: `0x80200000`.
HEX: `out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.hex`, SHA-256
`37262cc19c74cc86303b0a0c20ebca326ba1b73d68fe47bf322ed3588dad9d56`.
AXF: same directory, `cmsis-executorch.axf`, SHA-256
`0f31b7e28241452a1b7eda202b9c34280f763472b75836567de9ed149b3dc592`.
Model PTE: `6caa8a9876d3cd0c36b39b3d4474a0fe7dbdbfb32a4d6f80ed4776623a19e02e`.
The manifest also checks the generated CMSIS recipe, app/parser source and UART C.
Compile-only memory report: approximately 159,980 ROM bytes and 3,371,352 RAM
bytes; method/temp pools are 1 MiB/2 MiB, with a 22,528-byte float input buffer.
Compilation is not evidence of physical execution or cache/peripheral correctness.

## Proposed sequence after approval

1. Recheck candidate and rollback hashes. Export the current Chrome capture and
   release UART through its Disconnect control. Inspect the existing physical
   debug session through CMSIS; do not disturb it before deployment authorization.
2. Release the old debug/run task through CMSIS, select the isolated project's
   DevKit-E8 context and perform one application-only `load_and_debug`.
   The inherited debugger recipe includes ResetSystem behavior: do not describe
   this as reset-free or assume it resets only one core. If its actual sequence
   exceeds the approved scope, stop for review.
3. Wait for main, clear diagnostic breakpoints and explicitly continue using CMSIS.
   A successful load/task alone is insufficient: the receive loop and HELLO/READY
   must work. Do not use a standalone load_and_run task as proof of startup.
4. Keep SW4 on UART4. Chrome is the sole serial owner. Open
   `http://127.0.0.1:8000/?stream=multisubject`; Connect UART selects SEGGER J-Link
   at 115200 baud, 8N1, no flow control. No CMSIS serial tool may claim that port
   while Chrome owns it. READY must match the full v4 SHA, shape, wire length,
   scale and zero point before Play becomes usable. Stop if it does not.
5. Press Play. Select genuine recordings without replacement across all 1,080
   eligible IDs, reshuffling only after exhaustion. There is no fixed five-trial
   cap and no reference-based selection. Pause manually after the first few
   returned predictions; allow an in-flight request to finish safely.
6. Export capture. Verify actual request/session/trial IDs and input CRC/hash;
   compare returned logits with `artifacts/multisubject-v4/fvp-validation/backend-reference.npz`
   at 0.01 and verify exact output-code reconstruction. Keep the saved desktop
   diagnostic separate. Confirm EEG, true cue, cursor and accuracy show that same
   trial. No result may come from replay or laptop inference.
7. Continue prolonged operation only after that review. Timeout, disconnect or
   inference failure stops scheduling and retains results; do not automatically
   reset, reflash, provision or fabricate a response.

## Rollback choices — separately authorized

Preferred immediate fallback: retain the proven Model 2 streaming project at
`artifacts/streaming-firmware/project/`. Its HEX SHA is
`49fcd35c3e804273d631c39fdcedb313d3262a99762fe8a5037ebfac723dd460`, AXF SHA
`75d6cea5dc109b7596c72b920acca4538ac9211068a0faf7be692911c63cc33f`.
Load that checked application through CMSIS with explicitly approved reset/start
scope, continue from main, use `?stream=1`, and require its Model 2 READY before Play.

The original four-trial fallback is also preserved at
`artifacts/streaming-rollback/validated-four-trial.tar.gz` plus adjacent manifest,
SHA `90434792bdc757e225c78136e1c5261dd56acb6e3d5a51a1b8fc192d73c36015`.
Its 136 source/config/image files were reverified. Follow the existing exact
restoration steps and supplemental recipe in
`tools/model2_random_trials/PREDEPLOYMENT_REVIEW.md` and
`tools/model2_random_trials/validation/validated-four-trial.cbuild-run.yml`.
Use `?v=eeg-1` with a reader capturing before authorized startup; verify the four
predictions `[0,1,1,1]` and PASS. Do not overwrite the original firmware worktree.
Rollback is prepared and hash-verified, not newly exercised on hardware tonight.
