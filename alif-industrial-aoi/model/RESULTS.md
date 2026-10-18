# Model validation results

These results summarize the existing calibration evaluation reports for the
included MobileNetV2 PTE. They were not rerun during submission packaging.
Machine-readable values and SHA-256 provenance for the source reports are in
[validation-summary.json](validation-summary.json).

| Sample | FP32 top one | PT2E INT8 top one | TOSA INT8 top one | Matches expected top-one label |
| --- | --- | --- | --- | --- |
| cat.bmp | tabby | tabby | tabby | Yes |
| dog.bmp | Brabancon griffon | Brabancon griffon | Brabancon griffon | No |
| kimono.bmp | kimono | kimono | kimono | Yes |
| tiger.bmp | tiger | tiger | tiger | Yes |

All three execution paths agree on the top-one prediction for **4/4 samples**;
**3/4** match the recorded expected labels. The dog sample's accepted class is
pug (254), while all three paths predict Brabancon griffon (262). All four TOSA
runs returned `TOSA_VALID`.

This is a small smoke test, not an ImageNet benchmark or an industrial defect
accuracy measurement. TOSA evaluates pre-Vela semantics; agreement does not
prove compiled-NPU numerical parity. The original report could not execute the
Ethos-U delegate in its host runtime. The project author separately confirms
that the later working application using this calibrated model was deployed;
no new hardware validation was performed for this submission package.
