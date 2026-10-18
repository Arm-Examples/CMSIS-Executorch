# DC-Ops vision model

Deployed vision model for the DCSM-Simplified demo firmware: `dcops_v4_demo_ft_ethos-u85-256.pte` (MobileNetV2, 160 px, 16 classes + background; fine-tuned on frames from the E8's own camera, with loss and scoring over the 6 classes the demo shows). `labels.txt`, `metrics.json` and `export.json` describe it. Earlier versions: `dcops_v3_board_ft_ethos-u85-256.pte` (board fine-tune over all classes) and `dcops_v2_r160_kd_ethos-u85-256.pte` (before the board fine-tune).

PyTorch weights, ablations and teachers: https://huggingface.co/abhijitbetigeri/dcops-e8
