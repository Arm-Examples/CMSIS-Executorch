#!/usr/bin/env bash
# Runs on the Mac overnight: every 15 minutes, pull finished DC-Ops models from the Hub, export each
# new one to an Ethos-U85 ExecuTorch program, and rewrite REPORT.md. Light CPU work only.
cd "$(dirname "$0")/.."
VENV=~/projects/arm-executorch-hack/experiments/blazeface/.venv-et110/bin/python
TRAIN=.venv-train/bin/python
HUB=models/hub
while true; do
  $TRAIN -c "from huggingface_hub import snapshot_download as s; s('abhijitbetigeri/dcops-e8-student', local_dir='$HUB', allow_patterns=['vision/*','kws/*','logs/*'])" >/dev/null 2>&1
  for d in $HUB/vision/*/; do [ -f "$d/student.pt" ] && [ ! -f "$d/export.json" ] && PATH=$(dirname $VENV):$PATH $VENV export/export_student.py "$d" > "$d/export.log" 2>&1; done
  for d in $HUB/kws/*/;    do [ -f "$d/student.pt" ] && [ ! -f "$d/export.json" ] && PATH=$(dirname $VENV):$PATH $VENV export/export_kws.py "$d"     > "$d/export.log" 2>&1; done
  $TRAIN - <<'PY' > REPORT.md
import glob, json, os, time
print(f"# DC-Ops E8 overnight report\n\nUpdated {time.strftime('%Y-%m-%d %H:%M')}\n")
print("## Vision students (Ethos-U85 exports)\n\n| run | val acc | board-sim acc | int8 agrees w/ float | .pte KB | CPU ops |\n|---|---|---|---|---|---|")
for d in sorted(glob.glob("models/hub/vision/*/")):
    m = json.load(open(d + "metrics.json")) if os.path.exists(d + "metrics.json") else {}
    e = json.load(open(d + "export.json")) if os.path.exists(d + "export.json") else {}
    print(f"| {os.path.basename(d[:-1])} | {m.get('val_acc', 0):.3f} | {m.get('val_acc_board_sim', 0):.3f} | {e.get('float_int8_agreement', '–')} | {e.get('kb', '–')} | {', '.join(e.get('cpu_ops', [])) or '–'} |")
print("\n## Keyword spotters\n\n| run | test acc | params | int8 agrees w/ float | .pte KB |\n|---|---|---|---|---|")
for d in sorted(glob.glob("models/hub/kws/*/")):
    m = json.load(open(d + "metrics.json")) if os.path.exists(d + "metrics.json") else {}
    e = json.load(open(d + "export.json")) if os.path.exists(d + "export.json") else {}
    print(f"| {os.path.basename(d[:-1])} | {m.get('test_acc', 0):.4f} | {m.get('params', '–')} | {e.get('float_int8_agreement', '–')} | {e.get('kb', '–')} |")
PY
  sleep 900
done
