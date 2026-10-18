# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Export a trained DC-Ops student (vision/<run>/student.pt) to an ExecuTorch 1.1 program for the
Ethos-U85-256 of the Alif E8, int8-calibrated on the run's own calib.npz (real validation crops).
Run with the ExecuTorch 1.1.0 environment (resources/object_detection/blazeface/README.md)."""
import json, os, re, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from student_model import StudentNHWC  # noqa: E402
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner  # noqa: E402
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config  # noqa: E402
from executorch.backends.cortex_m.passes.quantized_op_fusion_pass import QuantizedOpFusionPass  # noqa: E402
from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import ReplaceQuantNodesPass  # noqa: E402
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e  # noqa: E402

run = sys.argv[1]  # folder with student.pt, metrics.json, calib.npz, labels.txt
m = json.load(open(f"{run}/metrics.json")); res = m["args"]["res"]
labels = open(f"{run}/labels.txt").read().split("\n"); labels = [l for l in labels if l]
model = StudentNHWC(len(labels), res, pretrained=False).eval()
model.load_state_dict(torch.load(f"{run}/student.pt", map_location="cpu"))
c = np.load(f"{run}/calib.npz"); calib = [torch.from_numpy(x[None]).float() / 255 for x in c["images"]]
repo = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = EthosUCompileSpec(target="ethos-u85-256", config_ini=f"{repo}/scripts/vela/ensemble_vela.ini",
                         system_config="Ethos_U85_SRAM_MRAM", memory_mode="Shared_Sram",
                         extra_flags=["--output-format=raw", "--debug-force-regor"])
g = torch.export.export(model, (calib[0],)).module()
q = EthosUQuantizer(spec); q.set_global(get_symmetric_quantization_config(is_per_channel=True))
p = prepare_pt2e(g, q)
with torch.no_grad():
    for x in calib: p(x)
qm = convert_pt2e(p)
with torch.no_grad():
    fl = np.array([model(x).argmax().item() for x in calib]); qq = np.array([qm(x).argmax().item() for x in calib])
agree = float((fl == qq).mean()); acc_f = float((fl == c["labels"]).mean()); acc_q = float((qq == c["labels"]).mean())
edge = to_edge_transform_and_lower(torch.export.export(qm, (calib[0],)), partitioner=[EthosUPartitioner(spec)],
                                   compile_config=EdgeCompileConfig(_check_ir_validity=False))
edge = edge.transform([ReplaceQuantNodesPass(), QuantizedOpFusionPass()])
prog = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
out = f"{run}/student_ethos-u85-256.pte"; open(out, "wb").write(prog.buffer)
cpu_ops = sorted({o.decode() for o in re.findall(rb"(?:aten|quantized_decomposed|cortex_m)::\w+", prog.buffer)})
rep = {"pte": out, "kb": round(len(prog.buffer) / 1024), "res": res, "cpu_ops": cpu_ops,
       "calib_acc_float": acc_f, "calib_acc_int8": acc_q, "float_int8_agreement": agree}
json.dump(rep, open(f"{run}/export.json", "w"), indent=2); print(json.dumps(rep))
