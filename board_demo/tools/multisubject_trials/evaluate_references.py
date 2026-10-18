"""Evaluate all genuine inputs; reconstructed TOSA remains provisional until release/FVP verification.

Run from the frontend worktree with the official Arm Python environment.
Fails closed when the saved Q/DQ references are not reproduced exactly.
No serial, hardware, training or model modification is performed.
"""
from pathlib import Path
import json, hashlib, warnings, collections
import numpy as np
import torch, torchao
import tosa_reference_model as ref
from tosa.TosaGraph import TosaGraph
root=Path.cwd(); base=root/'artifacts/multisubject-v4'; out=base/'numerical'
torch.set_num_threads(1)
m=torch.export.load(out/'reference_qdq.pt2').module()
params=json.loads((base/'bundle/preprocess_params.json').read_text())
z=np.load(base/'data/windows.npz'); inputs=z['inputs_int8']; saved=z['qdq_logits']; labels=z['labels']
scale=params['input_quant_scale'];zp=params['input_zero_point']
gpath=out/'compiler_debug_provisional/output_tag0_TOSA-1.0+INT+int16+int4+cf.tosa'
graph=gpath.read_bytes();block=TosaGraph.GetRootAs(graph).Regions(0).Blocks(0)
assert block.InputsLength()==block.OutputsLength()==1
report=json.loads((out/'reference-build.json').read_text()); oscale=report['output_scale'];ozp=report['output_zero_point']
qdq=[]; raw=[]
for i in range(1080):
    x=(inputs[i:i+1].astype(np.float32)-np.float32(zp))*np.float32(scale)
    with torch.no_grad(): y=m(torch.from_numpy(x))
    if isinstance(y,tuple):y=y[0]
    qdq.append(y.numpy()[0])
    values,status=ref.run(graph,[inputs[i:i+1].copy()])
    assert status==ref.GraphStatus.TOSA_VALID and values[0].dtype==np.int8 and values[0].shape==(1,2)
    raw.append(values[0][0])
    if (i+1)%120==0: print(f'Audited {i+1}/1080',flush=True)
qdq=np.asarray(qdq);raw=np.asarray(raw);tosa=(raw.astype(np.float32)-np.float32(ozp))*np.float32(oscale)
qraw=np.rint(qdq.astype(np.float64)/oscale+ozp).astype(np.int16)
delta=raw.astype(np.int16)-qraw
result={'trials':1080,'qdq_vs_saved_exact':bool(np.array_equal(qdq,saved)),'qdq_vs_saved_max_error':float(np.abs(qdq-saved).max()),'qdq_vs_saved_changed_classes':int(np.sum(qdq.argmax(1)!=saved.argmax(1))),'qdq_accuracy_correct':int(np.sum(qdq.argmax(1)==labels)),'diagnostic_tosa_accuracy_correct':int(np.sum(tosa.argmax(1)==labels)),'qdq_vs_diagnostic_tosa_changed_classes':int(np.sum(qdq.argmax(1)!=tosa.argmax(1))),'qdq_vs_diagnostic_tosa_001_failures':int(np.sum(np.max(np.abs(qdq-tosa),axis=1)>.01)),'qdq_vs_diagnostic_tosa_max_error':float(np.max(np.abs(qdq-tosa))),'trial_step_histogram':dict(collections.Counter(map(int,np.max(np.abs(delta),axis=1)))),'tosa_graph_sha256':hashlib.sha256(graph).hexdigest(),'output_scale':oscale,'output_zero_point':ozp,'tosa_status':'PROVISIONAL reconstructed compiler graph; NOT validated against released PTE/FVP','physical_board_used':False}
np.savez_compressed(out/'provisional-references.npz',qdq_logits=qdq,tosa_raw_int8=raw,tosa_logits=tosa,labels=labels)
(out/'desktop-evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
assert result['qdq_vs_saved_exact'], 'Saved Q/DQ mismatch: inspect desktop-evaluation.json; do not approve deployment'
