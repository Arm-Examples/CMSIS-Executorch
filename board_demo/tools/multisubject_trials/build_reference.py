"""Reproduce the release Q/DQ and TOSA reference; never replace the release PTE."""
import argparse, hashlib, json, sys, tempfile, shutil
from pathlib import Path
import numpy as np
import torch
import torchao
from executorch.exir._serialize._program import deserialize_pte_binary

def sha(data): return hashlib.sha256(data).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('artifacts/multisubject-v4'))
    root=p.parse_args().root.resolve();bundle=root/'bundle';out=root/'numerical';out.mkdir(exist_ok=True)
    manifest=json.loads((bundle/'release-manifest.json').read_text())
    for item in manifest['assets']:
        assert sha((bundle/item['name']).read_bytes())==item['sha256'],item['name']
    sys.path.insert(0,str(root/'upstream'))
    from prepare_dataset import SOURCE_SHA, check_file
    for name, expected in SOURCE_SHA.items(): check_file(root/'upstream'/name, expected)
    from src.quantize import load_float_model,compile_spec,quantize,input_qparams
    from src.export import lower
    torch.set_num_threads(1)
    widths=json.loads((bundle/'student.json').read_text());params=json.loads((bundle/'preprocess_params.json').read_text())
    shape=tuple(params['input_shape']);model=load_float_model(bundle/'tinyeegnet_float.pt',shape[-1],widths)
    count=sum(p.numel() for p in model.parameters());assert count==3426
    calibration=np.load(bundle/'calibration.npz')['x'];assert calibration.shape==(200,1,22,256)
    debug_dir=Path(tempfile.mkdtemp(prefix='edgebci-v4-reference-',dir='/private/tmp'))
    spec=compile_spec();spec.dump_intermediate_artifacts_to(str(debug_dir))
    qdq=quantize(model,calibration,spec=spec,input_shape=shape)
    assert input_qparams(qdq)==(params['input_quant_scale'],params['input_zero_point'])
    torch.export.save(torch.export.export(qdq,(torch.zeros(shape),)),out/'reference_qdq.pt2')
    print('Reconstructed original Q/DQ reference',flush=True)
    _,_,program=lower(qdq,spec,shape);rebuilt=bytes(program.buffer);original=(bundle/'tinyeegnet_int8.pte').read_bytes()
    (out/'diagnostic_rebuilt.pte').write_bytes(rebuilt)
    shutil.copytree(debug_dir,out/'compiler_debug_provisional',dirs_exist_ok=True)
    original_program=deserialize_pte_binary(original).program
    diagnostic_program=deserialize_pte_binary(rebuilt).program
    original_delegate=[bytes(d.data) for d in original_program.backend_delegate_data]
    diagnostic_delegate=[bytes(d.data) for d in diagnostic_program.backend_delegate_data]
    plan=original_program.execution_plan[0]
    tensors=[]
    for role,indices in [('input',plan.inputs),('output',plan.outputs)]:
        for i in indices:
            t=plan.values[i].val;tensors.append(dict(role=role,index=i,shape=list(t.sizes),dtype=int(t.scalar_type)))
    qnodes=[n for n in qdq.graph.nodes if n.op=='call_function' and 'quantize_per_tensor' in str(n.target) and 'dequantize' not in str(n.target)]
    last=qnodes[-1];output_scale,output_zp=float(last.args[1]),int(last.args[2])
    vectors=np.load(bundle/'test_vectors.npz');iq=vectors['x']
    x=(iq.astype(np.float32)-np.float32(params['input_zero_point']))*np.float32(params['input_quant_scale'])
    values=np.arange(-128,128,dtype=np.int16).astype(np.int8)
    reconstructed=(values.astype(np.float32)-np.float32(params['input_zero_point']))*np.float32(params['input_quant_scale'])
    roundtrip=torch.ops.quantized_decomposed.quantize_per_tensor.default(torch.from_numpy(reconstructed),params['input_quant_scale'],params['input_zero_point'],-128,127,torch.int8).numpy()
    assert np.array_equal(roundtrip,values)
    with torch.no_grad():logits=np.concatenate([qdq(torch.from_numpy(a[None])).numpy() for a in x])
    assert np.array_equal(logits,vectors['logits']),float(np.abs(logits-vectors['logits']).max())
    report=dict(original_pte_sha256=sha(original),reconstructed_pte_sha256=sha(rebuilt),pte_byte_identical=original==rebuilt,
        delegate_byte_identical=original_delegate==diagnostic_delegate,
        original_delegate_sha256=[sha(d) for d in original_delegate],diagnostic_delegate_sha256=[sha(d) for d in diagnostic_delegate],
        reference_status='PROVISIONAL until compiled delegate and released PTE/FVP are independently verified',
        parameters=count,tensor_contract=tensors,operators=[o.name for o in plan.operators],
        input_scale=params['input_quant_scale'],input_zero_point=params['input_zero_point'],
        output_scale=output_scale,output_zero_point=output_zp,output_quant_node=last.name,
        known_20_saved_logits_bit_exact=True,all_256_wire_values_requantize_exact=True,
        policy='Diagnostic reconstruction only; firmware uses unchanged release PTE')
    (out/'reference-build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if original!=rebuilt:raise RuntimeError('Rebuilt PTE differs; investigate before accepting its TOSA reference')
if __name__=='__main__':main()
