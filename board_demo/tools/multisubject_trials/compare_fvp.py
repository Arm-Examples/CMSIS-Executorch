"""Compare actual FVP binary replies with trial identities and independent references.

Never opens serial or uses desktop inference to manufacture target outputs.
The Q/DQ0.01 diagnostic is always reported independently of the TOSA check.
"""
import argparse, hashlib, json, struct, zlib
from pathlib import Path
import numpy as np
MODEL='6caa8a9876d3cd0c36b39b3d4474a0fe7dbdbfb32a4d6f80ed4776623a19e02e'
SCALE=0.024768464267253876
ZP=-4

def uncobs(b):
    out=bytearray();i=0
    while i<len(b):
        c=b[i];i+=1
        if not c or i+c-1>len(b):raise ValueError('Malformed COBS')
        out.extend(b[i:i+c-1]);i+=c-1
        if c!=255 and i<len(b):out.append(0)
    return bytes(out)
def frames(text):
    result=[]
    for line in text.splitlines():
        if not line.startswith('STREAM_FRAME '):continue
        wire=bytes.fromhex(line[13:]);assert wire[-1]==0
        raw=uncobs(wire[:-1]);assert zlib.crc32(raw[:-4])==struct.unpack('<I',raw[-4:])[0]
        magic,version,kind,flags,session,request,trial,res,n,res2=struct.unpack('<4sBBHQIIIHH',raw[:32])
        assert magic==b'EEG2' and version==1 and flags==res==res2==0 and len(raw)==36+n
        result.append(dict(kind=kind,session=session,request=request,trial=trial,payload=raw[32:-4]))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('artifacts/multisubject-v4'))
    p.add_argument('--log',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root;expected=json.loads(a.expected.read_text());assert expected['model_sha256']==MODEL
    assert hashlib.sha256((root/'bundle/tinyeegnet_int8.pte').read_bytes()).hexdigest()==MODEL
    meta=json.loads((root/'data/metadata.json').read_text());inputs=(root/'data/inputs.bin').read_bytes()
    saved=np.load(root/'data/windows.npz')['qdq_logits'];reference=np.load(root/'numerical/provisional-references.npz');tosa=reference['tosa_raw_int8']
    text=a.log.read_text();cases=expected['cases'];assert f'STREAM_TEST PASS results={len(cases)} calls={len(cases)} rejected=2' in text
    ff=frames(text);ready=[f for f in ff if f['kind']==2];assert len(ready)==1
    payload=ready[0]['payload'];assert len(payload)==69 and payload[:32].hex()==MODEL
    assert struct.unpack('<fiI4HB',payload[32:53])==(np.float32(.06955837458372116),-12,5632,1,1,22,256,1)
    assert not any(f['kind']==6 for f in ff)
    ack=[f for f in ff if f['kind']==4];results=[f for f in ff if f['kind']==5];assert len(ack)==len(results)==len(cases)
    rows=[];seen={}
    for e,f,accepted in zip(cases,results,ack):
        idx=e['trial'];row=meta['trials'][idx];data=inputs[idx*5632:(idx+1)*5632]
        assert row['trial_id']==e['trial_id'] and row['ground_truth']==e['ground_truth']
        assert hashlib.sha256(data).hexdigest()==e['input_sha256']==row['input_sha256']
        for frame in (f,accepted):assert (frame['session'],frame['request'],frame['trial'])==(e['session'],e['request'],idx)
        crc,l0,l1,pred=struct.unpack('<IffB3x',f['payload']);assert crc==e['input_crc32']==zlib.crc32(data)==struct.unpack('<I',accepted['payload'])[0]
        logits=np.array([l0,l1],np.float32);assert np.isfinite(logits).all() and pred==int(l1>l0)
        raw=np.rint(logits.astype(np.float64)/SCALE+ZP).astype(np.int16);assert np.all((raw>=-128)&(raw<=127))
        rebuilt=(raw.astype(np.float32)-np.float32(ZP))*np.float32(SCALE);assert np.array_equal(logits.view(np.uint32),rebuilt.view(np.uint32))
        if idx in seen:assert seen[idx]==f['payload'],'Nondeterministic duplicate result'
        seen[idx]=f['payload'];qdq_error=float(np.max(np.abs(logits.astype(np.float64)-saved[idx])))
        target_tosa=(tosa[idx].astype(np.float32)-np.float32(ZP))*np.float32(SCALE)
        rows.append(dict(index=idx,trial_id=row['trial_id'],request=e['request'],ground_truth=row['ground_truth'],prediction=pred,logits=logits.tolist(),raw_int8=raw.tolist(),saved_qdq_logits=saved[idx].tolist(),saved_qdq_prediction=int(saved[idx].argmax()),saved_qdq_max_error=qdq_error,saved_qdq_001_pass=qdq_error<=.01,tosa_raw=tosa[idx].tolist(),tosa_integer_exact=bool(np.array_equal(raw,tosa[idx])),tosa_001_pass=bool(np.max(np.abs(logits-target_tosa))<=.01)))
    result=dict(scope='Actual FVP execution of unchanged release PTE through production streaming parser/input adapter; no physical board',model_sha256=MODEL,cases=len(rows),unique_trials=len(seen),input_ids_and_integrity_verified=True,rejected_invalid_frames=2,output_scale=SCALE,output_zero_point=ZP,tosa_integer_mismatches=sum(not x['tosa_integer_exact'] for x in rows),tosa_001_failures=sum(not x['tosa_001_pass'] for x in rows),saved_qdq_001_failures=sum(not x['saved_qdq_001_pass'] for x in rows),saved_qdq_class_mismatches=sum(x['prediction']!=x['saved_qdq_prediction'] for x in rows),saved_qdq_max_error=max(x['saved_qdq_max_error'] for x in rows),rows=rows)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    assert result['tosa_integer_mismatches']==result['tosa_001_failures']==0,'Released-PTE/TOSA mismatch: investigation required'
if __name__=='__main__':main()
