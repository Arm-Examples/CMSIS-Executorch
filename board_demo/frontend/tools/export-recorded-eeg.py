"""Export only the four verified Model 2 windows for EEG display; no training/inference/downloads."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import sosfilt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-repo',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();root=args.baseline_repo.resolve();out=args.out_dir.resolve()
    stage=root/'artifacts/fvp-subject1-20261016-ready';export=root/'artifacts/subject1-ethosu85-seed42-20261016'
    refs=json.loads((stage/'reference.json').read_text());manifest=json.loads((export/'data_manifest.json').read_text());params=json.loads((export/'float_preprocess_params.json').read_text())
    d=torch.load(export/'export_data.pt',weights_only=True);v=np.load(stage/'validation_vectors.npz');indices=refs['indices']
    assert refs['labels']==[0,0,1,1];assert indices==[0,2,1,8]
    assert [manifest['development_trials'][i] for i in indices]==refs['trials']
    assert d['development_y'][indices].tolist()==refs['labels']==v['labels'].tolist()
    assert digest(export/'tinyeegnet_int8.pte')==refs['model_sha256']
    source=next((root/'mne_data').rglob('A01T.mat'));source_digest=digest(source)
    assert source_digest=='054f02e70cf9c4ada1517e9b9864f45407939c1062c6793516585c6f511d0325'
    mat=loadmat(source,simplify_cells=True)['data'];task_runs=[(i,r) for i,r in enumerate(mat) if np.size(r['trial'])]
    channels=params['channel_order'];assert len(channels)==22 and len(set(channels))==22
    mean=np.array(params['mean'])[:,None];std=np.array(params['std'])[:,None];sos=np.array(params['sos'])
    normalized=d['development_x'][indices].numpy();q=refs['input_quantization']
    quantized=torch.clamp(torch.round(d['development_x'][indices]/q['scale'])+q['zero_point'],-128,127).to(torch.int8).numpy()
    header=(stage/'project/src/eeg_vectors.h').read_text().split('={',1)[1]
    header_bytes=bytes(int(x)%256 for x in re.findall(r'-?\d+',header))
    assert quantized.tobytes()==v['inputs'].tobytes()==header_bytes
    inputs_sha=hashlib.sha256(header_bytes).hexdigest();assert inputs_sha=='038c1f93ea0818fb2c2b21da6521506e38d222280a1178b09ec5ce3e28d9c8fb'
    records={}
    for uart_trial,t in enumerate(refs['trials']):
        mat_index,run=task_runs[int(t['run'])];assert run['fs']==250
        ordinals=np.flatnonzero(np.asarray(run['trial'])-1+500==t['cue_sample']);assert len(ordinals)==1
        ordinal=int(ordinals[0]);assert int(run['y'][ordinal])-1==refs['labels'][uart_trial]
        assert bool(run['artifacts'][ordinal])==t['artifact']
        start,stop=t['start'],t['stop'];history=t['cue_sample']-500
        # Original .mat X is microvolts; MNE pipeline converts to volts before causal filtering.
        volts=np.asarray(run['X'][history:stop,:22],dtype=np.float64).T*1e-6
        filtered=sosfilt(sos,volts,axis=-1)[:,start-history:stop-history]
        assert filtered.shape==(22,500) and np.isfinite(filtered).all()
        assert np.array_equal(((filtered-mean)/std).astype(np.float32),normalized[uart_trial,0]),'Original filtered samples must reproduce pre-quantization export exactly'
        samples=(filtered*1e6).astype('<f4');key=f"S{t['subject']}/{t['session']}/run{t['run']}/cue{t['cue_sample']}"
        rms=np.sqrt(np.mean(samples.astype(np.float64)**2,axis=1))
        records[key]={'trial_id':key,'uart_trial':uart_trial,'source_trial':t,'development_index':indices[uart_trial],
                      'mat_run_1based':mat_index+1,'mat_trial_1based':ordinal+1,'ground_truth':refs['labels'][uart_trial],
                      'samples_sha256_float32_le':hashlib.sha256(samples.tobytes()).hexdigest(),
                      'rms_uv':rms.tolist(),'channels':{name:samples[i].tolist() for i,name in enumerate(channels)}}
    payload={'schema':1,'profile_id':'model2-four-window-20261017','model_sha256':refs['model_sha256'],'inputs_sha256':inputs_sha,
             'fs_hz':250,'window_samples':500,'window_cue_seconds':[1,3],'units':'µV','channel_order':channels,
             'processing':'causal 7–30 Hz Butterworth N=4 prototype (8th-order bandpass); SOS zero-state at cue minus 2 seconds, 3 seconds warm-up; float32 microvolts before normalization/quantization',
             'trials':records}
    provenance={'source':str(source.relative_to(root)),'source_sha256':source_digest,
                'source_units':'microvolts; converted to volts for original filtering','display_units':'microvolts; filtered, not normalized or quantized',
                'source_files':{str(p.relative_to(root)):digest(p) for p in [stage/'reference.json',export/'data_manifest.json',export/'float_preprocess_params.json',export/'export_data.pt',stage/'validation_vectors.npz',stage/'project/src/eeg_vectors.h',root/'src/data/preprocess.py']},
                'model_sha256':payload['model_sha256'],'inputs_sha256':inputs_sha,'original_export_reconstructed_exactly':True,'quantized_inputs_match_staged_npz_and_header':True,
                'channel_order':channels,'fs_hz':250,'window_samples':500,'filter_sos':sos.tolist(),
                'trials':[{k:value for k,value in rec.items() if k!='channels'} for rec in records.values()]}
    out.mkdir(parents=True,exist_ok=False)
    datafile=out/'model2-eeg-data.js';datafile.write_text('/* Genuine four-window prerecorded EEG. Generated by tools/export-recorded-eeg.py. */\nwindow.EdgeRecordedEEGData='+json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)+';\n')
    provenance['display_asset_sha256']=digest(datafile)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print('PASS: raw recording -> causal filtered volts -> normalized float32 export is exact for all four windows.')
    print('PASS: export labels/source metadata and all staged int8 bytes agree. Display data:',datafile,len(datafile.read_bytes()),'bytes')
    for rec in records.values():print(rec['uart_trial'],rec['trial_id'],'MAT run/trial',rec['mat_run_1based'],rec['mat_trial_1based'],'label',rec['ground_truth'])

if __name__=='__main__':main()
