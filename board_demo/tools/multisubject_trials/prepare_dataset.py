"""Prepare genuine v4 recordings with pinned upstream code and saved parameters.

Reads local MAT files only. Never downloads data, trains, exports a model or uses
serial/debug tools. Existing different outputs are refused. Run with Python -B.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

REVISION = 'ad61c79de17243f5befce069a1e9ca458057f43e'
MODEL_SHA = '6caa8a9876d3cd0c36b39b3d4474a0fe7dbdbfb32a4d6f80ed4776623a19e02e'
PARAMS_SHA = '3d34f6326c67441e9e6b636454436c63951e81f6e221da977723700f1c8152f7'
REFERENCE_SHA = 'd41cf8bca4149071b737cbbf8ad73f49e1f1223a2c6ef3b3a88108ec7aa66a70'
VECTORS_SHA = '250241cb0967b99e04c355c86c3f8ae4671ed42e71a1f0d65f82fea5affb98c3'
SOURCE_SHA = {
 'src/data/cache125.py':'2b8a1c6d9945a649bc066c4add994bd0cf0a59b9fa1119480b29a9c44a3c5854',
 'src/data/align.py':'50cacc07f0d14cc70ae69c2ad8cc3b01f976e1e6d3de5019129a204fd26e45b1',
 'src/data/preprocess.py':'4addf755b63d2716f1f53d76770289a0c97584e7eb6963122ea8ba3d1db18ef3',
 'src/data/download.py':'0c97c72f871d96ecf654e4f59dca962bb212e78031527080f35bb7355e812718',
 'src/data/external.py':'12dae1db42380e6428c0c070af7091b467877126f3ca9cc2240656498a447b50',
 'src/train.py':'d22294d1f3e3dfc9c19b9b4c6022776c65cd8cc44dcae0a20c1898f65a2f227d',
 'src/quantize.py':'7a78a4974ec5a40e6355a225ddb8f2060e810b0aaeb0ab1d4a92230c69e7d0e5',
}
CHANNELS = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
ARTIFACTS = [1,1,5,24,5,33,4,9,12]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def check_file(path, expected):
    actual = digest(path.read_bytes())
    require(actual == expected, f'Provenance mismatch: {path}: {actual}')

def save_same(path, content):
    """Idempotent output; never replace a different existing artifact."""
    if path.exists():
        require(path.read_bytes() == content, f'Refusing changed-file overwrite: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()

def raw_identities(path, subject):
    """Independent MAT metadata mapping; MATLAB trial samples are one-based."""
    import numpy as np
    from scipy.io import loadmat
    rows = loadmat(path, simplify_cells=True)['data']
    if isinstance(rows, dict):
        rows = [rows]
    identities, offset, retained_run, all_trial, lr_trial = [], 0, 0, 0, 0
    for mat_run, run in enumerate(rows):
        labels = np.asarray(run.get('y', [])).reshape(-1)
        if not len(labels):
            continue  # MOABB also omits calibration-only / no-trial raw runs.
        marks = np.asarray(run['trial']).reshape(-1)
        flags = np.asarray(run['artifacts']).reshape(-1)
        require(len(labels) == len(marks) == len(flags), 'MAT metadata length mismatch')
        require(float(run['fs']) == 250, 'Unexpected source sample rate')
        require(run['X'].shape[1] == 25, 'Unexpected source channel count')
        for mat_trial, label in enumerate(labels):
            all_trial += 1
            if label not in (1, 2):
                continue
            lr_trial += 1
            marker = int(marks[mat_trial])-1
            cue = marker+500
            identities.append({
                'subject':subject, 'session':'1test', 'run':retained_run,
                'cue_sample':cue, 'artifact':bool(flags[mat_trial]),
                'session_trial':lr_trial, 'original_session_trial':all_trial,
                'mat_run_index':mat_run, 'mat_trial_index':mat_trial,
                'mat_marker_sample_1based':int(marks[mat_trial]),
                'cue_sample_session_250hz':offset+cue,
                'ground_truth':int(label)-1,
            })
        offset += len(run['X'])
        retained_run += 1
    require(retained_run == 6 and len(identities) == 144, 'Unexpected Session2 trial structure')
    return identities

def prepare(args):
    import numpy as np
    from scipy.signal import butter
    from moabb.datasets.bnci.base import _convert_mi
    import mne
    source, bundle = args.upstream_source.resolve(), args.bundle.resolve()
    for name, expected in SOURCE_SHA.items():
        check_file(source/name, expected)
    for name, expected in [('tinyeegnet_int8.pte',MODEL_SHA),('preprocess_params.json',PARAMS_SHA),
                           ('offline_int8_predictions.json',REFERENCE_SHA),('test_vectors.npz',VECTORS_SHA)]:
        check_file(bundle/name, expected)
    params = json.loads((bundle/'preprocess_params.json').read_text())
    refs = json.loads((bundle/'offline_int8_predictions.json').read_text())
    require(params['channel_order'] == CHANNELS and params['input_shape'] == [1,1,22,256], 'Model shape/channel contract mismatch')
    require(params['fs'] == 250 and params['fs_model'] == 125 and params['reference'] == 'common_average_22', 'Pipeline contract mismatch')
    require(params['window']['start_samples_after_cue_125hz'] == 62 and params['window']['length'] == 256, 'Window contract mismatch')
    require(params['decimation']['factor'] == 2 and params['decimation']['phase'] == 0, 'Decimation mismatch')
    require(np.array_equal(butter(4,[7,30],btype='bandpass',fs=250,output='sos'),np.asarray(params['sos'])), 'Saved SOS and pinned implementation differ')
    sys.path.insert(0, str(source))
    from src.data import download, cache125
    from src.data.align import apply_ea, ea_matrix
    from src.quantize import quantize_input
    require(Path(download.__file__).resolve() == source/'src/data/download.py', 'Wrong upstream package imported')
    mne.set_log_level('ERROR')
    records, normalized, eeg, inputs, labels, references, sources, per_subject = [], [], [], [], [], [], [], []
    for subject in range(1,10):
        path = args.raw_dir/f'A{subject:02d}E.mat'
        require(path.is_file(), f'Missing genuine raw recording: {path}')
        sha = digest(path.read_bytes())
        ids = raw_identities(path, subject)
        runs, codes = _convert_mi(str(path),CHANNELS+['EOG1','EOG2','EOG3'],['eeg']*22+['eog']*3,
                                  artifact_handling='ignore')
        require(codes == {'left_hand':1,'right_hand':2,'feet':3,'tongue':4}, 'Unexpected event codes')
        raw = download._concat_runs({str(i):r for i,r in enumerate(runs)})
        events = download.find_cue_events(raw)
        lr_events = events[np.isin(events[:,2],[769,770])]
        require(np.array_equal(lr_events[:,0],[r['cue_sample_session_250hz'] for r in ids]), 'Raw cue identity mismatch')
        raw.pick(picks='eeg', exclude=[])
        raw.info['temp'] = {'events':events}
        # This is the original pipeline; it filters continuously across runs.
        uv, y = cache125.windows_2a(raw)
        require(uv.shape == (144,22,256) and np.isfinite(uv).all(), 'EEG shape/nonfinite failure')
        require(np.array_equal(y,[r['ground_truth'] for r in ids]), 'MAT/event labels differ')
        saved_a = np.asarray(params['ea_matrices'][str(subject)], dtype=np.float64)
        # Audit only the documented calibration block; inference always uses saved_a.
        calibration_a = ea_matrix(uv[:24], params['ea_calibration']['eps'])
        ea_error = float(np.max(np.abs(saved_a-calibration_a)))
        require(ea_error < 1e-8, f'A{subject:02d}: saved EA mismatches calibration: {ea_error}')
        model_x = apply_ea(uv[24:],saved_a)[:,None] / np.float32(params['input_scale'])
        model_x = np.asarray(model_x,dtype=np.float32)
        q = quantize_input(model_x,params['input_quant_scale'],params['input_zero_point'])
        recon = (q.astype(np.float32)-params['input_zero_point'])*np.float32(params['input_quant_scale'])
        require(np.array_equal(quantize_input(recon,params['input_quant_scale'],params['input_zero_point']),q), 'INT8/float32 reconstruction is lossy')
        ref = refs['subjects'][str(subject)]
        require(ref['trial'] == list(range(24,144)), 'Reference trial ordinals mismatch')
        require(np.array_equal(y[24:],ref['y_true']), 'Saved reference labels mismatch')
        require(sum(r['artifact'] for r in ids[24:]) == ARTIFACTS[subject-1], 'Artifact metadata mismatch')
        ref_logits = np.asarray(ref['logits'],dtype=np.float32)
        require(ref_logits.shape == (120,2) and np.isfinite(ref_logits).all(), 'Reference shape/nonfinite failure')
        for j, row in enumerate(ids[24:]):
            i = len(records)
            tid = f"S{subject}/1test/run{row['run']}/cue{row['cue_sample']}"
            records.append({'index':i,'trial_id':tid,'subject':subject,'ground_truth':int(y[j+24]),
                'artifact':row['artifact'],'source_trial':row,'input_sha256':digest(q[j].tobytes()),
                'source_file':path.name,'source_sha256':sha})
        inputs.append(q); normalized.append(model_x); eeg.append(uv[24:]); labels.append(y[24:]); references.append(ref_logits)
        sources.append({'file':path.name,'sha256':sha,'bytes':path.stat().st_size})
        per_subject.append({'subject':subject,'eligible':120,'class_counts':np.bincount(y[24:],minlength=2).tolist(),
                            'artifact_flagged':ARTIFACTS[subject-1], 'ea_calibration_max_abs_error':ea_error,
                            'saved_qdq_correct':int(np.sum(ref_logits.argmax(1)==y[24:]))})
        print(f'A{subject:02d}: 120 genuine inputs, labels verified, EA error {ea_error:.3g}', flush=True)
    q = np.concatenate(inputs); x = np.concatenate(normalized); uv = np.concatenate(eeg); y = np.concatenate(labels); ref_logits = np.concatenate(references)
    require(len(records) == len({r['trial_id'] for r in records}) == 1080, 'Duplicate/missing trial identities')
    with np.load(bundle/'test_vectors.npz',allow_pickle=False) as z:
        require(np.array_equal(q[240:260],z['x']), f'Known 20 exported inputs differ in {int(np.sum(q[240:260] != z["x"]))} elements')
        require(np.array_equal(y[240:260],z['y']), 'Known 20 vector labels differ')
        require(np.array_equal(ref_logits[240:260],z['logits']), 'Known 20 vector desktop logits differ')
    # Round trip across every representable INT8 value, independent of observed signals.
    all_q=np.arange(-128,128,dtype=np.int16).astype(np.int8)
    dq=(all_q.astype(np.float32)-params['input_zero_point'])*np.float32(params['input_quant_scale'])
    require(np.array_equal(quantize_input(dq,params['input_quant_scale'],params['input_zero_point']),all_q), 'INT8 scalar round-trip fails')
    payloads={'inputs.bin':q.tobytes(order='C'),'eeg.f32':uv.astype('<f4').tobytes(order='C')}
    meta={'schema':2,'model_id':'moamen-v4-wide32-ft-seed0','experiment_id':'multisubject-session2-heldout-v1',
          'model_sha256':MODEL_SHA,'input_shape':[1,1,22,256],'input_quant_scale':params['input_quant_scale'],
          'input_zero_point':params['input_zero_point'],'sample_rate_hz':125,'samples_per_channel':256,
          'channel_order':CHANNELS,'trials':records,'assets':{name:digest(b) for name,b in payloads.items()},
          'preprocessing':'Prerecorded BCI IV 2a Session2: CAR, continuous causal 7–30 Hz, decimated 125 Hz. Display is pre-EA microvolts; model uses saved participant EA and global scale.',
          'display_units':'uV','display_processing':'CAR-filtered-decimated before Euclidean Alignment and quantization',
          'training_unseen':True,'previously_evaluated':True,'artifact_policy':'Retain all 94 flagged eligible recordings',
          'source_revision':REVISION,'preprocess_params_sha256':PARAMS_SHA,'sources':sources}
    payloads['metadata.json']=json_bytes(meta)
    for folder in [args.out,args.frontend_out]:
        for name,b in payloads.items():save_same(folder/name,b)
    import io
    buf=io.BytesIO()
    np.savez_compressed(buf,inputs_int8=q,normalized_float32=x,filtered_uv_float32=uv,labels=y,
                        qdq_logits=ref_logits,subject_ids=np.array([r['subject'] for r in records],dtype=np.int64),
                        session_trial_ordinals=np.array([r['source_trial']['session_trial'] for r in records],dtype=np.int64))
    save_same(args.out/'windows.npz',buf.getvalue())
    summary={'schema':1,'source_revision':REVISION,'model_sha256':MODEL_SHA,'params_sha256':PARAMS_SHA,
             'desktop_reference_sha256':REFERENCE_SHA,'exported_vectors_sha256':VECTORS_SHA,
             'eligible_trials':1080,'class_counts':np.bincount(y,minlength=2).tolist(),'artifact_flagged':94,
             'original_ids_unique':True,'all_labels_match_saved_reference':True,'known_20_input_elements_exact':20*5632,
             'known_20_logits_exact':True,'all_observed_int8_reconstruction_exact':True,'all_256_int8_codes_roundtrip_exact':True,
             'saved_qdq_correct':int(np.sum(ref_logits.argmax(1)==y)),'saved_qdq_accuracy':float(np.mean(ref_logits.argmax(1)==y)),
             'reference_scope':'Saved desktop Q/DQ results; this preparation does not run model inference.',
             'assets':{n:{'bytes':len(b),'sha256':digest(b)} for n,b in payloads.items()},
             'windows_sha256':digest(buf.getvalue()),'per_subject':per_subject,'sources':sources,
             'upstream_source_sha256':SOURCE_SHA,'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','mne','moabb']},
             'license':'Local preparation only; redistribution is not authorized by this script.'}
    save_same(args.out/'validation.json',json_bytes(summary))
    if args.report:save_same(args.report,json_bytes(summary))
    print(json.dumps({k:summary[k] for k in ['eligible_trials','artifact_flagged','known_20_input_elements_exact','saved_qdq_accuracy']}))
    return summary

def main():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--bundle',type=Path,default=root/'artifacts/multisubject-v4/bundle')
    p.add_argument('--upstream-source',type=Path,default=root/'artifacts/multisubject-v4/upstream')
    p.add_argument('--out',type=Path,default=root/'artifacts/multisubject-v4/data')
    p.add_argument('--frontend-out',type=Path,default=root/'frontend/multisubject/data')
    p.add_argument('--report',type=Path,default=root/'tools/multisubject_trials/validation/dataset.json')
    args=p.parse_args()
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/edgebci-multisubject-mpl')
    sys.dont_write_bytecode=True
    prepare(args)
if __name__=='__main__':main()
