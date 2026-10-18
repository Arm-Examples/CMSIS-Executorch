"""Local dataset regression tests; requires prepare_dataset.py's genuine assets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'artifacts/multisubject-v4/data'
BUNDLE = ROOT/'artifacts/multisubject-v4/bundle'
spec = importlib.util.spec_from_file_location('prepare_dataset',Path(__file__).with_name('prepare_dataset.py'))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

class DatasetIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads((DATA/'metadata.json').read_text())
        cls.params=json.loads((BUNDLE/'preprocess_params.json').read_text())
        cls.refs=json.loads((BUNDLE/'offline_int8_predictions.json').read_text())
        with np.load(DATA/'windows.npz',allow_pickle=False) as z:
            cls.arrays={k:z[k].copy() for k in z.files}

    def test_complete_pool_and_identity(self):
        m=self.meta; trials=m['trials']
        self.assertEqual(m['schema'],2)
        self.assertEqual(m['model_sha256'],prep.MODEL_SHA)
        self.assertEqual(m['channel_order'],prep.CHANNELS)
        self.assertEqual(m['input_shape'],[1,1,22,256])
        self.assertEqual(len(trials),1080)
        self.assertEqual(len({t['trial_id'] for t in trials}),1080)
        self.assertEqual([t['index'] for t in trials],list(range(1080)))
        for s in range(1,10):
            group=trials[(s-1)*120:s*120]
            self.assertEqual([t['source_trial']['session_trial'] for t in group],list(range(25,145)))
            self.assertEqual(sum(t['artifact'] for t in group),prep.ARTIFACTS[s-1])
            self.assertEqual(np.bincount([t['ground_truth'] for t in group]).tolist(),[60,60])
            for t in group:
                r=t['source_trial']
                self.assertEqual(r['subject'],s)
                self.assertEqual(t['subject'],s)
                self.assertEqual(r['session'],'1test')
                self.assertEqual(t['trial_id'],f"S{s}/1test/run{r['run']}/cue{r['cue_sample']}")
                self.assertEqual(t['artifact'],r['artifact'])
                self.assertEqual(r['cue_sample'],r['mat_marker_sample_1based']-1+500)
                self.assertEqual(t['ground_truth'],r['ground_truth'])
                self.assertGreaterEqual(r['run'],1)  # first MI run supplied adaptation's first24
                self.assertLess(r['run'],6)

    def test_browser_bytes_and_trial_integrity(self):
        for name in ['inputs.bin','eeg.f32']:
            blob=(DATA/name).read_bytes()
            self.assertEqual(hashlib.sha256(blob).hexdigest(),self.meta['assets'][name])
            self.assertEqual(blob,(ROOT/'frontend/multisubject/data'/name).read_bytes())
        self.assertEqual((DATA/'metadata.json').read_bytes(),(ROOT/'frontend/multisubject/data/metadata.json').read_bytes())
        q=self.arrays['inputs_int8']
        self.assertEqual(q.shape,(1080,1,22,256)); self.assertEqual(q.dtype,np.int8)
        self.assertEqual(q.tobytes(),(DATA/'inputs.bin').read_bytes())
        for t in self.meta['trials']:
            self.assertEqual(prep.digest(q[t['index']].tobytes()),t['input_sha256'])
        self.assertEqual(self.arrays['filtered_uv_float32'].astype('<f4').tobytes(),(DATA/'eeg.f32').read_bytes())

    def test_display_windows_are_pre_ea_recordings(self):
        uv=self.arrays['filtered_uv_float32']; x=self.arrays['normalized_float32']
        self.assertEqual(uv.shape,(1080,22,256))
        self.assertTrue(np.isfinite(uv).all())
        self.assertEqual(self.meta['display_units'],'uV')
        for s in range(1,10):
            sl=slice((s-1)*120,s*120)
            a=np.asarray(self.params['ea_matrices'][str(s)],dtype=np.float64)
            aligned=np.einsum('cd,ndt->nct',a,uv[sl]).astype(np.float32)
            expected=aligned[:,None]/np.float32(self.params['input_scale'])
            np.testing.assert_array_equal(expected,x[sl])
            self.assertFalse(np.array_equal(uv[sl],x[sl,0]))

    def test_quantized_wire_preserves_model_input_codes(self):
        p=self.params; x=self.arrays['normalized_float32']; q=self.arrays['inputs_int8']
        scale=np.float32(p['input_quant_scale']); zp=p['input_zero_point']
        np.testing.assert_array_equal(np.clip(np.rint(x/scale)+zp,-128,127).astype(np.int8),q)
        reconstructed=(q.astype(np.float32)-zp)*scale
        np.testing.assert_array_equal(np.clip(np.rint(reconstructed/scale)+zp,-128,127).astype(np.int8),q)
        codes=np.arange(-128,128,dtype=np.int16).astype(np.int8)
        np.testing.assert_array_equal(np.rint(((codes.astype(np.float32)-zp)*scale)/scale)+zp,codes)

    def test_exported_known_twenty_are_bit_exact(self):
        with np.load(BUNDLE/'test_vectors.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(self.arrays['inputs_int8'][240:260],z['x'])
            np.testing.assert_array_equal(self.arrays['labels'][240:260],z['y'])
            np.testing.assert_array_equal(self.arrays['qdq_logits'][240:260],z['logits'])

    def test_saved_references_and_true_labels_stay_separate(self):
        correct=0
        for s in range(1,10):
            ref=self.refs['subjects'][str(s)]; sl=slice((s-1)*120,s*120)
            self.assertEqual(ref['trial'],list(range(24,144)))
            np.testing.assert_array_equal(self.arrays['labels'][sl],ref['y_true'])
            np.testing.assert_array_equal(self.arrays['qdq_logits'][sl],np.asarray(ref['logits'],dtype=np.float32))
            correct+=int(np.sum(np.argmax(ref['logits'],axis=1)==np.asarray(ref['y_true'])))
        self.assertEqual(correct,804)
        for t in self.meta['trials']:
            self.assertTrue({'prediction','logits','qdq_logits','reference_prediction'}.isdisjoint(t))
        self.assertTrue(self.meta['training_unseen'])
        self.assertTrue(self.meta['previously_evaluated'])

    def test_wrong_artifacts_and_changed_outputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'artifact'
            p.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'Provenance mismatch'):
                prep.check_file(p,prep.MODEL_SHA)
            with self.assertRaisesRegex(ValueError,'Refusing changed-file overwrite'):
                prep.save_same(p,b'replacement')
            self.assertEqual(p.read_bytes(),b'corrupted')
            prep.save_same(p,b'corrupted')

if __name__=='__main__':unittest.main(verbosity=2)
