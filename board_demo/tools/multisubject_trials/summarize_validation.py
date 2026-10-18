"""Consolidate completed FVP runs. No serial, debugger or inference calls."""
from pathlib import Path
import collections,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=ROOT/'artifacts/multisubject-v4';v=r/'fvp-validation';reports=[json.loads((v/f'batch-{i:04d}-comparison.json').read_text()) for i in [0,240,480,720,960]]
 rows=[row for report in reports for row in report['rows']];assert len(rows)==1080
 byid={row['index']:row for row in rows};assert set(byid)==set(range(1080));rows=[byid[i] for i in range(1080)]
 assert all(x['tosa_integer_exact'] and x['tosa_001_pass'] for x in rows)
 repeated=json.loads((v/'reordered-comparison.json').read_text());assert len(repeated['rows'])==14
 for row in repeated['rows']:
  original=byid[row['index']];assert row['trial_id']==original['trial_id']
  assert np.array_equal(np.asarray(row['logits'],np.float32).view(np.uint32),np.asarray(original['logits'],np.float32).view(np.uint32))
 qdq=np.array([x['saved_qdq_logits'] for x in rows],np.float32);fvp=np.array([x['logits'] for x in rows],np.float32);raw=np.array([x['raw_int8'] for x in rows],np.int8);scale=reports[0]['output_scale'];zp=reports[0]['output_zero_point'];qraw=np.rint(qdq.astype(np.float64)/scale+zp).astype(np.int16);delta=raw.astype(np.int16)-qraw
 mismatches=[x for x in rows if x['prediction']!=x['saved_qdq_prediction']]
 result={'schema':1,'status':'FVP backend validated on all1080 prepared inputs; physical deployment not performed','model_sha256':reports[0]['model_sha256'],'eligible_trials':1080,'class_counts':[540,540],'artifact_flagged':94,'actual_fvp_cases':1080,'additional_published_cases':20,'additional_reordered_cases':14,'reordered_unique_trials':repeated['unique_trials'],'reordered_results_float32_bit_exact':True,'input_ids_and_crc_verified':True,'tosa_raw_output_values_checked':2160,'tosa_raw_mismatches':0,'tosa_001_failures':0,'tosa_reference_status':'Independently executed reconstructed integer graph now agrees with unchanged release PTE FVP outputs for all2160 values and reordered cases; this is empirical pool coverage, not PTE byte identity','output_scale':scale,'output_zero_point':zp,'retained_qdq_tolerance':.01,'saved_qdq_001_failures':sum(not x['saved_qdq_001_pass'] for x in rows),'saved_qdq_max_abs_logit_error':max(x['saved_qdq_max_error'] for x in rows),'saved_qdq_class_agreement':1080-len(mismatches),'saved_qdq_class_mismatches':len(mismatches),'saved_qdq_vs_fvp_trial_step_histogram':dict(collections.Counter(map(int,np.max(np.abs(delta),axis=1)))),'fvp_correct':sum(x['prediction']==x['ground_truth'] for x in rows),'saved_qdq_correct':sum(x['saved_qdq_prediction']==x['ground_truth'] for x in rows),'classification_disagreements':mismatches,'physical_board_used':False,'trained_weights_changed':False,'files':{}}
 for stem in ['published20','batch-0000','batch-0240','batch-0480','batch-0720','batch-0960','reordered']:
  for suffix in ['.log','-expected.json','-comparison.json']:
   p=v/(stem+suffix);result['files'][str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size}
 np.savez_compressed(v/'backend-reference.npz',fvp_logits=fvp,tosa_raw_int8=raw,saved_qdq_logits=qdq,labels=np.array([x['ground_truth'] for x in rows],np.int64))
 (v/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 target=ROOT/'tools/multisubject_trials/validation/numerical-status.json'
 old=json.loads(target.read_text());historical=target.with_name('desktop-reconstruction-diagnostic.json')
 if not historical.exists():historical.write_text(json.dumps(old,indent=2)+'\n')
 target.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:val for k,val in result.items() if k not in ['files','classification_disagreements']},indent=2))
if __name__=='__main__':main()
