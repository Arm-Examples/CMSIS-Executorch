/* Verified model/experiment metadata for the session summary, keyed by the PTE SHA-256 that the
 * streaming firmware reports in READY and that frontend/streaming/protocol.js verifies.
 * Nothing here is inferred from predictions. Add one entry per deployed model; the summary never
 * hardcodes a model name, a subject, a recording count or a class order.
 */
(function(root){'use strict';
const models=[
 {
  modelSha256:'644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298',
  modelId:'model2-subject1-tinyeegnet-int8-ethosu85',
  modelName:'Model 2 · subject-specific TinyEEGNet (INT8, Ethos-U85)',
  experimentName:'Participant 1 held-out Session 2 · continuous random trials',
  datasetId:'bci-iv-2a/subject1/1test',
  datasetName:'BCI Competition IV 2a · Participant 1 · Session 2 (1test)',
  subjects:[1],
  sessions:['1test'],
  // artifacts/preprocess_params.json class_map and README "left_hand=0, right_hand=1".
  // order = display position in the summary (Left before Right); index = model output class, used for all counts.
  classLabels:{0:{name:'left_hand',display:'Left hand',short:'Left',order:0},1:{name:'right_hand',display:'Right hand',short:'Right',order:1}},
  // eeg.f32 is windows.npz filtered_uv_float32: "filtered microvolts, before normalization and int8 quantization"
  // (tools/model2_random_trials/prepare_session2.py -> prepare_streaming_assets.py). Not live acquisition.
  eeg:{units:'µV',unitsVerified:true,fs_hz:250,description:'filtered microvolts before normalization and int8 quantization',source:'tools/model2_random_trials/prepare_session2.py filtered_uv_float32 → prepare_streaming_assets.py eeg.f32'},
  verification:'PTE SHA-256 from the READY handshake and tools/model2_random_trials/validation/physical-streaming-build.json; class_map from artifacts/preprocess_params.json; dataset identity enforced by frontend/streaming/dataset.js.'
 }
];
function find(sha){return models.find(m=>m.modelSha256===sha)||null;}
/* Resolve verified metadata for a connected session. Returns null unless the firmware hash is known and the
 * loaded dataset agrees with the entry (same model hash, subjects and sessions). */
function resolve({firmware=null,dataset=null}={}){
 const sha=firmware?.model_sha256??dataset?.model_sha256??null;const entry=sha?find(sha):null;if(!entry)return null;
 if(dataset){
  if(dataset.model_sha256&&dataset.model_sha256!==entry.modelSha256)return null;
  const trials=Array.isArray(dataset.trials)?dataset.trials:[];
  if(trials.some(t=>!entry.subjects.includes(t?.source_trial?.subject)||(entry.sessions&&!entry.sessions.includes(t?.source_trial?.session))))return null;
 }
 return entry;
}
root.EdgeModelMetadata={models,find,resolve};
if(typeof module!=='undefined')module.exports=root.EdgeModelMetadata;
})(globalThis);
