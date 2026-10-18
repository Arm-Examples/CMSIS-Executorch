/* Genuine prerecorded EEG only; no model execution or reference predictions. */
(function(root){
'use strict';
const profile=root.EdgeMultisubjectProfile?.profile||(typeof require==='function'?require('./profile.js').profile:null);
function validateMetadata(meta){
 if(meta.schema!==2||meta.model_sha256!==profile.model_sha256||JSON.stringify(meta.input_shape)!==JSON.stringify(profile.input_shape)||meta.input_quant_scale!==profile.input_quant_scale||meta.input_zero_point!==profile.input_zero_point||meta.sample_rate_hz!==profile.sample_rate_hz||meta.samples_per_channel!==profile.samples_per_channel||JSON.stringify(meta.channel_order)!==JSON.stringify(profile.channel_order)||!Array.isArray(meta.trials)||meta.trials.length!==profile.trial_count)throw Error('Invalid multi-subject dataset/model contract');
 const ids=new Set(),ordinals=new Set(),counts=Array(10).fill(0);let artifacts=0;
 for(const [i,t] of meta.trials.entries()){
  const s=t.source_trial,subject=s?.subject;
  if(t.index!==i||!Number.isInteger(subject)||subject<1||subject>9||t.subject!==subject||s.session!=='1test'||!Number.isInteger(s.run)||s.run<0||!Number.isInteger(s.cue_sample)||s.cue_sample<0||!Number.isInteger(s.session_trial)||s.session_trial<25||s.session_trial>144||typeof s.artifact!=='boolean'||t.artifact!==s.artifact||![0,1].includes(t.ground_truth)||t.trial_id!==`S${subject}/1test/run${s.run}/cue${s.cue_sample}`||!/^[a-f0-9]{64}$/.test(t.input_sha256)||ids.has(t.trial_id)||ordinals.has(subject+'/'+s.session_trial))throw Error('Invalid multi-subject trial identity or eligibility');
  ids.add(t.trial_id);ordinals.add(subject+'/'+s.session_trial);counts[subject]++;artifacts+=Number(s.artifact);
 }
 if(counts.slice(1).some(n=>n!==120)||artifacts!==94)throw Error('Multi-subject pool count/artifact mismatch');
 if(!meta.assets||['inputs.bin','eeg.f32'].some(name=>!/^[a-f0-9]{64}$/.test(meta.assets[name])))throw Error('Missing EEG asset checksums');
 return meta;
}
async function load(base='multisubject/data/',deps={}){
 const fetcher=deps.fetch||root.fetch,hash=deps.hash||root.EdgeStream.hash;
 const json=await fetcher(base+'metadata.json');if(!json.ok)throw Error('Local multi-subject data is missing; prepare the verified recordings first');
 const meta=validateMetadata(await json.json());
 const buffers=await Promise.all(['inputs.bin','eeg.f32'].map(async name=>{const r=await fetcher(base+name);if(!r.ok)throw Error('Multi-subject EEG asset missing');const b=new Uint8Array(await r.arrayBuffer());if(await hash(b)!==meta.assets[name])throw Error('Multi-subject EEG asset integrity mismatch');return b;}));
 if(buffers[0].length!==profile.trial_count*profile.wire_bytes||buffers[1].length!==profile.trial_count*profile.wire_bytes*4)throw Error('Multi-subject EEG asset shape mismatch');
 const channelCache=new Map(),view=new DataView(buffers[1].buffer,buffers[1].byteOffset,buffers[1].byteLength);
 function validIndex(i){if(!Number.isInteger(i)||i<0||i>=meta.trials.length)throw Error('Unknown multi-subject trial index');}
 return {...meta,profile,input:async i=>{validIndex(i);return buffers[0].slice(i*profile.wire_bytes,(i+1)*profile.wire_bytes);},channels(i){
  validIndex(i);if(channelCache.has(i))return channelCache.get(i);const channels={};
  for(let c=0;c<22;c++){const a=Array.from({length:256},(_,j)=>view.getFloat32(((i*22+c)*256+j)*4,true));if(!a.every(Number.isFinite))throw Error('Non-finite prerecorded EEG samples');channels[meta.channel_order[c]]=Object.freeze(a);}
  Object.freeze(channels);channelCache.set(i,channels);return channels;
 }};
}
root.EdgeMultisubjectDataset={load,validateMetadata};
if(typeof module!=='undefined')module.exports={load,validateMetadata};
})(globalThis);
