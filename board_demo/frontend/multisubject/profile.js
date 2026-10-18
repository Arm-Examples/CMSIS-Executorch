/* Unfused Moamen v4 profile; the existing Model 2 protocol remains unchanged. */
(function(root){
'use strict';
const profile=Object.freeze({
 id:'moamen-v4-wide32-ft-seed0', experiment_id:'multisubject-session2-heldout-v1',
 model_sha256:'6caa8a9876d3cd0c36b39b3d4474a0fe7dbdbfb32a4d6f80ed4776623a19e02e',
 input_shape:Object.freeze([1,1,22,256]), input_quant_scale:0.06955837458372116,
 input_zero_point:-12, wire_bytes:5632, wire_dtype:'int8', pte_input_dtype:'float32',
 sample_rate_hz:125, samples_per_channel:256, trial_count:1080,
 channel_order:Object.freeze(['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz'])
});
function protocol(base){
 function ready(f){
  const p=f.payload;
  if(p.length!==69||f.request!==0||f.trial!==0xffffffff)throw Error('Multi-subject firmware handshake format mismatch');
  const v=new DataView(p.buffer,p.byteOffset,p.byteLength),hex=Array.from(p.subarray(0,32),b=>b.toString(16).padStart(2,'0')).join('');
  if(hex!==profile.model_sha256||v.getFloat32(32,true)!==Math.fround(profile.input_quant_scale)||v.getInt32(36,true)!==profile.input_zero_point||v.getUint32(40,true)!==profile.wire_bytes||profile.input_shape.some((n,i)=>v.getUint16(44+2*i,true)!==n)||p[52]!==1)throw Error('Multi-subject firmware/model input contract mismatch');
  return {...profile,build:new TextDecoder().decode(p.subarray(53)).replace(/\0+$/,'')};
 }
 return {...base,model:profile.model_sha256,ready};
}
root.EdgeMultisubjectProfile={profile,protocol};
if(root.EdgeMultisubjectMode)root.EdgeStreamProtocol=protocol(root.EdgeStreamProtocol);
if(typeof module!=='undefined')module.exports={profile,protocol};
})(globalThis);
