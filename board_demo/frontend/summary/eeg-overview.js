/* EEG signal overview: per-channel RMS over the EEG windows of unique, successfully completed recordings.
 * Pure arithmetic over provider samples; never touches logits, predictions, normalized inputs or scalp colors.
 * Separate from classification metrics. Repeat evaluations of a recording are counted once.
 *
 * Accumulator: createAccumulator({units, unitsVerified, label})
 *   add(sourceTrialId, channels, {artifact})  channels = {name: finite samples[]} | null; returns true when newly counted
 *   result()                                  frozen overview, see below
 * Overview: {available, reason, label, units, recordings:{total, withEEG, missing, used}, artifact:{known, flagged},
 *            channels:[{name, rms, samples}] (desc by rms), highest, lowest, excludedChannels, sampleCount}
 * Rule: a channel is ranked only when it has valid (finite, non-flat) samples in every recording that has EEG, so every
 * compared channel covers the same recordings; otherwise it is listed in excludedChannels. Fewer than two rankable
 * channels, no EEG, or unverified units -> available:false with a reason. Missing channels are never treated as zeros.
 */
(function(root){'use strict';
function isList(a){return Array.isArray(a)||(ArrayBuffer.isView(a)&&!(a instanceof DataView));}
function validSamples(a){if(!isList(a)||a.length<2)return false;let min=Infinity,max=-Infinity;for(let i=0;i<a.length;i++){const x=a[i];if(typeof x!=='number'||!Number.isFinite(x))return false;if(x<min)min=x;if(x>max)max=x;}return max>min;}
function channelRMS(a){if(!validSamples(a))return null;let s=0;for(let i=0;i<a.length;i++)s+=a[i]*a[i];return Math.sqrt(s/a.length);}
function createAccumulator({units=null,unitsVerified=false,label='Prerecorded EEG · channel RMS'}={}){
 const seen=new Set(),stats=new Map();let withEEG=0,missing=0,flagged=0,artifactKnown=false;
 function add(sourceTrialId,channels,{artifact=null}={}){
  if(sourceTrialId===null||sourceTrialId===undefined||sourceTrialId==='')return false;const id=String(sourceTrialId);if(seen.has(id))return false;seen.add(id);
  const names=channels&&typeof channels==='object'?Object.keys(channels):[];
  if(!names.length){missing++;return true;}
  withEEG++;if(artifact===true||artifact===false){artifactKnown=true;if(artifact)flagged++;}
  for(const name of names){if(!stats.has(name))stats.set(name,{sumSq:0,count:0,valid:0});const st=stats.get(name),a=channels[name];if(!validSamples(a)){continue;}for(let i=0;i<a.length;i++)st.sumSq+=a[i]*a[i];st.count+=a.length;st.valid++;}
  return true;
 }
 function result(){
  const total=seen.size,base={label,units:units??null,recordings:Object.freeze({total,withEEG,missing,used:0}),artifact:Object.freeze({known:artifactKnown,flagged}),channels:Object.freeze([]),highest:null,lowest:null,excludedChannels:Object.freeze([]),sampleCount:0};
  const fail=reason=>Object.freeze({...base,available:false,reason});
  if(!units||!unitsVerified)return fail('EEG units not verified for this dataset');
  if(total===0)return fail('no completed recordings');
  if(withEEG===0)return fail('no EEG samples for the completed recordings');
  const ranked=[],excluded=[];for(const [name,st] of stats){if(st.valid===withEEG)ranked.push({name,rms:Math.sqrt(st.sumSq/st.count),samples:st.count});else excluded.push(name);}
  if(ranked.length<2){const partial=excluded.some(name=>stats.get(name).valid>0);return Object.freeze({...base,excludedChannels:Object.freeze(excluded),available:false,reason:partial?'channel coverage differs across recordings':'fewer than two channels with valid samples'});}
  ranked.sort((a,b)=>b.rms-a.rms||a.name.localeCompare(b.name));
  const channels=Object.freeze(ranked.map(c=>Object.freeze(c)));
  return Object.freeze({...base,available:true,reason:null,recordings:Object.freeze({total,withEEG,missing,used:withEEG}),channels,highest:channels[0],lowest:channels[channels.length-1],excludedChannels:Object.freeze(excluded),sampleCount:channels.reduce((s,c)=>s+c.samples,0)});
 }
 return {add,result,has:id=>seen.has(String(id)),get size(){return seen.size;}};
}
function summarize(recordings,options){const acc=createAccumulator(options);for(const r of recordings||[])acc.add(r.sourceTrialId,r.channels,{artifact:r.artifact});return acc.result();}
root.EdgeEEGOverview={createAccumulator,summarize,channelRMS,validSamples};
if(typeof module!=='undefined')module.exports=root.EdgeEEGOverview;
})(globalThis);
