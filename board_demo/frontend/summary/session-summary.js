/* Session summary: model-independent metrics over completed, validated inference results.
 * Pure functions only. No DOM, no hardware, no dataset access. Loaded in the browser and by Node tests.
 *
 * Session metadata (all optional unless noted):
 *   sessionId, modelId, modelName, experimentName, datasetId, datasetName, subjectId,
 *   modelHash, modelHashVerified, firmwareBuild, seed, source ('live-hardware' | 'saved-uart-replay' | 'synthetic-preview'),
 *   status ('completed' | 'interrupted' | 'empty'), interruption {message, unresolvedRequest, trialId}, note,
 *   startedAt, endedAt (ms since epoch, Date or ISO string), durationLabel,
 *   classLabels (required): {index: name} | [name, ...] | [{index, name, display}], classNamesVerified (bool).
 *
 * Completed result (one per successful hardware inference attempt):
 *   trialId (required, unique attempt id), sourceTrialId (required, stable recording identity that already
 *   includes dataset/subject/session, never a bare array index), subjectId, groundTruth, predictedClass,
 *   logits, roundTripMs (browser request-to-result time; only when measured).
 */
(function(root){'use strict';
function normalizeClassLabels(labels){
 let list=[];
 const pick=(v,index)=>({index,name:String(v.name??index),display:v.display,short:v.short,order:Number.isFinite(v.order)?v.order:null});
 if(Array.isArray(labels))list=labels.map((v,i)=>typeof v==='object'&&v!==null?pick(v,Number(v.index??i)):{index:i,name:String(v),order:null});
 else if(labels&&typeof labels==='object')list=Object.keys(labels).map(k=>{const v=labels[k];return typeof v==='object'&&v!==null?pick(v,Number(k)):{index:Number(k),name:String(v),order:null};});
 list=list.filter(c=>Number.isInteger(c.index)).sort((a,b)=>a.index-b.index);
 if(new Set(list.map(c=>c.index)).size!==list.length)throw Error('Duplicate class index in classLabels');
 // order: display position (metadata-defined, e.g. Left before Right); calculations stay in index order.
 return list.map(c=>{const display=c.display??c.name;return Object.freeze({index:c.index,name:c.name,display,short:c.short??display,order:c.order??c.index});});
}
function toMs(value){if(value===null||value===undefined||value==='')return null;if(value instanceof Date)return Number.isFinite(value.getTime())?value.getTime():null;if(typeof value==='number')return Number.isFinite(value)?value:null;const t=Date.parse(value);return Number.isFinite(t)?t:null;}
function classesFromResults(results){const seen=new Set();for(const r of results||[]){if(r&&Number.isInteger(r.groundTruth))seen.add(r.groundTruth);if(r&&Number.isInteger(r.predictedClass))seen.add(r.predictedClass);}return [...seen].sort((a,b)=>a-b).map(i=>({index:i,name:'class '+i,display:'Class '+i}));}
/* options.eeg: an EdgeEEGOverview result computed separately from the classification metrics (or null). */
function build(session={},results=[],{eeg=null}={}){
 const labelsGiven=session.classLabels!==undefined&&session.classLabels!==null;
 const classes=normalizeClassLabels(labelsGiven?session.classLabels:classesFromResults(results));
 const valid=new Set(classes.map(c=>c.index));
 const seen=new Set(),firstBySource=new Map();let duplicatesIgnored=0,excluded=0;const accepted=[];
 for(const r of results||[]){
  if(!r||typeof r!=='object'){excluded++;continue;}
  if(r.trialId===undefined||r.trialId===null||r.trialId===''){excluded++;continue;}
  const trialId=String(r.trialId);if(seen.has(trialId)){duplicatesIgnored++;continue;}seen.add(trialId);
  if(r.sourceTrialId===undefined||r.sourceTrialId===null||r.sourceTrialId===''||!valid.has(r.groundTruth)||!valid.has(r.predictedClass)){excluded++;continue;}
  const sourceTrialId=String(r.sourceTrialId),isRepeated=firstBySource.has(sourceTrialId);
  const row=Object.freeze({trialId,sourceTrialId,subjectId:r.subjectId??null,groundTruth:r.groundTruth,predictedClass:r.predictedClass,logits:Array.isArray(r.logits)?Object.freeze(r.logits.map(Number)):null,isCorrect:r.groundTruth===r.predictedClass,isRepeated,roundTripMs:Number.isFinite(r.roundTripMs)?r.roundTripMs:null});
  if(!isRepeated)firstBySource.set(sourceTrialId,row);
  accepted.push(row);
 }
 const total=accepted.length,correct=accepted.filter(r=>r.isCorrect).length;
 const unique=[...firstBySource.values()],uniqueCorrect=unique.filter(r=>r.isCorrect).length;
 const perClass=classes.map(c=>{const rows=accepted.filter(r=>r.groundTruth===c.index),k=rows.filter(r=>r.isCorrect).length;return Object.freeze({index:c.index,name:c.name,display:c.display,short:c.short,total:rows.length,correct:k,incorrect:rows.length-k,accuracy:rows.length?k/rows.length:null});});
 const confusion=Object.freeze(classes.map(a=>Object.freeze(classes.map(p=>accepted.filter(r=>r.groundTruth===a.index&&r.predictedClass===p.index).length))));
 const timed=accepted.filter(r=>r.roundTripMs!==null).map(r=>r.roundTripMs);
 const latency=timed.length?Object.freeze({count:timed.length,meanMs:timed.reduce((s,x)=>s+x,0)/timed.length,minMs:Math.min(...timed),maxMs:Math.max(...timed)}):null;
 const startedAt=toMs(session.startedAt),endedAt=toMs(session.endedAt);
 const duration=startedAt!==null&&endedAt!==null&&endedAt>=startedAt?Object.freeze({ms:endedAt-startedAt,label:session.durationLabel||'Elapsed session time · first Play to End Run, pauses included'}):null;
 const status=session.status||(total?'completed':'empty');
 const meta=Object.freeze({
  sessionId:session.sessionId??null,modelId:session.modelId??null,modelName:session.modelName??null,experimentName:session.experimentName??null,
  datasetId:session.datasetId??null,datasetName:session.datasetName??null,subjectId:session.subjectId??null,
  modelHash:session.modelHash??null,modelHashVerified:!!session.modelHashVerified,firmwareBuild:session.firmwareBuild??null,seed:session.seed??null,
  source:session.source||'live-hardware',status,interruption:session.interruption?Object.freeze({...session.interruption}):null,note:session.note??null,
  startedAt,endedAt,classNamesVerified:labelsGiven&&session.classNamesVerified!==false});
 return Object.freeze({session:meta,classes:Object.freeze(classes),results:Object.freeze(accepted),
  metrics:Object.freeze({total,correct,incorrect:total-correct,accuracy:total?correct/total:null,unique:Object.freeze({count:unique.length,correct:uniqueCorrect,accuracy:unique.length?uniqueCorrect/unique.length:null}),repeated:total-unique.length,perClass:Object.freeze(perClass),confusion,includesRepeats:true,duplicatesIgnored,excluded,latency,duration}),
  eeg:eeg&&typeof eeg==='object'?Object.freeze(eeg):null,
  generatedAt:Date.now()});
}
/* Stable recording identity for a streaming record: dataset id + metadata trial_id (subject/session/run/cue). */
function sourceIdentity(r,{datasetId=null,dataset=null}={}){
 let recording=r?.trial_id??null;
 if(recording===null&&dataset){const meta=(dataset.trials||[]).find(t=>t&&t.index===r?.trial),source=meta?.source_trial;if(source)recording=`S${source.subject}/${source.session}/index${r.trial}`;}
 return recording===null||recording===undefined?null:(datasetId?datasetId+'|':'')+recording;
}
/* Adapter for the streaming controller's frozen records (frontend/streaming/controller.js).
 * trialId = session nonce + request id (one per inference attempt); sourceTrialId = dataset id + metadata trial_id.
 * roundTrips: Map('<session>:<request>' -> ms) measured in the browser, optional.
 */
function adaptStreamingRecords(records,{datasetId=null,dataset=null,roundTrips=null}={}){
 const byTrialId=new Map();for(const t of dataset?.trials??[])if(t&&t.trial_id)byTrialId.set(t.trial_id,t);
 return (records||[]).map(r=>{
  const trialId=String(r.session??'')+':'+String(r.request??'');
  const meta=r.trial_id?byTrialId.get(r.trial_id):null,source=meta?.source_trial??null;
  return {trialId,sourceTrialId:sourceIdentity(r,{datasetId,dataset}),subjectId:source?.subject??r.subject??null,groundTruth:r.ground_truth,predictedClass:r.class_id,logits:r.logits,roundTripMs:roundTrips?.get(trialId)??null};
 });
}
root.EdgeSessionSummary={build,adaptStreamingRecords,sourceIdentity,normalizeClassLabels,toMs};
if(typeof module!=='undefined')module.exports=root.EdgeSessionSummary;
})(globalThis);
