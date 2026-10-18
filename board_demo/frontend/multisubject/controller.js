/* Reuses the proven one-request-at-a-time controller; adds model/session metadata. */
(function(root){
'use strict';
const base=root.EdgeStream||(typeof require==='function'?require('../streaming/controller.js'):null);
const profile=root.EdgeMultisubjectProfile?.profile||(typeof require==='function'?require('./profile.js').profile:null);
const P=root.EdgeStreamProtocol;
class Controller extends base.Controller {
 constructor(options){
  let owner;
  super({...options,onRecord:r=>owner.completed(r),visualize:(r,t)=>owner.ending?Promise.resolve():options.visualize?.(owner.completedRecords[owner.completedRecords.length-1],t)});
  owner=this;this.completedRecords=[];this.completedArchives=[];this.seen=new Set();this.timings=new Map();this.ended=false;this.ending=false;
  this.onCompleted=options.onRecord||(()=>{});this.clock=options.clock||(()=>performance.now());
 }
 state(){return {...super.state(),model_id:profile.id,experiment_id:profile.experiment_id,pool_size:profile.trial_count,ended:!!this.ended};}
 async write(bytes){
  if(bytes.length>1){const f=P.decode(bytes.subarray(0,-1));if(f.type===P.T.INFER)this.timings.set(f.request,{sent:this.clock(),ack:null,result:null});}
  return super.write(bytes);
 }
 accept(f){
  if(f.session===this.session&&this.pending&&f.request===this.pending.request&&f.trial===this.pending.trial.index){const t=this.timings?.get(f.request);if(t&&f.type===P.T.ACCEPTED)t.ack=this.clock();if(t&&f.type===P.T.RESULT)t.result=this.clock();}
  return super.accept(f);
 }
 completed(record){
  const trial=this.dataset.trials[record.trial],t=this.timings.get(record.request),repeated=this.seen.has(record.trial_id);
  const detail=Object.freeze({...record,schema_version:1,model_id:profile.id,model_sha256:profile.model_sha256,experiment_id:profile.experiment_id,subject_id:trial.source_trial.subject,session_id:'1test',session_trial:trial.source_trial.session_trial,unique:!repeated,repeated,
   timing:Object.freeze({hardware_inference_ms:null,host_round_trip_ms:t?.result!=null?t.result-t.sent:null,host_ack_to_result_ms:t?.ack!=null&&t?.result!=null?t.result-t.ack:null,visualization_duration_ms:1800,timing_source:'host-observed UART, including transport and processing; hardware-only timing unavailable'})});
  this.seen.add(record.trial_id);this.timings.delete(record.request);this.completedRecords.push(detail);this.onCompleted(detail);
 }
 play(){if(this.ended||this.ending)return Promise.reject(Error('Run ended; Reset starts a new sequence'));return super.play();}
 async endRun(){
  this.ending=true;this.pause();this.cancelVisualization();if(this.task)await this.task;
  this.ending=false;this.ended=true;this.emit();return this.summary();
 }
 async reset(seed){
  if(this.ending)throw Error('End Run is finishing the in-flight request');
  const old=this.completedRecords.slice();await super.reset(seed);
  if(old.length)this.completedArchives.push(old);this.completedRecords=[];this.seen.clear();this.timings.clear();this.ended=false;this.emit();
 }
 summary(){return {model_id:profile.id,model_sha256:profile.model_sha256,experiment_id:profile.experiment_id,state:this.state(),records:this.completedRecords.slice(),archives:this.completedArchives.map(a=>a.slice())};}
}
root.EdgeMultisubject={Controller};
if(typeof module!=='undefined')module.exports={Controller};
})(globalThis);
