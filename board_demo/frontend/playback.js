/* Visualization clock only: preserves immutable hardware results, never runs inference. */
(function(root){
'use strict';
function createPlayback({onTrial=()=>{},onPosition=()=>{},onComplete=()=>{},continuous=false}={}){
 const duration=1800, movement=1400;
 let results=[],index=0,active=null,time=0,finished=false;
 return {
  push(message){
   if(![0,1].includes(message.class_id))return false;
   const copy={...message,logits:Array.isArray(message.logits)?Object.freeze([...message.logits]):undefined};
   results.push(Object.freeze(copy));return true;
  },
  tick(dt){
   if(!active&&index<results.length){active=results[index];time=0;onPosition(.5);onTrial(active,index);}
   if(!active)return;
   time=Math.min(duration,time+dt);
   // Recenter for each trial so consecutive RIGHT results remain visibly distinct.
   onPosition(.5+(active.class_id===0?-.28:.28)*Math.min(1,time/movement));
   if(time===duration){index++;active=null;if(index===results.length&&(finished||(!continuous&&results.length>=4)))onComplete();}
  },
  finish(){finished=true;if(!active&&index===results.length)onComplete();},
  reset(){index=0;active=null;time=0;onPosition(.5);},
  clear(){results=[];finished=false;this.reset();},
  state(){return {results:results.map(m=>({...m,logits:m.logits&&[...m.logits]})),index,active_trial:active?.trial??null,trial_elapsed_ms:time,duration_ms:duration,pending:results.length-index,complete:index===results.length&&(finished||(!continuous&&results.length>=4))};}
 };
}
root.EdgePlayback={createPlayback};
if(typeof module!=='undefined')module.exports=root.EdgePlayback;
})(globalThis);
