/* Same visual controls and APIs; model-specific metadata and real UART results only. */
(async()=>{
'use strict';if(!window.EdgeMultisubjectMode)return;
const $=id=>document.getElementById(id),profile=EdgeMultisubjectProfile.profile;
let controller=null,link=null,visualResolve=null,lastFrames=[];const retainedSessions=[];
function notice(message){$('transportStatus').textContent='Multi-subject v4 · '+message;}
function cancel(){if(visualResolve){visualResolve();visualResolve=null;}}
function visualize(record,trial){return new Promise(resolve=>{
 visualResolve=resolve;const start=ArmEEGDemo.getState().playback.index;
 ArmEEGHardware.receive({...record,eeg_channels:controller.dataset.channels(trial.index)});
 function check(){if(visualResolve!==resolve)return;if(ArmEEGDemo.getState().playback.index>start){visualResolve=null;resolve();}else requestAnimationFrame(check);}requestAnimationFrame(check);
});}
function state(s){
 if(link&&link.connected===false)ArmEEGHardware.disconnected();
 ArmEEGDemo.setPlaying(!s.paused&&!s.error&&!s.ended);$('playBtn').textContent=s.ended?'Run ended':s.paused?'▶ Play':'Ⅱ Pause';
 notice((s.error||!s.ready?'Stopped: '+(s.error||'not connected'):s.ended?'Run ended':s.paused?'Paused'+(s.pending?' · finishing request '+s.pending.request:''):s.pending?'Waiting for hardware request '+s.pending.request:'Playing')+' · '+s.total+' predictions · '+s.unique+' unique/'+profile.trial_count+' · attempts '+s.correct+'/'+s.total+' · unique-first '+s.unique_correct+'/'+s.unique+' · repeated disagreements '+s.repeat_disagreements);
}
function capture(){return {schema_version:1,model:profile,retainedSessions,firmware:controller?.firmware,seed:controller?.seed,...controller?.summary(),frames:link?.frames??lastFrames};}
function exportCapture(){
 if(!controller)return;const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(capture(),null,2)],{type:'application/json'}));a.download='edgebci-multisubject-hardware-stream.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}
function completed(record){
 $('capturedResults').textContent=controller.completedRecords.map(x=>JSON.stringify(x)).join('\n');
 $('latestResult').textContent='S'+record.subject_id+' · '+record.trial_id+' · actual hardware class '+record.class_id+' · logits ['+record.logits.join(', ')+']';
 window.dispatchEvent(new CustomEvent('edgebci:trial-completed',{detail:record}));
}
// Correct the prerecorded metadata while preserving Jihyung's renderer, layout and colors.
const drawEEG=ArmEEGDemo.onEEGFrame;
ArmEEGDemo.onEEGFrame=(channels,metadata={})=>{
 drawEEG(channels,metadata);
 if(metadata.origin!=='prerecorded')return;
 const label='Prerecorded EEG · '+metadata.trial_id+' · CAR-filtered µV';
 $('mapSource').textContent=label;
 $('mapSource').setAttribute('title',metadata.trial_id+' · 125 Hz, 2.048 s · common average reference, 7–30 Hz causal filtering, phase-0 decimation · µV before alignment/scaling/quantization · relative channel RMS; not cortical source localization or a clinical measurement');
 $('traces').setAttribute('aria-label',label+' · 22 channels');document.querySelector('.trace-note').textContent=label;
};
$('metadataProfile').disabled=true;$('metadataProfile').innerHTML='<option>Participants 1–9 · Session 2 · 1,080 verified recordings</option>';
$('participant').disabled=true;$('participant').innerHTML='<option>Participants 1–9 · held-out Session 2</option>';
$('replayBtn').textContent='Open four-trial replay';$('mockBtn').textContent='Export streaming capture';
$('hardwareControls').querySelector('section > p:not([id])').textContent='Prerecorded Session 2 EEG · actual E8 inference only. One request at a time. Pause stops new requests; Reset starts a new sequence without resetting the board. Unique and repeated recordings are counted separately.';
$('replayBtn').onclick=async()=>{if(link)await link.disconnect();location.href=location.pathname+'?v=eeg-1';};
$('mockBtn').onclick=exportCapture;
$('connectBtn').onclick=async()=>{
 if(link){notice('Already connected or unresolved; disconnect before reconnecting');return;}
 try{
  notice('Loading and verifying 1,080 prerecorded EEG windows · no handshake yet');const dataset=await EdgeMultisubjectDataset.load();
  if(controller)retainedSessions.push({...controller.summary(),firmware:controller.firmware,frames:lastFrames});lastFrames=[];
  link=new EdgeStream.SerialLink();controller=new EdgeMultisubject.Controller({dataset,transport:link,visualize,cancelVisualization:cancel,onState:state,onRecord:completed});window.EdgeStreamingController=controller;
  notice('Select SEGGER UART4 in Chrome’s port chooser');await link.connect();ArmEEGHardware.begin();ArmEEGHardware.disconnected();ArmEEGDemo.setPlaying(false);
  notice('UART open · verifying multi-subject firmware');await controller.handshake();ArmEEGHardware.begin();ArmEEGHardware.connected();document.querySelector('.pill.mock').textContent='Physical E8 · prerecorded EEG';notice('Multi-subject firmware verified · press Play');
 }catch(e){if(controller)controller.fail(e);ArmEEGHardware.disconnected();notice('Connection failed: '+e.message+' · stopped; no automatic retry');}
};
$('disconnectBtn').onclick=async()=>{
 controller?.pause();cancel();try{if(link){lastFrames=link.frames;await link.disconnect();}link=null;ArmEEGHardware.disconnected();notice('Disconnected · capture retained · no automatic restart');}catch(e){notice('Disconnect failed: '+e.message+' · capture retained');}
};
$('playBtn').addEventListener('click',event=>{
 event.stopImmediatePropagation();if(!controller||!controller.ready){notice(controller?.failed?'Connection failed: '+controller.failed.message:'Multi-subject firmware not verified · use Connect UART before Play');return;}
 if(controller.paused)controller.play().catch(e=>notice(e.message));else controller.pause();
},true);
$('resetBtn').addEventListener('click',async event=>{
 event.stopImmediatePropagation();if(!controller)return;
 try{await controller.reset(crypto.getRandomValues(new Uint32Array(1))[0]);ArmEEGHardware.begin();ArmEEGDemo.setPlaying(false);$('capturedResults').textContent='Previous captures retained for export';$('latestResult').textContent='New seeded sequence · press Play';}catch(e){notice(e.message);}
},true);
window.EdgeMultisubjectSession={
 getCompletedTrials:()=>controller?.completedRecords.slice()??[],getCapture:capture,exportCapture,
 async endRun(){if(!controller)return null;const summary=await controller.endRun();window.dispatchEvent(new CustomEvent('edgebci:run-ended',{detail:summary}));return summary;}
};
ArmEEGHardware.begin();ArmEEGHardware.disconnected();ArmEEGDemo.setPlaying(false);notice('Connect matching multi-subject firmware before Play');
})();
