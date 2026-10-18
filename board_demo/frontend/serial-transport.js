/* Read-only transport. Capture timing is independent of the visualization clock. */
(()=>{
'use strict';
const api=window.ArmEEGHardware, $=id=>document.getElementById(id);
let port=null,reader=null,task=null,replayTimer=null,busy=false;
let seen=new Set(),passed=null,ended=false,eot=false,link=false,mode='UART',notice='',captured=[],profile=null;
function status(){
 const run=passed===false?'Firmware execution failed':passed&&seen.size===4?'Complete · four trials · Execution_result: PASS':passed?'PASS received · incomplete capture ('+seen.size+'/4)':ended?'Execution ended · incomplete capture ('+seen.size+'/4)':'Captured '+seen.size+'/4 trials';
 $('transportStatus').textContent=mode+' · '+run+(eot?' · EOT':'')+' · '+(mode==='SAVED UART REPLAY'?'no board connected':link?'UART connected':'UART disconnected')+(notice?' · '+notice:'');
}
function begin(label){
 api.begin();document.querySelector('.pill.mock').textContent=label==='SAVED UART REPLAY'?'Saved replay':'Hardware mode';if(label==='SAVED UART REPLAY')$('boardStat').textContent='No board · saved replay';seen=new Set();passed=null;ended=false;eot=false;link=false;mode=label;notice='';captured=[];
 // Explicit deployment profile, not a label derived from logits or predicted class.
 profile=$('metadataProfile').value===window.EdgeTrialMetadata?.id?window.EdgeTrialMetadata:null;
 $('latestResult').textContent='No UART prediction received';$('capturedResults').textContent='No captured results';status();
}
function accept(m){
 if(m.type==='prediction'){
  if(ended||passed!==null){notice='Unexpected result after execution end ignored';status();return;}
  if(m.trial>3){notice='Unexpected trial '+m.trial+' (expected 0–3)';status();return;}
  if(seen.has(m.trial)){notice='Duplicate trial '+m.trial+' ignored';status();return;}
  seen.add(m.trial);
  const ground_truth=profile?.labels[m.trial]??null;
  const record={...m,logits:[...m.logits],ground_truth,metadata_profile:profile?.id??null};
  captured.push(record);api.receive(record);
  $('capturedResults').textContent=captured.map(r=>r.raw_line+' | recorded task: '+(r.ground_truth===0?'LEFT':r.ground_truth===1?'RIGHT':'unavailable')).join('\n');
  $('latestResult').textContent=`Captured trial ${m.trial} · ${m.class_id===0?'LEFT':'RIGHT'} · logits [${m.logits.join(', ')}] · see delayed playback`;
 }else if(m.type==='execution'){
  passed=m.pass;ended=true;window.ArmEEGDemo.finishCapture();
 }else if(m.type==='eot'){
  eot=true;ended=true;window.ArmEEGDemo.finishCapture();
 }
 status(); // Completion is a firmware state; it does not disconnect the transport or erase playback.
}
async function stop(){
 clearTimeout(replayTimer);replayTimer=null;
 if(reader)await reader.cancel().catch(()=>{});
 if(task){await task;task=null;}
 if(port){await port.close().catch(()=>{});port=null;}
 link=false;api.disconnected();status();
}
async function connect(){
 if(busy)return;busy=true;
 try{
  await stop();begin('UART');notice='Choose SEGGER UART4 · 115200 8N1';status();
  if(!navigator.serial)throw new Error('Web Serial requires desktop Chrome or Edge at localhost');
  port=await navigator.serial.requestPort();
  await port.open({baudRate:115200,dataBits:8,stopBits:1,parity:'none',flowControl:'none'});
  link=true;notice='Awaiting firmware output (no commands sent)';api.connected();status();
  const parser=EdgeUART.createParser(accept,e=>{notice=e;status();}),decoder=new TextDecoder();
  const currentReader=port.readable.getReader();reader=currentReader;
  task=(async()=>{
   try{while(true){const {value,done}=await currentReader.read();if(done)break;notice='';parser.push(decoder.decode(value,{stream:true}));}}
   catch(e){notice='UART read error: '+e.message;}
   finally{parser.push(decoder.decode());parser.finish();currentReader.releaseLock();reader=null;link=false;api.disconnected();status();}
  })();
 }catch(e){link=false;api.disconnected();notice='Connection error: '+e.message;if(port){await port.close().catch(()=>{});port=null;}status();}
 finally{busy=false;}
}
// Exact strings from the 340-byte physical Model 2 capture; deliberately delivered as a burst.
const saved=['EEG_RESULT 0 3.41962433 -3.47223377 0','EEG_RESULT 1 -0.946972847 0.894363284 1','EEG_RESULT 2 -0.420876831 0.526096046 1','EEG_RESULT 3 -0.631315231 0.526096046 1','Execution_result: PASS (compare host reference separately)\n\x04'];
async function replay(){
 if(busy)return;busy=true;
 try{await stop();$('metadataProfile').value=window.EdgeTrialMetadata?.id??'unknown';begin('SAVED UART REPLAY');
 const parser=EdgeUART.createParser(accept,e=>{notice=e;status();});replayTimer=setTimeout(()=>{replayTimer=null;parser.push(saved.join('\n'));},0);
 }finally{busy=false;}
}
$('connectBtn').onclick=connect;
$('disconnectBtn').onclick=async()=>{if(busy)return;busy=true;try{await stop();notice='Disconnected by user · captured results retained';status();}finally{busy=false;}};
$('replayBtn').onclick=replay;
$('mockBtn').onclick=async()=>{if(busy)return;busy=true;try{await stop();api.useMock();$('transportStatus').textContent='Mock preview · synthetic EEG and fixture predictions';$('latestResult').textContent='No UART prediction received';}finally{busy=false;}};
})();
