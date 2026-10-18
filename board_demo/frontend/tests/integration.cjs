const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm');
const {createParser}=require('../uart-parser.js');
const fixture=fs.readFileSync(require.resolve('./model2-uart.txt'),'utf8');
const messages=[],errors=[];
const parser=createParser(m=>messages.push(m),e=>errors.push(e));
for(const c of fixture)parser.push(c); // One-character fragments, real PASS suffix, EOT.
assert.deepEqual(messages.filter(m=>m.type==='prediction').map(m=>m.class_id),[0,1,1,1]);
assert.equal(messages.at(-2).pass,true);assert.equal(messages.at(-1).type,'eot');
for(const [line,pass] of [['Execution_result: PASS',true],['Execution_result: PASS (compare host reference separately)',true],['Execution_result: PASSING',false],['Execution_result: FAIL',false],['EEG_ERROR trial=0 err=1',false]]){
 const events=[];const p=createParser(m=>events.push(m));p.push(line+'\r\n');assert.equal(events.length,1);assert.equal(events[0].pass,pass);
}
parser.push('EEG_RESULT 1 NaN 2 0\nEEG_RESULT 2 1 2 9\n'+'x'.repeat(4200)+'\n');assert.equal(errors.length,3);
console.log('PASS: actual captured UART, fragmentation, firmware PASS suffix, malformed results, failure and EOT.');
// Execute actual GUI + adapter + transport with a deterministic animation clock and fake port.
const nodes=new Map();
function node(id){if(!nodes.has(id)){const classes=new Set();nodes.set(id,{textContent:'',value:id==='metadataProfile'?'unknown':'',style:{},clientWidth:800,children:[],attributes:{},classList:{add(...v){v.forEach(x=>classes.add(x));},remove(...v){v.forEach(x=>classes.delete(x));},contains:x=>classes.has(x),toggle(x,v){v=v===undefined?!classes.has(x):v;v?classes.add(x):classes.delete(x);return v;}},addEventListener(k,f){this[k]=f;},setAttribute(k,v){this.attributes[k]=v;},appendChild(x){this.children.push(x);},querySelector(){return node(id+'child');},getBoundingClientRect(){return {width:800,left:id==='rightTarget'?700:0,right:id==='leftTarget'?100:800};}});}return nodes.get(id);}
let frame,now=0;const timers=new Map();let timerId=0;
const ctx={console,document:{getElementById:node,querySelector:node,createElement:()=>node(Symbol()),createElementNS:()=>node(Symbol())},setInterval(){},setTimeout(f){timers.set(++timerId,f);return timerId;},clearTimeout(i){timers.delete(i);},requestAnimationFrame(f){frame=f;},addEventListener(){},TextDecoder};
ctx.window=ctx;vm.createContext(ctx);
function load(file){vm.runInContext(fs.readFileSync(require.resolve('../'+file),'utf8'),ctx,{filename:file});}
load('playback.js');load('trial-metadata.js');load('recorded-eeg/model2-eeg-data.js');load('recorded-eeg.js');
assert.deepEqual(JSON.parse(JSON.stringify(ctx.EdgeTrialMetadata)),JSON.parse(fs.readFileSync(require.resolve('../model2-trials.json'),'utf8')));
node('drawer').classList.add('open');load('gui.js');load('hardware-adapter.js');
const gui=ctx.ArmEEGDemo,api=ctx.ArmEEGHardware;
function advance(ms){frame(now);for(let t=0;t<ms;t+=100)frame(now+=100);}
node('playBtn').click();advance(2800);assert.equal(gui.getState().current_class,0);assert(gui.getState().cursor_x<.5);
api.begin();assert.equal(node('heat-POz').attributes.opacity,'0');assert.equal(node('trace-POz').style.display,'none');advance(6000);assert.equal(gui.getState().current_class,null,'no mock output in hardware mode');
const message={type:'prediction',class_id:0,trial:0,logits:[3,-3]};assert(api.receive(message));message.logits[0]=999;
assert.equal(gui.getState().playback.results[0].logits[0],3,'adapter and archive own their result data');advance(700);assert(gui.getState().cursor_x<.5);assert.equal(node('groundTruth').textContent,'—');
assert.equal(api.receive({type:'prediction',class_id:3}),false);
assert(api.receive({type:'eeg',channels:{POz:[1,2,3]}}));assert.equal(node('trace-POz').style.display,'flex');assert.equal(node('trace-C3').style.display,'none');
api.receive({type:'ground_truth',label:1});assert(node('person').classList.contains('raiseL'),'public explicit ground-truth API preserved');
api.useMock();
let releaseRead,closed=0;
const fakeReader={read:()=>new Promise(resolve=>{releaseRead=resolve;}),cancel:async()=>releaseRead({done:true}),releaseLock(){}};
const fakePort={readable:{getReader:()=>fakeReader},close:async()=>{closed++;}};
// Objects from a VM have distinct prototypes; compare transport options by serialization.
fakePort.open=async opts=>assert.equal(JSON.stringify(opts),JSON.stringify({baudRate:115200,dataBits:8,stopBits:1,parity:'none',flowControl:'none'}));
ctx.navigator={serial:{requestPort:async()=>fakePort}};ctx.EdgeUART={createParser};load('serial-transport.js');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 await node('connectBtn').onclick();assert.match(node('transportStatus').textContent,/UART connected/);
 node('playBtn').click();assert.equal(gui.getState().playing,false); // Pausing never stops capture.
 releaseRead({value:new TextEncoder().encode(fixture),done:false});await flush();
 assert.match(node('transportStatus').textContent,/Complete.*Execution_result: PASS.*EOT.*UART connected/);
 assert.equal(gui.getState().playback.results.length,4);advance(1000);assert.equal(gui.getState().cursor_x,.5);
 assert.equal(node('capturedResults').textContent.split('\n').length,4);
 assert.match(node('latestResult').textContent,/trial 3.*RIGHT/);
 // Duplicate/post-completion lines cannot erase or append a prediction.
 releaseRead({value:new TextEncoder().encode('EEG_RESULT 0 0 0 0\n'),done:false});await flush();assert.equal(gui.getState().playback.results.length,4);
 await node('disconnectBtn').onclick();assert.equal(closed,1);
 assert.match(node('transportStatus').textContent,/Complete.*PASS.*UART disconnected/);
 node('playBtn').click();
 for(let i=0;i<4;i++){
  advance(700);const state=gui.getState();assert.equal(state.playback.active_trial,i);assert.equal(state.current_class,[0,1,1,1][i]);
  assert(i===0?state.cursor_x<.5:state.cursor_x>.5);assert.equal(node('groundTruth').textContent,'—','unknown profile stays neutral');
  assert(state.cursor_x>.3&&state.cursor_x<.7,'each trial starts a distinct movement from center');
  advance(1100);
 }
 assert.equal(gui.getState().playing,false);assert.match(node('playbackState').textContent,/Playback complete.*delayed/);
 assert.equal(node('playBtn').textContent,'▶ Replay results');
 assert.equal(gui.getState().playback.results.length,4);
 assert.equal(node('accuracyValue').textContent,'—');assert.equal(node('correctValue').textContent,'0 / 0');
 console.log('PASS: four-result burst queues while paused; disconnect/EOT preserve results; all four move in the correct direction over 1.8 seconds.');
 node('resetBtn').click();assert.equal(gui.getState().cursor_x,.5);assert.equal(gui.getState().playback.results.length,4);assert.equal(gui.getState().playback.index,0);
 node('playBtn').click();advance(700);node('playBtn').click();const held=gui.getState();advance(2000);assert.equal(gui.getState().cursor_x,held.cursor_x);assert.equal(gui.getState().playback.trial_elapsed_ms,held.playback.trial_elapsed_ms);
 node('playBtn').click();advance(300);assert(gui.getState().cursor_x<held.cursor_x);assert.equal(node('playBtn').attributes['aria-pressed'],'true');
 console.log('PASS: Play/Pause freeze and resume visual time; Reset rewinds without losing logits/results.');
 // Saved UART uses precisely the same burst parser, adapter, archive and animation path.
 await node('replayBtn').onclick();for(const [id,f] of [...timers]){timers.delete(id);f();}
 assert.equal(gui.getState().playback.results.length,4);assert.match(node('transportStatus').textContent,/SAVED UART REPLAY.*Complete.*PASS.*EOT/);
 const labels=[0,0,1,1];
 function assertRecordedSweep(trial){
  const row=ctx.EdgeRecordedEEG.resolve(gui.getState().playback.results[trial]);
  const fraction=gui.getState().playback.trial_elapsed_ms/1800;
  ctx.EdgeRecordedEEGData.channel_order.forEach((name,i)=>{
   const a=row.channels[name],peak=Math.max(...a.map(Math.abs)),count=1+Math.floor(fraction*(a.length-1));
   const expected=Array.from({length:count},(_,j)=>(j?'L':'M')+(j/(a.length-1)*600).toFixed(1)+' '+(9.5-a[j]/peak*7).toFixed(1)).join(' ');
   assert.equal(node('wave'+i).attributes.d,expected,'actual recorded samples, fixed scale and playback position: '+name);
  });
 }

 for(let i=0;i<4;i++){
  advance(700);assert.equal(gui.getState().current_class,[0,1,1,1][i]);assertRecordedSweep(i);
  assert.equal(ctx.EdgeRecordedEEG.getState().uart_trial,i);assert.match(node('mapSource').textContent,/Prerecorded EEG.*filtered µV/);for(const name of ctx.EdgeRecordedEEGData.channel_order){assert.equal(node('trace-'+name).style.display,'flex');assert(Number(node('heat-'+name).attributes.opacity)>0);}
  assert.equal(node('groundTruth').textContent,labels[i]===0?'LEFT':'RIGHT');
  assert(node('person').classList.contains(labels[i]===0?'raiseR':'raiseL'));
  if(i===0){const wave=node('wave0').attributes.d;node('playBtn').click();advance(900);assert.equal(node('wave0').attributes.d,wave,'Pause freezes real EEG sweep');node('playBtn').click();}
  if(i===1){assert(gui.getState().cursor_x>.5);assert(node('person').classList.contains('raiseR'),'trial 1 LEFT ground truth must be independent of RIGHT prediction');}
  advance(1100);
 }
 assertRecordedSweep(3);assert.equal(node('wave0').attributes.d.match(/[ML]/g).length,500,'all 500 original samples revealed at completion');
 assert.equal(gui.getState().playing,false);assert.equal(node('groundTruth').textContent,'—');
 const rows=JSON.parse(JSON.stringify(gui.getState().playback.results));
 assert.deepEqual(rows.map(r=>r.class_id),[0,1,1,1]);assert.deepEqual(rows.map(r=>r.ground_truth),labels);
 assert.deepEqual(rows.map(r=>r.logits),messages.filter(m=>m.type==='prediction').map(m=>m.logits));
 assert.equal(node('accuracyValue').textContent,'75%');assert.equal(node('correctValue').textContent,'3 / 4');
 console.log('PASS: saved burst replay; verified task cues LEFT/LEFT/RIGHT/RIGHT remain independent of predictions LEFT/RIGHT/RIGHT/RIGHT.');
 // Same prerecorded matching in the hardware-result path, with a fake serial port.
 node('metadataProfile').value=ctx.EdgeTrialMetadata.id;await node('connectBtn').onclick();
 releaseRead({value:new TextEncoder().encode(fixture),done:false});await flush();await node('disconnectBtn').onclick();
 for(let i=0;i<4;i++){advance(700);assert.equal(ctx.EdgeRecordedEEG.getState().uart_trial,i);assert.equal(gui.getState().current_class,[0,1,1,1][i]);advance(1100);}
 assert.equal(node('accuracyValue').textContent,'75%');assert.equal(gui.getState().playback.results.length,4);
 console.log('PASS: known-profile hardware-result playback uses the same genuine EEG per trial, including after UART disconnect.');
 // A physical link ending without PASS is incomplete, never labeled successful.
 node('metadataProfile').value='unknown';await node('connectBtn').onclick();
 releaseRead({value:new TextEncoder().encode('EEG_RESULT 0 1 -1 0\n'),done:false});await flush();
 releaseRead({done:true});await flush();assert.match(node('transportStatus').textContent,/Captured 1\/4.*UART disconnected/);assert.doesNotMatch(node('transportStatus').textContent,/Complete/);assert.equal(gui.getState().playback.results.length,1);
 await node('disconnectBtn').onclick();
 ctx.navigator.serial.requestPort=async()=>{throw new Error('Port busy');};await node('connectBtn').onclick();assert.match(node('transportStatus').textContent,/Port busy/);assert.equal(gui.getState().source,'hardware');
 await node('mockBtn').onclick();assert.equal(gui.getState().source,'mock');
 console.log('PASS: 22-channel EEG adapter, missing hardware EEG stays blank; 75% (3/4) verified accuracy and no score without matching metadata.');
 console.log('PASS: completion distinct from disconnection, incomplete capture, busy-port error and explicit mock switch; no real serial port used.');
})().catch(e=>{console.error(e);process.exitCode=1;});
