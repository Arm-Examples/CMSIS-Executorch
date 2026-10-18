/* Execute the actual GUI/bridge/controller with genuine prepared EEG and simulated UART. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const root=path.resolve(__dirname,'../..'),nodes=new Map(),events=[],frames=[];
function node(id){
 if(!nodes.has(id)){const classes=new Set();nodes.set(id,{textContent:'',style:{},clientWidth:800,children:[],attributes:{},handlers:[],classList:{add(...v){v.forEach(x=>classes.add(x));},remove(...v){v.forEach(x=>classes.delete(x));},contains:x=>classes.has(x),toggle(x,v){v=v===undefined?!classes.has(x):v;v?classes.add(x):classes.delete(x);return v;}},addEventListener(name,fn,capture){this.handlers.push({name,fn,capture:!!capture});},async click(){let stop=false;const event={stopImmediatePropagation(){stop=true;}};for(const h of this.handlers.filter(h=>h.name==='click').sort((a,b)=>Number(b.capture)-Number(a.capture))){await h.fn(event);if(stop)return;}if(this.onclick)return this.onclick(event);},setAttribute(k,v){this.attributes[k]=v;},appendChild(x){this.children.push(x);},querySelector(){return node(String(id)+'child');},getBoundingClientRect(){return {width:800,left:id==='rightTarget'?700:0,right:id==='leftTarget'?100:800};}});}return nodes.get(id);
}
let now=0;const ctx={console,crypto,performance,Uint8Array,Int8Array,DataView,TextEncoder,TextDecoder,Map,Set,Array,Number,JSON,Promise,Math,Date,Error,Blob,URL,location:{pathname:'/'},EdgeStreamingMode:true,EdgeMultisubjectMode:true,
 document:{getElementById:node,querySelector:node,createElement:()=>node(Symbol()),createElementNS:()=>node(Symbol())},setInterval(){},setTimeout,clearTimeout,requestAnimationFrame:f=>frames.push(f),addEventListener(){},dispatchEvent:event=>events.push(event),CustomEvent:class{constructor(type,options){this.type=type;this.detail=options.detail;}},
 fetch:async file=>{try{const b=fs.readFileSync(path.join(root,'frontend',file));return {ok:true,json:async()=>JSON.parse(b),arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};}catch{return {ok:false};}}
};ctx.window=ctx;vm.createContext(ctx);
const load=f=>vm.runInContext(fs.readFileSync(path.join(root,'frontend',f),'utf8'),ctx,{filename:f});
load('playback.js');node('drawer').classList.add('open');load('gui.js');load('hardware-adapter.js');load('streaming/protocol.js');load('multisubject/profile.js');load('streaming/controller.js');load('multisubject/controller.js');load('multisubject/dataset.js');
const P=ctx.EdgeStreamProtocol,profile=ctx.EdgeMultisubjectProfile.profile;let serial,requests=0;
ctx.EdgeStream.SerialLink=class{
 constructor(){serial=this;this.connected=false;this.frames=[];this.onData=()=>{};this.onDisconnect=()=>{};}
 async connect(){this.connected=true;}
 send(f){const bytes=P.encode(f);this.frames.push({direction:'rx',bytes:Array.from(bytes)});for(let i=0;i<bytes.length;i+=9)this.onData(bytes.subarray(i,i+9));}
 async write(b){this.frames.push({direction:'tx',bytes:Array.from(b)});if(b.length===1)return;const f=P.decode(b.subarray(0,-1));
  if(f.type===P.T.HELLO){const p=new Uint8Array(69),v=new DataView(p.buffer);p.set(profile.model_sha256.match(/../g).map(h=>parseInt(h,16)));v.setFloat32(32,profile.input_quant_scale,true);v.setInt32(36,-12,true);v.setUint32(40,5632,true);profile.input_shape.forEach((n,i)=>v.setUint16(44+2*i,n,true));p[52]=1;p.set(new TextEncoder().encode('SIMULATION-ONLY'),53);this.send({...f,type:P.T.READY,payload:p});return;}
  requests++;const checksum=P.crc(f.payload),ack=new Uint8Array(4);new DataView(ack.buffer).setUint32(0,checksum,true);this.send({...f,type:P.T.ACCEPTED,payload:ack});
  // A deliberate classification opposite to metadata verifies independent task/cursor paths.
  const trial=ctx.EdgeStreamingController.dataset.trials[f.trial],cls=1-trial.ground_truth,result=new Uint8Array(16),v=new DataView(result.buffer);v.setUint32(0,checksum,true);v.setFloat32(4,cls===0?2:-1,true);v.setFloat32(8,cls===1?2:-1,true);result[12]=cls;this.send({...f,type:P.T.RESULT,payload:result});
 }
 async disconnect(){this.connected=false;this.onDisconnect();}
};
load('multisubject/bridge.js');
for(const f of ['session-summary','eeg-overview','summary-view','multisubject-view'])load('summary/'+f+'.js');
const flush=()=>new Promise(r=>setImmediate(r));
async function advance(ms){for(let t=0;t<ms;t+=100){const pending=frames.splice(0);now+=100;for(const fn of pending)fn(now);await flush();}}
async function until(f){for(let i=0;i<500;i++){if(f())return;await flush();}throw Error('Test condition timed out');}
(async()=>{
 assert.equal(ctx.ArmEEGDemo.getState().source,'hardware');assert.equal(ctx.ArmEEGDemo.getState().playing,false);assert.equal(requests,0);
 await node('connectBtn').click();const c=ctx.EdgeStreamingController;assert(c.ready);assert.match(node('transportStatus').textContent,/firmware verified/);assert.equal(requests,0);
 await node('playBtn').click();await until(()=>c.completedRecords.length===1);await advance(700);
 const r=c.completedRecords[0],state=ctx.ArmEEGDemo.getState();assert.equal(state.current_class,r.class_id);assert(r.class_id===0?state.cursor_x<.5:state.cursor_x>.5);assert.equal(node('groundTruth').textContent,r.ground_truth===0?'LEFT':'RIGHT');assert(node('person').classList.contains(r.ground_truth===0?'raiseR':'raiseL'));assert.equal(state.accuracy.scored,1);assert.equal(state.accuracy.correct,0);
 assert.match(node('mapSource').attributes.title,/125 Hz, 2.048 s/);assert.match(node('mapSource').attributes.title,/before alignment/);assert(node('mapSource').textContent.includes(r.trial_id));
 const channelValues=c.dataset.channels(r.trial),rms=profile.channel_order.map(name=>Math.sqrt(channelValues[name].reduce((s,x)=>s+x*x,0)/256)),max=Math.max(...rms,1e-9);
 const fraction=state.playback.trial_elapsed_ms/state.playback.duration_ms;
 for(const [i,name]of profile.channel_order.entries()){assert.equal(node('trace-'+name).style.display,'flex');assert.equal(Number(node('heat-'+name).attributes.opacity),rms[i]/max);const a=channelValues[name],peak=Math.max(...a.map(Math.abs)),count=1+Math.floor(fraction*(a.length-1));const expected=Array.from({length:count},(_,j)=>(j?'L':'M')+(j/(a.length-1)*600).toFixed(1)+' '+(9.5-a[j]/peak*7).toFixed(1)).join(' ');assert.equal(node('wave'+i).attributes.d,expected,'all displayed points must use this trial’s real samples');}
 await node('playBtn').click();const held=ctx.ArmEEGDemo.getState().cursor_x,heldWave=node('wave0').attributes.d;await advance(2300);assert.equal(requests,1);assert.equal(ctx.ArmEEGDemo.getState().cursor_x,held);assert.equal(node('wave0').attributes.d,heldWave,'Pause freezes EEG at identical sample position');
 assert.equal(node('sessionSummary').hidden,false);assert.match(node('sessionSummary').innerHTML,/Run paused/);assert.match(node('sessionSummary').innerHTML,/Top EEG Channels/);
 await node('playBtn').click();await advance(1200);assert.equal(node('sessionSummary').hidden,true);await until(()=>c.completedRecords.length>=2);await advance(100);
 const second=c.completedRecords[1];assert.notEqual(second.trial_id,r.trial_id);assert(node('mapSource').textContent.includes(second.trial_id));assert.equal(ctx.ArmEEGDemo.getState().playback.active_trial,second.trial);assert.notEqual(node('wave0').attributes.d,heldWave,'new trial restarts progressive sweep with its own data');
 const completed=events.filter(e=>e.type==='edgebci:trial-completed');assert.equal(completed.length,c.completedRecords.length);assert.equal(completed[0].detail.model_id,profile.id);assert.equal(completed[0].detail.timing.hardware_inference_ms,null);
 await node('endRunBtn').click();assert(c.ended);assert.match(node('sessionSummary').innerHTML,/Run complete/);assert.equal(c.pending,null);assert.equal(ctx.ArmEEGDemo.getState().playing,false);assert.equal(events.filter(e=>e.type==='edgebci:run-ended').length,1);const count=requests;await advance(2500);assert.equal(requests,count);
 await node('newRunBtn').click();await until(()=>c.completedRecords.length===0);await advance(100);assert.equal(node('sessionSummary').hidden,true);assert.equal(c.records.length,0);assert.equal(c.completedArchives[0].length,2);assert.equal(ctx.ArmEEGDemo.getState().playback.results.length,0);assert.equal(ctx.ArmEEGDemo.getState().cursor_x,.5);assert.equal(node('wave0').attributes.d,'','Reset clears prior EEG visualization');
 await node('disconnectBtn').click();const exported=ctx.EdgeMultisubjectSession.getCapture();assert.equal(exported.archives[0].length,2);assert(exported.frames.length>0);assert.equal(serial.connected,false);
 console.log('PASS: final video summary controls, paused/resumed overlay, native End Run, New Run archive;  production Cindy Ku's GUI + multi-subject bridge/controller, genuine EEG synchronization and exact scalp RMS, independent ground truth/actual simulated prediction, cursor direction, accuracy, Pause/Resume, safe End Run, Reset/archive, disconnect retention and session-summary events. Simulated UART only.');
})().catch(e=>{console.error(e);process.exitCode=1;});
