/* Isolated browser visual QA. All UART responses below are explicitly simulated;
 * production protocol/transport/model code is never replaced on disk. */
import {page} from './browser-cdp.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const out=process.env.EDGEBCI_QA_DIR||'/private/tmp/edgebci-final-review';await fs.mkdir(out,{recursive:true});
const url=process.env.EDGEBCI_URL||'http://127.0.0.1:8000/?stream=multisubject';
for(const width of [1512,900,720,390]){
 const t=await page(url,width,950);
 assert(await t.evaluate('document.documentElement.scrollWidth<=innerWidth'),'overflow at '+width);
 await t.screenshot(out+'/layout-'+width+'.png');await t.close();
}
const t=await page(url);
const until=async expression=>{for(let i=0;i<100;i++){if(await t.evaluate(expression))return;await new Promise(r=>setTimeout(r,100));}throw Error('Timed out: '+expression);};
await t.evaluate(`(()=>{
 const banner=document.createElement('div');banner.textContent='VISUAL QA · SIMULATED UART RESPONSES · genuine prerecorded EEG · no physical board';banner.style.cssText='position:fixed;bottom:0;left:0;right:0;z-index:1000;background:#342811;color:#ffdc9e;text-align:center;font:12px sans-serif;padding:6px';document.body.appendChild(banner);
 const P=EdgeStreamProtocol,profile=EdgeMultisubjectProfile.profile;
 EdgeStream.SerialLink=class {
 constructor(){this.connected=false;this.frames=[];this.onData=()=>{};this.onDisconnect=()=>{};}
 async connect(){this.connected=true;}
 send(f){this.onData(P.encode(f));}
 async write(b){if(b.length===1)return;const f=P.decode(b.subarray(0,-1));
 if(f.type===P.T.HELLO){const p=new Uint8Array(69),v=new DataView(p.buffer);p.set(profile.model_sha256.match(/../g).map(h=>parseInt(h,16)));v.setFloat32(32,profile.input_quant_scale,true);v.setInt32(36,-12,true);v.setUint32(40,5632,true);profile.input_shape.forEach((n,i)=>v.setUint16(44+2*i,n,true));p[52]=1;p.set(new TextEncoder().encode('SIMULATED-QA'),53);this.send({...f,type:P.T.READY,payload:p});return;}
 const checksum=P.crc(f.payload),ack=new Uint8Array(4);new DataView(ack.buffer).setUint32(0,checksum,true);this.send({...f,type:P.T.ACCEPTED,payload:ack});
 const r=new Uint8Array(16),v=new DataView(r.buffer);v.setUint32(0,checksum,true);v.setFloat32(4,2,true);v.setFloat32(8,-1,true);r[12]=0;this.send({...f,type:P.T.RESULT,payload:r});
 }
 async disconnect(){this.connected=false;this.onDisconnect();}
 };
 document.getElementById('connectBtn').click();
})()`);
await until('window.EdgeStreamingController?.ready');
await t.evaluate("document.getElementById('playBtn').click()");
await until('ArmEEGDemo.getState().playback.trial_elapsed_ms>700');
await t.screenshot(out+'/playing-simulated-uart.png');
await t.evaluate("document.getElementById('playBtn').click()");
await until("!document.getElementById('sessionSummary').hidden");
assert.match(await t.evaluate("document.getElementById('sessionSummary').textContent"),/Run paused/);
await t.screenshot(out+'/paused-summary.png');
await t.evaluate("document.getElementById('ssResume').click()");
await until("document.getElementById('sessionSummary').hidden");
await t.evaluate("document.getElementById('endRunBtn').click()");
await until("document.getElementById('sessionSummary').textContent.includes('Run complete')");
assert(await t.evaluate('EdgeStreamingController.ended'));
await t.screenshot(out+'/completed-summary.png');
await t.evaluate("document.getElementById('ssNewRun').click()");
await until("document.getElementById('sessionSummary').hidden && EdgeStreamingController.completedRecords.length===0");
assert(await t.evaluate('EdgeStreamingController.completedArchives.length>0'));
assert.deepEqual(t.events.filter(x=>x.method==='Runtime.exceptionThrown'||x.method==='Log.entryAdded'&&x.params.entry.level==='error'),[]);
await t.close();console.log('PASS: desktop/mobile overflow, genuine EEG with explicitly simulated UART, Play/Pause/Resume summary, native End Run/New Run archive; no browser errors. Screenshots: '+out);
