import {page} from './browser-cdp.mjs';import fs from 'node:fs/promises';import assert from 'node:assert/strict';
import path from 'node:path';import os from 'node:os';import {pathToFileURL} from 'node:url';
const sourceURL=pathToFileURL(path.resolve(process.env.CINDY_UI_DIR||path.join(os.homedir(),'Desktop/cindy-new-ui'),'index.html')).href;
const outputDir=path.resolve(process.env.EDGEBCI_QA_DIR||'/tmp/edgebci-cindy-review');await fs.mkdir(outputDir,{recursive:true});
const testURL=process.env.EDGEBCI_URL||'http://127.0.0.1:8000/?v=eeg-1';

const tab=await page(testURL);
const sleep=ms=>new Promise(ok=>setTimeout(ok,ms));
async function until(expression){for(let i=0;i<160;i++){const value=await tab.evaluate(expression);if(value)return value;await sleep(100);}throw new Error('Timed out: '+expression);}
const click=id=>tab.evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
await tab.evaluate("document.getElementById('hardwareControls').open=true");
assert(await tab.evaluate("['connectBtn','disconnectBtn','replayBtn','mockBtn'].every(id=>document.getElementById(id).getBoundingClientRect().width>0)"));
await tab.screenshot(outputDir+'/uart-controls.png');
await click('replayBtn');await until('ArmEEGDemo.getState().playback.results.length===4');
await tab.evaluate("document.getElementById('hardwareControls').open=false");
const samples=[];
for(let trial=0;trial<4;trial++){
 await until(`ArmEEGDemo.getState().playback.active_trial===${trial} && ArmEEGDemo.getState().playback.trial_elapsed_ms>=650`);
 const sample=await tab.evaluate(`(()=>{const s=ArmEEGDemo.getState();return {...s,cue:document.getElementById('groundTruth').textContent,person:document.getElementById('person').getAttribute('class'),accuracyText:document.getElementById('accuracyValue').textContent,correctText:document.getElementById('correctValue').textContent,mouseLeft:document.getElementById('mouse').style.left,pill:document.querySelector('.pill.mock').textContent,eeg:EdgeRecordedEEG.getState(),eegLabel:document.getElementById('mapSource').textContent,shownChannels:Array.from(document.querySelectorAll('.trace')).filter(e=>e.style.display!=='none').length};})()`);
 assert.equal(sample.current_class,[0,1,1,1][trial]);assert.equal(sample.cue,['LEFT','LEFT','RIGHT','RIGHT'][trial]);assert(sample.person.includes(trial<2?'raiseL':'raiseR'));assert(trial===0?sample.cursor_x<.5:sample.cursor_x>.5);assert.equal(sample.shownChannels,22);assert.equal(sample.eeg.uart_trial,trial);assert.equal(sample.eeg.units,'µV');assert.match(sample.eegLabel,/Prerecorded EEG.*filtered µV/);assert.equal(sample.pill,'Saved replay');
 assert(await tab.evaluate(`(()=>{const data=EdgeRecordedEEGData,row=data.trials[EdgeRecordedEEG.getState().trial_id],max=Math.max(...row.rms_uv);return data.channel_order.every((name,i)=>{const a=row.channels[name],peak=Math.max(...a.map(Math.abs));let path='';for(let j=0;j<160;j++){const x=j/159*600,y=9.5-(a[Math.floor(j/160*a.length)]||0)/peak*7;path+=(j?' L':'M')+x.toFixed(1)+' '+y.toFixed(1);}return document.getElementById('wave'+i).getAttribute('d')===path&&Math.abs(Number(document.getElementById('heat-'+name).getAttribute('opacity'))-row.rms_uv[i]/max)<1e-12;});})()`),'rendered waveforms and scalp RMS match this trial');
 samples.push(sample);await tab.screenshot(outputDir+'/trial-'+trial+'.png');
 if(trial===0){await click('playBtn');const held=await tab.evaluate('ArmEEGDemo.getState().cursor_x');await sleep(350);assert.equal(await tab.evaluate('ArmEEGDemo.getState().cursor_x'),held);await click('playBtn');}
}
await until('ArmEEGDemo.getState().playback.complete && !ArmEEGDemo.getState().playing');
assert.equal(await tab.evaluate("document.getElementById('accuracyValue').textContent"),'75%');assert.equal(await tab.evaluate("document.getElementById('correctValue').textContent"),'3 / 4');
assert.match(await tab.evaluate("document.getElementById('transportStatus').textContent"),/SAVED UART REPLAY.*Complete.*PASS.*EOT/);
await tab.screenshot(outputDir+'/replay-complete.png');
await click('resetBtn');assert.deepEqual(await tab.evaluate('({count:ArmEEGDemo.getState().playback.results.length,x:ArmEEGDemo.getState().cursor_x,scored:ArmEEGDemo.getState().accuracy.scored,playing:ArmEEGDemo.getState().playing})'),{count:4,x:.5,scored:0,playing:false});
// Exercise actual buttons/lifecycle against an injected fake serial object in this isolated profile.
await tab.evaluate(`(()=>{let pending;window.fakeSerial={opens:[],closed:0,requests:0};const reader={read:()=>new Promise(r=>{pending=r}),cancel:async()=>pending({done:true}),releaseLock(){}};window.fakeUARTPush=text=>pending({value:new TextEncoder().encode(text),done:false});Object.defineProperty(navigator,'serial',{configurable:true,value:{requestPort:async()=>{fakeSerial.requests++;return {open:async opts=>fakeSerial.opens.push(opts),close:async()=>fakeSerial.closed++,readable:{getReader:()=>reader}}}}});document.getElementById('metadataProfile').value='unknown';})()`);
await click('connectBtn');await until('fakeSerial.opens.length===1');
assert.deepEqual(await tab.evaluate('fakeSerial.opens[0]'),{baudRate:115200,dataBits:8,stopBits:1,parity:'none',flowControl:'none'});
await click('playBtn'); // Pause visual clock; capture must continue.
const fixture=await fs.readFile(new URL('./model2-uart.txt',import.meta.url),'utf8');
await tab.evaluate(`fakeUARTPush(${JSON.stringify(fixture)})`);await until('ArmEEGDemo.getState().playback.results.length===4');
await click('disconnectBtn');await until('fakeSerial.closed===1');
assert.equal(await tab.evaluate('ArmEEGDemo.getState().playback.results.length'),4);
assert.match(await tab.evaluate("document.getElementById('transportStatus').textContent"),/Complete.*PASS.*UART disconnected/);
assert.equal(await tab.evaluate("document.getElementById('groundTruth').textContent"),'—');assert.equal(await tab.evaluate('ArmEEGDemo.getState().accuracy.scored'),0);
// Public EEG API supports all 22 names and never fills omitted hardware channels with mocks.
await tab.evaluate("ArmEEGHardware.receive({type:'eeg',channels:{POz:[1,2,3]}})");
assert.equal(await tab.evaluate("document.getElementById('trace-POz').style.display"),'flex');assert.equal(await tab.evaluate("document.getElementById('trace-C3').style.display"),'none');
const errors=tab.events.filter(m=>m.method==='Runtime.exceptionThrown'||m.method==='Log.entryAdded'&&m.params.entry.level==='error');assert.deepEqual(errors,[]);
await fs.writeFile(outputDir+'/browser-results.json',JSON.stringify({samples,finalAccuracy:'75%',correct:'3 / 4',cursorPredictions:[0,1,1,1],taskCues:[0,0,1,1],webSerial:'fake port only; settings/buttons/capture/disconnect verified',errors},null,2));
await tab.close();
// No responsive overrides to Jihyung's original panels: compare original layout at each size.
for(const width of [390,720,900]){
 const original=await page(sourceURL,width,844);
 const actual=await page(testURL,width,844);
 const layout=`(()=>{const r={};for(const s of ['.shell','.top-grid','.drawer','.cursor-panel','.brain-topomap']){const b=document.querySelector(s).getBoundingClientRect();r[s]={x:b.x,y:b.y,w:b.width,h:b.height}}return r})()`;
 assert.deepEqual(await actual.evaluate(layout),await original.evaluate(layout));
 assert(await actual.evaluate('document.documentElement.scrollWidth<=innerWidth'),'horizontal overflow at '+width);
 await actual.evaluate("document.getElementById('hardwareControls').open=true");
 const bounds=await actual.evaluate("(()=>{const r=document.querySelector('.hardware-dock-panel').getBoundingClientRect();return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:innerWidth,height:innerHeight}})()");assert(bounds.left>=0&&bounds.right<=bounds.width&&bounds.top>=0&&bounds.bottom<=bounds.height,'hardware controls fit viewport '+width);
 await actual.screenshot(outputDir+'/responsive-'+width+'.png');await actual.close();await original.close();
}
console.log('PASS: real Chrome saved replay L/R/R/R; task L/L/R/R; 75% accuracy; pause/reset retention; fake serial buttons/settings/disconnect; verified prerecorded EEG/RMS for all four trials; responsive original geometry at 390/720/900; no browser errors.');
