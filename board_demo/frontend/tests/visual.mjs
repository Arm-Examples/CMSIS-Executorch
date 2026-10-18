import {page} from './browser-cdp.mjs';import fs from 'node:fs/promises';import assert from 'node:assert/strict';
import path from 'node:path';import os from 'node:os';import {pathToFileURL} from 'node:url';
const sourceURL=pathToFileURL(path.resolve(process.env.CINDY_UI_DIR||path.join(os.homedir(),'Desktop/cindy-new-ui'),'index.html')).href;
const outputDir=path.resolve(process.env.EDGEBCI_QA_DIR||'/tmp/edgebci-cindy-review');await fs.mkdir(outputDir,{recursive:true});
const testURL=process.env.EDGEBCI_URL||'http://127.0.0.1:8000/?v=cindy-5';

const original=await page(sourceURL);
const merged=await page(testURL);
const inspect=`(()=>{const sels=['.shell','.top-grid','.brain-panel','.drawer','.traces','.cursor-panel','.prediction-summary','.target-area','.target.left','.target.right','.person','.brain-topomap'];return Object.fromEntries(sels.map(s=>{const el=document.querySelector(s),r=el.getBoundingClientRect(),c=getComputedStyle(el);return [s,{x:r.x,y:r.y,w:r.width,h:r.height,background:c.background,border:c.border,borderRadius:c.borderRadius,font:c.font}]}));})()`;
const reference=await original.evaluate(inspect),actual=await merged.evaluate(inspect);
assert.deepEqual(actual,reference,'Original panel geometry, fonts, borders and backgrounds must match');
assert.equal(await merged.evaluate('document.querySelectorAll(".electrode").length'),22);
assert.deepEqual(await merged.evaluate('Array.from(document.querySelectorAll(".trace-column"),c=>c.children.length)'),[8,7,7]);
await original.screenshot(outputDir+'/original-desktop.png');await merged.screenshot(outputDir+'/merged-desktop.png');
const errors=merged.events.filter(m=>m.method==='Runtime.exceptionThrown'||m.method==='Log.entryAdded'&&m.params.entry.level==='error');assert.deepEqual(errors,[]);
await fs.writeFile(outputDir+'/design-comparison.json',JSON.stringify({viewport:{width:1512,height:950},reference,actual,identical:true,electrodes:22,traceColumns:[8,7,7],errors},null,2));
console.log('PASS: original and merged desktop geometry, fonts, borders and backgrounds identical; 22 electrodes and 8/7/7 waveforms; no JavaScript errors.');
await original.close();await merged.close();
