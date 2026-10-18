/* EEG signal overview: channel RMS over unique completed recordings. Synthetic arrays only. */
'use strict';
const assert=require('node:assert/strict');
const E=require('../summary/eeg-overview.js'),S=require('../summary/session-summary.js'),V=require('../summary/summary-view.js');
const uV={units:'µV',unitsVerified:true};
const close=(a,b,eps=1e-12)=>assert(Math.abs(a-b)<eps,`${a} != ${b}`);
// Known arithmetic: RMS = sqrt(sum of squares / sample count), pooled across recordings per channel.
close(E.channelRMS([3,-4,3,-4]),3.5355339059327378);assert.equal(E.channelRMS([5,5,5]),null,'flat samples are not valid');assert.equal(E.channelRMS([1,NaN]),null);
let o=E.summarize([{sourceTrialId:'d|S1/a',channels:{Cz:[1,-1,1,-1],C3:[3,-3,3,-3]},artifact:false},{sourceTrialId:'d|S1/b',channels:{Cz:[2,-2,2,-2],C3:[0.5,-0.5,0.5,-0.5]},artifact:true}],uV);
assert.equal(o.available,true);assert.equal(o.recordings.used,2);assert.equal(o.recordings.total,2);
close(o.channels.find(c=>c.name==='Cz').rms,Math.sqrt((4*1+4*4)/8));close(o.channels.find(c=>c.name==='C3').rms,Math.sqrt((4*9+4*.25)/8));
assert.equal(o.highest.name,'C3');assert.equal(o.lowest.name,'Cz');assert.equal(o.units,'µV');assert.equal(o.sampleCount,16);assert.deepEqual(o.artifact,{known:true,flagged:1});assert.equal(o.label,'Prerecorded EEG · channel RMS');
// Repeated recordings count once, regardless of how many completed trials used them.
const acc=E.createAccumulator(uV);assert.equal(acc.add('r1',{Cz:[1,-1],C3:[2,-2]}),true);assert.equal(acc.add('r1',{Cz:[100,-100],C3:[200,-200]}),false);assert.equal(acc.add('r2',{Cz:[1,-1],C3:[2,-2]}),true);assert.equal(acc.has('r1'),true);assert.equal(acc.size,2);
o=acc.result();assert.equal(o.recordings.used,2);close(o.channels.find(c=>c.name==='Cz').rms,1);close(o.channels.find(c=>c.name==='C3').rms,2);
// Unverified units, missing EEG, flat and non-finite channels.
o=E.summarize([{sourceTrialId:'a',channels:{Cz:[1,-1],C3:[2,-2]}}],{units:'µV',unitsVerified:false});assert.equal(o.available,false);assert.match(o.reason,/units not verified/);
o=E.summarize([{sourceTrialId:'a',channels:{Cz:[1,-1],C3:[2,-2]}}],{units:null,unitsVerified:true});assert.equal(o.available,false);assert.match(o.reason,/units/);
o=E.summarize([],uV);assert.equal(o.available,false);assert.match(o.reason,/no completed recordings/);
o=E.summarize([{sourceTrialId:'a',channels:null},{sourceTrialId:'b',channels:{}}],uV);assert.equal(o.available,false);assert.match(o.reason,/no EEG samples/);assert.equal(o.recordings.missing,2);
o=E.summarize([{sourceTrialId:'a',channels:{Cz:[1,-1],C3:[7,7],C4:[1,Infinity]}}],uV);assert.equal(o.available,false);assert.match(o.reason,/fewer than two channels/);assert.deepEqual(o.excludedChannels,['C3','C4']);
// Consistent coverage: a channel missing or flat in one recording is excluded from the ranking instead of being ranked over fewer recordings; missing channels are never zeros.
o=E.summarize([{sourceTrialId:'a',channels:{Cz:[1,-1],C3:[2,-2],C4:[3,-3]}},{sourceTrialId:'b',channels:{Cz:[1,-1],C3:[2,-2]}},{sourceTrialId:'c',channels:{Cz:[1,-1],C3:[2,-2],C4:[0,0]}}],uV);
assert.equal(o.available,true);assert.deepEqual(o.channels.map(c=>c.name),['C3','Cz']);assert.deepEqual(o.excludedChannels,['C4']);assert.equal(o.recordings.used,3);assert.equal(o.channels.every(c=>c.samples===6),true,'every ranked channel covers the same recordings');
o=E.summarize([{sourceTrialId:'a',channels:{Cz:[1,-1],C3:[2,-2]}},{sourceTrialId:'b',channels:{Cz:[1,-1]}}],uV);assert.equal(o.available,false);assert.match(o.reason,/coverage differs/);
// A recording without EEG reduces coverage for all channels equally and is reported, not zero-filled.
o=E.summarize([{sourceTrialId:'a',channels:{Cz:[1,-1],C3:[2,-2]}},{sourceTrialId:'b',channels:null}],uV);assert.equal(o.available,true);assert.equal(o.recordings.missing,1);assert.equal(o.recordings.used,1);close(o.channels.find(c=>c.name==='C3').rms,2);
// Different channel sets and typed arrays work; ranking is by RMS, ties by name.
o=E.summarize([{sourceTrialId:'x',channels:{A:new Float32Array([1,-1]),B:new Float32Array([1,-1]),P9:[4,-4]}}],uV);assert.deepEqual(o.channels.map(c=>c.name),['P9','A','B']);
// Artifact flags unknown when the provider gives none.
o=E.summarize([{sourceTrialId:'a',channels:{Cz:[1,-1],C3:[2,-2]}}],uV);assert.deepEqual(o.artifact,{known:false,flagged:0});
// View: available and unavailable states; never calls amplitude activation.
let s=S.build({classLabels:['left_hand','right_hand']},[{trialId:'t1',sourceTrialId:'a',groundTruth:0,predictedClass:0}],{eeg:E.summarize([{sourceTrialId:'a',channels:{Cz:[1,-1],C3:[2,-2]},artifact:true}],uV)});
let html=V.renderHTML(s);assert.match(html,/Top EEG Channels/);assert.match(html,/ss-eeg-label">RMS \(µV\)</);assert.doesNotMatch(html,/By signal amplitude/);assert.match(html,/ss-rank-n">1<\/span><span class="ss-rank-name">C3<\/span>/);assert.match(html,/2\.00 µV/);assert.match(html,/ss-rank-n">2<\/span><span class="ss-rank-name">Cz<\/span>/);assert.match(html,/<\/ol><div class="ss-foot"><p class="ss-eeg-meta">Prerecorded EEG \| 1 recording \| 1 artifact-flagged<\/p>.*<\/section>/,'supporting line sits inside the Top EEG Channels section');assert.match(html,/Amplitude ranking does not indicate neural activation/);assert.doesNotMatch(html,/Lowest|significant|most active|important|brain activity/);assert.doesNotMatch(html.slice(html.indexOf('ss-foot')),/unique/,'EEG line says "recordings", not "unique recordings"');
assert.match(V.renderEEGFoot(E.summarize([{sourceTrialId:'a',channels:{Cz:[1,-1],C3:[2,-2]}},{sourceTrialId:'b',channels:{Cz:[1,-1],C3:[2,-2]},artifact:true},{sourceTrialId:'c',channels:null}],uV)),/Prerecorded EEG \| 2 recordings \| 1 artifact-flagged \| 1 without EEG/);
// Top three only, shared zero-based scale, deterministic ties, two-channel state; full channel list stays in the snapshot.
const five=E.summarize([{sourceTrialId:'a',channels:{A:[4,-4],B:[2,-2],C:[1,-1],D:[0.5,-0.5],E:[0.25,-0.25]}}],uV);
html=V.renderEEG(five);assert.equal((html.match(/ss-rank-n"/g)||[]).length,3);assert.equal(five.channels.length,5);
assert.match(html,/A<\/span><span class="ss-bar"[^>]*><span style="width:100\.0%"/);assert.match(html,/B<\/span><span class="ss-bar"[^>]*><span style="width:50\.0%"/);assert.match(html,/C<\/span><span class="ss-bar"[^>]*><span style="width:25\.0%"/);assert.doesNotMatch(html,/rank-name">D</);
const tie=E.summarize([{sourceTrialId:'a',channels:{Pz:[2,-2],C3:[2,-2],Fz:[1,-1]}}],uV);assert.deepEqual(tie.channels.map(c=>c.name),['C3','Pz','Fz'],'equal RMS ordered by name');html=V.renderEEG(tie);assert.equal((html.match(/width:100\.0%/g)||[]).length,2);
const two=E.summarize([{sourceTrialId:'a',channels:{Cz:[1,-1],C3:[3,-3]}}],uV);html=V.renderEEG(two);assert.equal((html.match(/ss-rank-n"/g)||[]).length,2);assert.match(html,/rank-name">C3<\/span>/);
s=S.build({classLabels:['left_hand','right_hand']},[],{eeg:E.summarize([],uV)});html=V.renderHTML(s);assert.match(html,/EEG amplitude unavailable \| no completed recordings/);
s=S.build({classLabels:['left_hand','right_hand']},[]);assert.equal(s.eeg,null);assert.match(V.renderHTML(s),/EEG amplitude unavailable/);
assert.throws(()=>{s.metrics.total=5;},TypeError,'finalized snapshot is immutable');
console.log('PASS: pooled channel RMS arithmetic, unique-recording counting, unverified units, missing/flat/non-finite samples, consistent coverage across ranked channels, artifact flags, channel sets, and the EEG overview rendering.');
