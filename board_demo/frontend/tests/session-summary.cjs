/* Session summary metrics, adapter, verified metadata and view rendering. Synthetic fixtures only; no hardware. */
const assert=require('node:assert/strict');
const S=require('../summary/session-summary.js'),M=require('../summary/model-metadata.js'),V=require('../summary/summary-view.js');
const LR={0:{name:'left_hand',display:'Left hand'},1:{name:'right_hand',display:'Right hand'}};
const r=(trialId,sourceTrialId,groundTruth,predictedClass,extra={})=>({trialId,sourceTrialId,groundTruth,predictedClass,logits:predictedClass?[-1,1]:[1,-1],...extra});
// Overall accuracy, confusion matrix, incorrect predictions, per-class accuracy.
let s=S.build({modelName:'Fixture',classLabels:LR,startedAt:1000,endedAt:61000},[r('a:1','d|S1/x/1',0,0),r('a:2','d|S1/x/2',0,1),r('a:3','d|S1/x/3',1,1),r('a:4','d|S1/x/4',1,1),r('a:5','d|S1/x/5',1,0)]);
assert.equal(s.metrics.total,5);assert.equal(s.metrics.correct,3);assert.equal(s.metrics.incorrect,2);assert.equal(s.metrics.accuracy,0.6);
assert.deepEqual(s.metrics.confusion,[[1,1],[1,2]],'rows actual, columns predicted');
assert.deepEqual(s.metrics.perClass.map(c=>[c.name,c.total,c.correct,c.accuracy]),[['left_hand',2,1,0.5],['right_hand',3,2,2/3]]);
assert.equal(s.metrics.unique.count,5);assert.equal(s.metrics.repeated,0);assert.equal(s.metrics.duration.ms,60000);assert.equal(s.metrics.latency,null,'no latency unless measured');
assert.equal(s.session.status,'completed');assert.equal(s.results[1].isCorrect,false);
// Per-class accuracy with a class that has zero observations stays unavailable, not 0%.
s=S.build({classLabels:['left_hand','right_hand','feet']},[r('a:1','s1',0,0),r('a:2','s2',1,2)]);
assert.deepEqual(s.metrics.perClass.map(c=>[c.index,c.total,c.accuracy]),[[0,1,1],[1,1,0],[2,0,null]]);
assert.deepEqual(s.metrics.confusion,[[1,0,0],[0,0,1],[0,0,0]]);
let html=V.renderHTML(s);assert.match(html,/<th scope="row"><span>feet<\/span><small>Unavailable<\/small>/,'zero-observation class row shows Unavailable');assert.doesNotMatch(html,/NaN/);assert.doesNotMatch(html,/ss-classes/,'per-class bars removed');
// Matrix orientation, metadata-driven cell labels, category classes and per-row accuracy.
s=S.build({classLabels:M.models[0].classLabels,startedAt:0,endedAt:462000},[...Array.from({length:17},(_,k)=>r('l'+k,'L'+k,0,0)),...Array.from({length:3},(_,k)=>r('lw'+k,'LW'+k,0,1)),...Array.from({length:13},(_,k)=>r('r'+k,'R'+k,1,1)),...Array.from({length:7},(_,k)=>r('rw'+k,'RW'+k,1,0))]);
html=V.renderHTML(s);assert.match(html,/Confusion Matrix/);assert.match(html,/Actual ↓/);assert.match(html,/Predicted →/);
assert.match(html,/<th scope="row"><span>Left hand<\/span><small>85% correct<\/small>/);assert.match(html,/<th scope="row"><span>Right hand<\/span><small>65% correct<\/small>/);
assert.match(html,/ss-diag"[^>]*><strong>17<\/strong><small>Correct left</);assert.match(html,/ss-off"[^>]*><strong>3<\/strong><small>Incorrect left</);assert.match(html,/ss-off"[^>]*><strong>7<\/strong><small>Incorrect right</);assert.match(html,/ss-diag"[^>]*><strong>13<\/strong><small>Correct right</);
assert.equal((html.match(/class="ss-diag"/g)||[]).length,2);assert.equal((html.match(/class="ss-off"/g)||[]).length,2);assert.equal(s.metrics.accuracy,0.75);
assert.match(html,/>Trials</);assert.match(html,/>Duration</);assert.match(html,/Includes pauses/);assert.doesNotMatch(html,/Completed trials|Elapsed|ss-sub|ss-eyebrow|Misclassifications/);
// Repeats, unique counts and first-result unique accuracy.
s=S.build({classLabels:LR},[r('a:1','rec1',0,1),r('a:2','rec2',1,1),r('a:3','rec1',0,0),r('a:4','rec1',0,0),r('a:5','rec2',1,0)]);
assert.equal(s.metrics.total,5);assert.equal(s.metrics.correct,3);assert.equal(s.metrics.unique.count,2);assert.equal(s.metrics.repeated,3);
assert.equal(s.metrics.unique.correct,1,'first result for rec1 was wrong, for rec2 right');assert.equal(s.metrics.unique.accuracy,0.5);
assert.deepEqual(s.results.map(x=>x.isRepeated),[false,false,true,true,true]);
// Identical recording indices from different subjects are different recordings; the same index + subject is a repeat.
s=S.build({classLabels:LR},[r('a:1','bci|S1/1test/index3',0,0),r('a:2','bci|S2/1test/index3',0,0),r('a:3','bci|S1/1test/index3',0,1)]);
assert.equal(s.metrics.unique.count,2);assert.equal(s.metrics.repeated,1);
// Duplicate responses (same attempt id) are counted once; incomplete records are excluded.
s=S.build({classLabels:LR},[r('a:1','s1',0,0),r('a:1','s1',0,1),r('a:2',null,1,1),{trialId:'a:3',sourceTrialId:'s3',groundTruth:1,predictedClass:7},{trialId:'',sourceTrialId:'s4',groundTruth:0,predictedClass:0}]);
assert.equal(s.metrics.total,1);assert.equal(s.metrics.duplicatesIgnored,1);assert.equal(s.metrics.excluded,3);
// Empty session: zero trials, accuracy unavailable (never 0%), duration unavailable without timing.
s=S.build({classLabels:LR,startedAt:null,endedAt:Date.now()},[]);
assert.equal(s.metrics.total,0);assert.equal(s.metrics.accuracy,null);assert.equal(s.metrics.unique.accuracy,null);assert.equal(s.session.status,'empty');assert.equal(s.metrics.duration,null);
html=V.renderHTML(s);assert.match(html,/ss-status-empty">Run complete</);assert.doesNotMatch(html,/0\.0%/);assert.match(html,/Unavailable/);
// Interrupted session keeps completed results and names the unresolved request.
s=S.build({classLabels:LR,status:'interrupted',interruption:{message:'Hardware result timeout; request unresolved',unresolvedRequest:4,trialId:'S1/1test/run2/cue9'}},[r('a:1','s1',0,0),r('a:2','s2',1,0),r('a:3','s3',1,1)]);
assert.equal(s.metrics.total,3);html=V.renderHTML(s);assert.match(html,/ss-status-interrupted">Run interrupted</);assert.match(html,/Request 4 \(S1\/1test\/run2\/cue9\) not counted/);assert.doesNotMatch(html,/Run complete/);
// Live (paused) view versus finalized view: labels, hint vs New Run, and no modal controls anywhere.
s=S.build({classLabels:LR,status:'paused'},[r('a:1','s1',0,0)]);html=V.renderHTML(s,{final:false,resumable:true});assert.match(html,/ss-status-paused">Run paused</);assert.match(html,/id="ssResume" type="button">Resume</);assert.doesNotMatch(html,/ssNewRun|ssClose|Back to live view|Press Play|role="dialog"|aria-modal/);
html=V.renderHTML(s,{final:false,resumable:false});assert.doesNotMatch(html,/ssResume|ssNewRun/,'no Resume when the controller cannot continue');
html=V.renderHTML(S.build({classLabels:LR},[r('a:1','s1',0,0)]),{final:true});assert.match(html,/ss-status-completed">Run complete</);assert.match(html,/id="ssNewRun" type="button">New Run</);assert.doesNotMatch(html,/✦|ssResume|ssClose|Press Play/);
html=V.renderHTML(S.build({classLabels:LR,status:'interrupted',interruption:{message:'UART disconnected; results retained'}},[r('a:1','s1',0,0)]),{final:false});assert.match(html,/Run interrupted/);assert.doesNotMatch(html,/ssResume|End Run to finalize/);
assert.doesNotMatch(V.renderEnding({request:3}),/ssClose/);assert.match(V.renderEnding({request:3}),/Ending run…/);
// Missing optional metadata: the subtitle says so, nothing is invented, and provenance stays in the snapshot rather than on screen.
s=S.build({classLabels:LR,modelHash:'abc',sessionId:'sid',seed:42},[r('a:1','s1',0,0)]);
html=V.renderHTML(s);assert.match(html,/Session Summary/);assert.doesNotMatch(html,/Model name|Experiment|Model hash|Session ID|Firmware|round trip|animated counter|class 0|left_hand/);assert.equal(s.session.modelHash,'abc');assert.equal(s.session.sessionId,'sid');assert.equal(s.session.subjectId,null);
s=S.build({classLabels:LR,modelName:'Model 2 · subject-specific'},[r('a:1','s1',0,0)]);assert.doesNotMatch(V.renderHTML(s),/Model 2/,'model subtitle removed from the card; kept in session.modelName');assert.equal(s.session.modelName,'Model 2 · subject-specific');
// Unique/repeat line and the repeat note appear only when meaningful.
s=S.build({classLabels:LR},[r('a:1','rec1',0,0),r('a:2','rec2',1,1),r('a:3','rec1',0,1),r('a:4','rec1',0,0),r('a:5','rec2',1,0)]);html=V.renderHTML(s);
assert.match(html,/2 unique recordings \| 3 repeat evaluations/);assert.doesNotMatch(html,/Includes repeats/);assert.match(html,/3 correct \| 2 incorrect/);
s=S.build({classLabels:LR},[r('a:1','rec1',0,0)]);html=V.renderHTML(s);assert.match(html,/1 unique recording</);assert.doesNotMatch(html,/repeat/);assert.match(html,/1 correct \| 0 incorrect/);
s=S.build({classLabels:LR},[]);assert.doesNotMatch(V.renderHTML(s),/unique recording/);
// Class names come from the supplied metadata, including a reversed mapping; nothing assumes 0 = left.
s=S.build({classLabels:{0:'right_hand',1:'left_hand'}},[r('a:1','s1',0,0),r('a:2','s2',1,0)]);
assert.deepEqual(s.classes.map(c=>[c.index,c.name]),[[0,'right_hand'],[1,'left_hand']]);
assert.deepEqual(s.metrics.perClass.map(c=>[c.name,c.accuracy]),[['right_hand',1],['left_hand',0]]);
html=V.renderHTML(s);assert.equal(html.indexOf('right_hand')<html.indexOf('left_hand'),true,'row order follows class index');
assert.match(html,/Actual ↓/);assert.match(html,/Predicted →/);assert.match(html,/<span>right_hand<\/span><small>100% correct<\/small>/);assert.match(html,/<span>left_hand<\/span><small>0% correct<\/small>/);
assert.match(html,/<strong>1<\/strong><small>Correct right_hand</);assert.match(html,/<strong>1<\/strong><small>Incorrect left_hand</,'labels name the actual class from the reversed metadata, not an assumed order');
// Metadata display order: indices reversed (0 = right, 1 = left) but order puts Left first; counts follow identity.
s=S.build({classLabels:{0:{name:'right_hand',display:'Right hand',short:'Right',order:1},1:{name:'left_hand',display:'Left hand',short:'Left',order:0}}},[r('a:1','s1',0,0),r('a:2','s2',1,0)]);
assert.deepEqual(s.metrics.confusion,[[1,0],[1,0]],'snapshot stays in index order');assert.deepEqual(s.classes.map(c=>[c.index,c.order]),[[0,1],[1,0]]);
html=V.renderHTML(s);const cells=[...html.matchAll(/<strong>(\d+)<\/strong><small>([^<]+)<\/small>/g)].map(x=>x[1]+' '+x[2]);
assert.deepEqual(cells,['0 Correct left','1 Incorrect left','0 Incorrect right','1 Correct right']);
assert.deepEqual([...html.matchAll(/<th scope="col">([^<]+)<\/th>/g)].map(x=>x[1]),['Left hand','Right hand']);
assert.deepEqual([...html.matchAll(/<th scope="row"><span>([^<]+)<\/span><small>([^<]+)</g)].map(x=>x[1]+' '+x[2]),['Left hand 0% correct','Right hand 100% correct']);
assert.deepEqual([...html.matchAll(/class="(ss-diag|ss-off)"/g)].map(x=>x[1]),['ss-diag','ss-off','ss-off','ss-diag'],'diagonal follows identity');
// Without class labels the summary derives indices from results and marks the names unverified.
s=S.build({},[r('a:1','s1',1,1),r('a:2','s2',0,1)]);
assert.equal(s.session.classNamesVerified,false);assert.deepEqual(s.classes.map(c=>c.display),['Class 0','Class 1']);assert.match(V.renderHTML(s),/Class names not verified/);
// Round-trip latency stays in the snapshot (export) only; it is not shown in the simplified summary.
s=S.build({classLabels:LR},[r('a:1','s1',0,0,{roundTripMs:1200}),r('a:2','s2',1,1,{roundTripMs:1400}),r('a:3','s3',1,1)]);
assert.deepEqual(s.metrics.latency,{count:2,meanMs:1300,minMs:1200,maxMs:1400});assert.doesNotMatch(V.renderHTML(s),/round trip|latency/);
// Synthetic preview wording lives outside the card (preview page bar); the card itself never mentions it or "validated hardware results". HTML is escaped.
s=S.build({classLabels:LR,source:'synthetic-preview',modelName:'<b>x</b>',note:'<i>n</i>'},[r('a:1','s1',0,0)]);
html=V.renderHTML(s);assert.doesNotMatch(html,/ynthetic|validated hardware results/);assert.equal(s.session.source,'synthetic-preview');assert.match(html,/&lt;i&gt;n&lt;\/i&gt;/);assert.doesNotMatch(html,/<i>n<\/i>|<b>x<\/b>/);
assert.doesNotMatch(V.renderHTML(S.build({classLabels:LR},[r('a:1','s1',0,0)])),/ynthetic|validated hardware/);
assert.match(V.renderEnding({request:7,trialId:'t'}),/request 7 \(t\)/);
// Streaming record adapter: attempt identity = session nonce + request; recording identity = dataset + trial_id; subject from the dataset.
const dataset={model_sha256:M.models[0].modelSha256,trials:[{index:0,trial_id:'S1/1test/run0/cue10',ground_truth:0,source_trial:{subject:1,session:'1test',run:'0',cue_sample:10,artifact:false}},{index:1,trial_id:'S1/1test/run0/cue20',ground_truth:1,source_trial:{subject:1,session:'1test',run:'0',cue_sample:20,artifact:true}}]};
const records=[{session:'11',request:1,trial:0,trial_id:'S1/1test/run0/cue10',ground_truth:0,class_id:0,logits:[1,-1]},{session:'11',request:2,trial:1,trial_id:'S1/1test/run0/cue20',ground_truth:1,class_id:0,logits:[1,-1]},{session:'22',request:1,trial:0,trial_id:'S1/1test/run0/cue10',ground_truth:0,class_id:1,logits:[-1,1]}];
const adapted=S.adaptStreamingRecords(records,{datasetId:'bci',dataset,roundTrips:new Map([['11:2',950]])});
assert.deepEqual(adapted.map(a=>[a.trialId,a.sourceTrialId,a.subjectId,a.roundTripMs]),[['11:1','bci|S1/1test/run0/cue10',1,null],['11:2','bci|S1/1test/run0/cue20',1,950],['22:1','bci|S1/1test/run0/cue10',1,null]]);
s=S.build({classLabels:LR},adapted);assert.equal(s.metrics.total,3);assert.equal(s.metrics.unique.count,2);assert.equal(s.metrics.unique.correct,1);assert.equal(s.metrics.correct,1);
// Verified metadata registry resolves by firmware hash and refuses a dataset from another subject or model.
const firmware={model_sha256:M.models[0].modelSha256,build:'eeg-stream-v1.0'};
assert.equal(M.resolve({firmware,dataset}).modelId,'model2-subject1-tinyeegnet-int8-ethosu85');
assert.equal(M.resolve({firmware:{model_sha256:'0'.repeat(64)},dataset}),null);
assert.equal(M.resolve({firmware,dataset:{...dataset,trials:[{...dataset.trials[0],source_trial:{...dataset.trials[0].source_trial,subject:2}}]}}),null);
assert.equal(M.resolve({firmware,dataset:{...dataset,model_sha256:'1'.repeat(64)}}),null);
assert.deepEqual(S.normalizeClassLabels(M.models[0].classLabels).map(c=>[c.index,c.name]),[[0,'left_hand'],[1,'right_hand']]);
assert.throws(()=>S.normalizeClassLabels([{index:0,name:'a'},{index:0,name:'b'}]),/Duplicate class index/);
console.log('PASS: accuracy/confusion/per-class (incl. zero-observation class), repeats and unique-first, cross-subject identity, duplicates, empty/interrupted sessions, missing metadata, reversed class map, latency labeling, adapter identities and verified metadata registry.');
