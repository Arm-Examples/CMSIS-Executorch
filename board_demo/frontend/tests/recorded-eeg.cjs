const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),crypto=require('node:crypto');
const ctx={};ctx.window=ctx;let shown=null,cleared='';ctx.ArmEEGDemo={onEEGFrame(channels,metadata){shown={channels,metadata};},clearEEG(reason){shown=null;cleared=reason;}};vm.createContext(ctx);
for(const file of ['trial-metadata.js','recorded-eeg/model2-eeg-data.js','recorded-eeg.js'])vm.runInContext(fs.readFileSync(require.resolve('../'+file),'utf8'),ctx);
const bundle=ctx.EdgeRecordedEEGData,profile=ctx.EdgeTrialMetadata,api=ctx.EdgeRecordedEEG;
const provenance=JSON.parse(fs.readFileSync(require.resolve('../recorded-eeg/provenance.json'),'utf8'));
assert.equal(crypto.createHash('sha256').update(fs.readFileSync(require.resolve('../recorded-eeg/model2-eeg-data.js'))).digest('hex'),provenance.display_asset_sha256);
assert.equal(bundle.channel_order.length,22);assert.equal(new Set(bundle.channel_order).size,22);assert.equal(bundle.fs_hz,250);assert.equal(bundle.units,'µV');
const identities=['S1/0train/run0/cue20544','S1/0train/run5/cue750','S1/0train/run0/cue4671','S1/0train/run4/cue81163'];
// Deliberately out of order: matching uses trial/profile identity, never arrival order or class.
for(const trial of [3,0,2,1]){
 const message={trial,class_id:[0,1,1,1][trial],metadata_profile:profile.id};
 assert(api.showTrial(message));assert.equal(api.getState().trial_id,identities[trial]);
 const row=bundle.trials[identities[trial]];assert.equal(row.ground_truth,[0,0,1,1][trial]);assert.equal(shown.metadata.origin,'prerecorded');
 const bytes=Buffer.alloc(22*500*4);let offset=0;
 const rms=[];
 for(const name of bundle.channel_order){const a=shown.channels[name];assert.equal(a.length,500);assert(a.every(Number.isFinite));assert(Math.max(...a)>Math.min(...a));for(const x of a){bytes.writeFloatLE(x,offset);offset+=4;}rms.push(Math.sqrt(a.reduce((s,x)=>s+x*x,0)/a.length));}
 assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),row.samples_sha256_float32_le);
 rms.forEach((r,i)=>assert(Math.abs(r-row.rms_uv[i])<1e-12));
}
assert.equal(api.showTrial({trial:0,class_id:0}),false);assert.equal(shown,null);assert.match(cleared,/unverified/);
assert.equal(api.showTrial({trial:4,metadata_profile:profile.id}),false);
assert.equal(api.showTrial({trial:0,metadata_profile:'different-deployment'}),false);
const hash=profile.inputs_sha256;profile.inputs_sha256='different-window-bytes';assert.equal(api.showTrial({trial:0,metadata_profile:profile.id}),false);profile.inputs_sha256=hash;
const row=bundle.trials[identities[0]],cue=row.source_trial.cue_sample;row.source_trial.cue_sample++;assert.equal(api.showTrial({trial:0,metadata_profile:profile.id}),false);row.source_trial.cue_sample=cue;
const data=ctx.EdgeRecordedEEGData;ctx.EdgeRecordedEEGData=undefined;assert.equal(api.showTrial({trial:0,metadata_profile:profile.id}),false);ctx.EdgeRecordedEEGData=data;
assert(api.showTrial({trial:1,metadata_profile:profile.id}));api.clear();assert.equal(api.getState(),null);assert.equal(shown,null);
console.log('PASS: all four genuine 22×500 float32 windows, source/profile identity, shuffled trial order, data hashes and RMS; unknown/missing/mismatched metadata clears EEG without substitutes.');
