(function(root){'use strict';
async function load(base='streaming/data/'){
 const json=await fetch(base+'metadata.json');if(!json.ok)throw Error('Local Session2 data is missing; run prepare_streaming_assets.py');const meta=await json.json();
 const expectedChannels=['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz'];
 if(JSON.stringify(meta.channel_order)!==JSON.stringify(expectedChannels))throw Error('EEG channel order mismatch');
 if(meta.schema!==1||meta.model_sha256!==EdgeStreamProtocol.model||meta.trials.length!==144||new Set(meta.trials.map(t=>t.trial_id)).size!==144||meta.channel_order.length!==22||meta.trials.some((t,i)=>t.index!==i||![0,1].includes(t.ground_truth)||t.source_trial.subject!==1||t.source_trial.session!=='1test'||t.trial_id!==`S1/1test/run${t.source_trial.run}/cue${t.source_trial.cue_sample}`||typeof t.source_trial.artifact!=='boolean'))throw Error('Invalid Session2 metadata');
 const buffers=await Promise.all(['inputs.bin','eeg.f32'].map(async name=>{const r=await fetch(base+name);if(!r.ok)throw Error('EEG asset missing');const b=new Uint8Array(await r.arrayBuffer());if(await EdgeStream.hash(b)!==meta.assets[name])throw Error('EEG asset integrity mismatch');return b;}));
 if(buffers[0].length!==144*11000||buffers[1].length!==144*11000*4)throw Error('EEG asset shape mismatch');
 const channelCache=new Map();
 return {...meta,input:async i=>buffers[0].slice(i*11000,(i+1)*11000),channels(i){if(channelCache.has(i))return channelCache.get(i);const view=new DataView(buffers[1].buffer,buffers[1].byteOffset,buffers[1].byteLength),channels={};for(let c=0;c<22;c++)channels[meta.channel_order[c]]=Array.from({length:500},(_,j)=>view.getFloat32(((i*22+c)*500+j)*4,true));for(const a of Object.values(channels))Object.freeze(a);Object.freeze(channels);channelCache.set(i,channels);return channels;}};
}
root.EdgeStreamDataset={load};
})(globalThis);
