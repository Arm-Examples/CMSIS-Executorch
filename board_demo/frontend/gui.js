(()=>{
'use strict';
// Standalone UI integration fixture: NOT actual EEG data or board inference.
const $=id=>document.getElementById(id), INTERVAL=2500,DURATION=60000;
const predFixture=[0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0];
const cueFixture =[0,0,1,1,0,1,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,0];
let source="mock";
let correctPredictions=0,scoredPredictions=0;
function updateTotals(){setText('accuracyValue',scoredPredictions?Math.round(100*correctPredictions/scoredPredictions)+'%':'—');setText('correctValue',correctPredictions+' / '+scoredPredictions);setText('accuracyLabel',window.EdgeStreamingMode?'Visualized accuracy':'Accuracy');}
// Only score a label explicitly paired with THIS prediction's EEG window.
// Hardware adapter must supply ground_truth: 0|1 on the prediction message.
// A separately displayed cue can belong to a different window and is never used here.
function scorePrediction(message){if(message.ground_truth!==0&&message.ground_truth!==1)return;scoredPredictions++;if(message.class_id===message.ground_truth)correctPredictions++;updateTotals();}
let playing=false, elapsed=0,lastFrame=null,nextPrediction=0,fixtureIndex=0,lastPredictionTime=-1,lastClass=null, cursorFrac=.5,dwell=0,hitCount=0,currentHit=null,lastViz=0;
const playback=EdgePlayback.createPlayback({
 continuous:!!window.EdgeStreamingMode,
 onTrial(message,index){feedPrediction(message);humanCue([0,1].includes(message.ground_truth)?message.ground_truth:null);setText('truthInfo',[0,1].includes(message.ground_truth)?'Verified Model 2 task · trial '+message.trial:'Ground truth unavailable · person neutral');setText('predDesc','Trial '+(message.trial??index)+' · Class '+message.class_id);if(message.streaming){window.ArmEEGDemo.onEEGFrame(message.eeg_channels,{origin:"prerecorded",trial_id:message.trial_id,uart_trial:message.trial});setText("truthInfo","Verified Session2 task · "+message.trial_id+(message.artifact?" · artifact flagged":""));}else window.EdgeRecordedEEG?.showTrial(message);},
 onPosition(x){cursorFrac=x;},
 onComplete(){playing=false;humanCue(null);syncPlayback();}
});
function syncPlayback(){
 const p=playback.state();
 $('playBtn').textContent=playing?'Ⅱ Pause':source==='hardware'&&p.results.length&&p.complete?'▶ Replay results':'▶ Play';
 $('playBtn').classList.toggle('running',playing);$('playBtn').setAttribute('aria-pressed',String(playing));
 const state=source==='hardware'?(p.results.length&&p.complete?'Playback complete':playing?(p.pending?'Playing trial '+(p.index+1)+' / '+p.results.length:'Playing · waiting for results'):'Paused · '+p.pending+' queued')+' · delayed visualization (1.8 s/trial)':playing?'Playing mock preview':'Paused mock preview';
 setText('playbackState',state);
 setText('cursorInfo',source==='hardware'?state:'TinyEEGNet · 2.5 s');
}
const panel=$('targetArea'),mouse=$('mouse'),person=$('person'),truth=$('groundTruth'),predict=$('predictText');
function setText(id,s){$(id).textContent=s}
function humanCue(label){person.classList.remove('raiseL','raiseR');// SVG arm names are anatomical: swap only the displayed side for the viewer.
if(label===0)person.classList.add('raiseR');if(label===1)person.classList.add('raiseL');truth.textContent=label===0?'LEFT':label===1?'RIGHT':'—';setText('cueArrow',label===0?'←':label===1?'→':'↔');document.querySelector('.cue-direction').setAttribute('data-direction',label===0?'left':label===1?'right':'none');}
function feedPrediction(message){ // Replace only this adapter with UART/bridge output handler later.
 if(message.class_id!==0&&message.class_id!==1)return;
 lastClass=message.class_id;lastPredictionTime=elapsed;scorePrediction(message);
 const left=lastClass===0;document.querySelector('.current-pred').setAttribute('data-direction',left?'left':'right');
 setText('predTitle',left?'LEFT':'RIGHT');setText('predDesc','Class '+lastClass);
 setText('directionIcon',left?'←':'→');$('directionIcon').style.color=left?'#75a994':'#8aaaca';
 predict.className='predict '+(left?'left':'right');predict.textContent=(left?'← LEFT':'→ RIGHT');
 $('leftTarget').classList.toggle('active',left);$('rightTarget').classList.toggle('active',!left);
 }
function mmss(ms){let s=Math.floor(ms/1000);return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}
function reset(){playback.reset();document.querySelector('.current-pred').setAttribute('data-direction','none');correctPredictions=0;scoredPredictions=0;updateTotals();playing=false;elapsed=0;lastFrame=null;nextPrediction=0;fixtureIndex=0;lastPredictionTime=-1;lastClass=null;cursorFrac=.5;dwell=0;hitCount=0;currentHit=null;$('playBtn').textContent='▶ Play';$('playBtn').classList.remove('running');setText('clock',source==='mock'?'00:00 / 01:00':'00:00 · VISUAL PLAYBACK');$('progress').style.width='0%';$('predictionAge').style.width='0%';setText('predTitle','Waiting');setText('predDesc','No predictions yet');setText('directionIcon','↔');$('directionIcon').style.color='#b3c7dd';predict.textContent='AWAITING PREDICTION';predict.className='predict wait';$('leftTarget').classList.remove('active','hit');$('rightTarget').classList.remove('active','hit');$('popup').classList.remove('show');humanCue(null);updateCursor();syncPlayback();if(source==='hardware')window.EdgeRecordedEEG?.clear();}
function updateCursor(){let w=panel.getBoundingClientRect().width;mouse.style.left=(cursorFrac*w)+'px';}
function processFixture(){if(source!=="mock")return;if(elapsed>=nextPrediction){while(elapsed>=nextPrediction&&fixtureIndex<predFixture.length){const i=fixtureIndex++;feedPrediction({class_id:predFixture[i],ground_truth:cueFixture[i],timestamp_ms:nextPrediction});humanCue(cueFixture[i]);nextPrediction+=INTERVAL;}}}
function updateMotion(dt){if(source==='hardware')return;if(!playing||lastClass===null||elapsed-lastPredictionTime>INTERVAL+900)return;
 const w=panel.clientWidth, speed=156; // px/sec; smooth per-frame motion, predictions update 0.4 Hz.
 cursorFrac=Math.min(.78,Math.max(.22,cursorFrac+(lastClass===0?-1:1)*(speed*(dt/1000))/w));
 const target=lastClass===0?'left':'right';const targetEl= $(target+'Target');const tr=targetEl.getBoundingClientRect(), pr=panel.getBoundingClientRect(),pos=pr.left+cursorFrac*w;
 if(pos>tr.left&&pos<tr.right){if(currentHit!==target){currentHit=target;dwell=0}else dwell+=dt; if(dwell>650&&!targetEl.classList.contains('hit')){targetEl.classList.add('hit');hitCount++;showHit(target);}}else{currentHit=null;dwell=0;$('leftTarget').classList.remove('hit');$('rightTarget').classList.remove('hit')}
}
function showHit(target){const el=$('popup');el.textContent=target.toUpperCase()+' target reached · '+hitCount;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),1200)}
function render(dt){updateCursor();const p=playback.state();if(source==='hardware'&&realEEG){eegProgress=Math.min(1,p.trial_elapsed_ms/p.duration_ms);drawSignals();}$('progress').style.width=(source==='mock'?elapsed/DURATION*100:p.results.length?(p.index+(p.active_trial!==null?p.trial_elapsed_ms/p.duration_ms:0))/p.results.length*100:0)+'%';setText('clock',mmss(elapsed)+(source==='mock'?' / 01:00':' · VISUAL PLAYBACK'));const age=lastPredictionTime<0?0:Math.min(100,(elapsed-lastPredictionTime)/INTERVAL*100);$('predictionAge').style.width=age+'%';if(lastClass!==null){predict.className='predict '+(lastClass===0?'left':'right');predict.textContent=(lastClass===0?'← LEFT':'→ RIGHT');}}
function loop(now){if(lastFrame===null)lastFrame=now;const dt=Math.min(now-lastFrame,100);lastFrame=now;if(playing){elapsed=source==="mock"?Math.min(DURATION,elapsed+dt):elapsed+dt;processFixture();if(source==='hardware')playback.tick(dt);updateMotion(dt);render(dt);syncPlayback();if(source==="mock"&&elapsed>=DURATION){playing=false;$('playBtn').textContent='▶ Replay';$('playBtn').classList.remove('running');predict.textContent='SESSION COMPLETE';}}requestAnimationFrame(loop)}
$('playBtn').addEventListener('click',()=>{const p=playback.state();if(source==='mock'&&elapsed>=DURATION||source==='hardware'&&p.results.length&&p.complete)reset();playing=!playing;syncPlayback();if(playing&&elapsed===0&&source==='mock'){processFixture();render(0);}});
$('resetBtn').addEventListener('click',reset);
$('drawerBtn').addEventListener('click',()=>{const isOpen=$('drawer').classList.toggle('open');$('drawerBtn').setAttribute('aria-expanded',String(isOpen));$('drawerBtn').querySelector('span span').textContent=isOpen?'(expanded)':'';drawSignals();});
window.addEventListener('resize',updateCursor);
// Prototype hook. The bridge can call window.ArmEEGDemo.onBoardPrediction({class_id:0|1})
window.ArmEEGDemo={
 onBoardPrediction(message){if(source==='hardware'){playback.push(message);syncPlayback();}else feedPrediction(message);},
 onGroundTruth(label){if(label===0||label===1||label===null)humanCue(label)},
 setSource(value){if(value!=='mock'&&value!=='hardware')throw new Error('Source must be mock or hardware');playback.clear();reset();source=value;playing=value==='hardware';updateTotals();syncPlayback();document.querySelector('.pill.mock').textContent=value==='mock'?'Mock mode':'Hardware mode';setText('boardStat',value==='mock'?'Board not connected':'Awaiting board connection');setText('truthInfo',value==='mock'?'Ground-truth cue':'Ground truth unavailable · select verified metadata');document.querySelector('.optional').textContent=value==='mock'?'Model output: every 2.5 s':'Delayed visualization · 1.8 s/trial';realEEG=null;setText('mapSource',value==='mock'?'EEG amplitude · relative RMS':'Awaiting EEG samples');document.querySelector('.trace-note').textContent=value==='mock'?'':'Awaiting EEG samples';drawScalp();drawSignals();},
 finishCapture(){playback.finish();syncPlayback();},
 setConnection(connected){if(!connected&&source==='hardware')playback.finish();setText('boardStat',connected?'Board connected':'Board disconnected');syncPlayback();},
 setPlaying(value){playing=!!value;syncPlayback();},
 reset:reset,
 getState:()=>({source,playing,elapsed_ms:elapsed,current_class:lastClass,cursor_x:cursorFrac,playback:playback.state(),accuracy:{correct:correctPredictions,scored:scoredPredictions}})
};
// Real EEG adapter accepts channel-keyed arrays; map uses RMS, independent of cue/class.
let realEEG=null,eegProgress=0;
const montage={"Fz":[110,48],"FC3":[58,83],"FC1":[84,83],"FCz":[110,83],"FC2":[136,83],"FC4":[162,83],"C5":[32,119],"C3":[58,119],"C1":[84,119],"Cz":[110,119],"C2":[136,119],"C4":[162,119],"C6":[188,119],"CP3":[58,155],"CP1":[84,155],"CPz":[110,155],"CP2":[136,155],"CP4":[162,155],"P1":[84,190],"Pz":[110,190],"P2":[136,190],"POz":[110,218]};
Object.entries(montage).forEach(([name,xy])=>{const circle=document.createElementNS('http://www.w3.org/2000/svg','circle');circle.id='heat-'+name;circle.setAttribute('cx',xy[0]);circle.setAttribute('cy',xy[1]);circle.setAttribute('r','45');circle.setAttribute('fill','url(#heat)');$('heatField').appendChild(circle)});
function samplesFor(name,i){i=Object.keys(montage).indexOf(name);if(i<0)return [];if(source==='hardware'&&!realEEG)return [];if(realEEG)return realEEG[name]||[];return Array.from({length:160},(_,j)=>{const t=elapsed/1000*2+j*.085;return (1+.5*Math.sin(elapsed/3000+i))*(3.1*Math.sin(t*(1.8+i*.4)+i)+1.6*Math.sin(t*(4.5+i*.12)+i*.4)+.75*Math.sin(t*9+i*1.7))})}
function drawScalp(){const vals=Object.keys(montage).map((name,i)=>{const a=samplesFor(name,i);return a.length?Math.sqrt(a.reduce((s,x)=>s+x*x,0)/a.length):0});const max=Math.max(...vals,1e-9);Object.keys(montage).forEach((name,i)=>{const c=$('heat-'+name);c.setAttribute('opacity',String(vals[i]/max));c.setAttribute('r',String(28+26*vals[i]/max))})}
window.ArmEEGDemo.clearEEG=(label='Awaiting EEG samples')=>{realEEG=null;eegProgress=0;setText('mapSource',label);document.querySelector('.trace-note').textContent=label;drawScalp();drawSignals();};
window.ArmEEGDemo.onEEGFrame=(channels,metadata={})=>{if(!channels||typeof channels!=='object')return;const clean={};for(const name of Object.keys(montage)){const a=channels[name];if(Array.isArray(a)&&a.length&&a.every(Number.isFinite))clean[name]=a.slice(-500)}if(!Object.keys(clean).length)return;realEEG=clean;eegProgress=0;const label=metadata.origin==='prerecorded'?'Prerecorded EEG · trial '+metadata.uart_trial+' · filtered µV':'Live EEG · relative channel RMS';setText('mapSource',label);$('mapSource').setAttribute('title',metadata.origin==='prerecorded'?metadata.trial_id+' · 250 Hz, 2 s · 7–30 Hz causal filtered µV, before normalization/quantization · relative per-channel RMS; not cortical source localization or clinical measurement':label);$('traces').setAttribute('aria-label',label+' · 22 channels');document.querySelector('.trace-note').textContent=metadata.origin==='prerecorded'?label:'';drawScalp();drawSignals()};
$('participant').addEventListener('change',()=>{realEEG=null;reset();setText('mapSource',source==='mock'?'EEG amplitude · relative RMS':'Awaiting EEG samples');document.querySelector('.trace-note').textContent=source==='mock'?'':'Awaiting EEG samples';drawScalp();drawSignals()});
const colors=['#9f93c8','#779fcb','#80b9b7','#84b09c','#b1a1c8','#b7a47c'];const names=Object.keys(montage);const traces=$('traces');const traceColumns=Array.from({length:3},(_,i)=>{const col=document.createElement('div');col.className='trace-column';traces.appendChild(col);return col});names.forEach((n,i)=>{const tr=document.createElement('div');tr.className='trace';tr.id='trace-'+n;tr.innerHTML='<div class="trace-label">'+n+'</div><svg viewBox="0 0 600 19" preserveAspectRatio="none"><path stroke="'+colors[i%colors.length]+'" stroke-width="1.5" fill="none" id="wave'+i+'"></path></svg>';traceColumns[i<8?0:i<15?1:2].appendChild(tr)});
function drawSignals(){
 if(!$('drawer').classList.contains('open'))return;
 names.forEach((name,i)=>{
  const a=samplesFor(name,i),hasSignal=a.length>=2&&Math.max(...a)>Math.min(...a);
  $('trace-'+name).style.display=hasSignal?'flex':'none';
  if(!hasSignal){$('wave'+i).setAttribute('d','');return;}
  // Reveal original samples, retaining a fixed full-trial axis and amplitude scale.
  const peak=Math.max(...a.map(Math.abs),Number.MIN_VALUE),recorded=source==='hardware'&&realEEG;
  const count=recorded?1+Math.floor(eegProgress*(a.length-1)):160,out=[];
  for(let j=0;j<count;j++){
   const x=j/(recorded?a.length-1:159)*600,y=9.5-a[recorded?j:Math.floor(j/160*a.length)]/peak*7;
   out.push((j?'L':'M')+x.toFixed(1)+' '+y.toFixed(1));
  }
  $('wave'+i).setAttribute('d',out.join(' '));
 });
}

setInterval(()=>{if(playing&&source==='mock'){drawScalp();drawSignals()}},70);reset();drawScalp();drawSignals();requestAnimationFrame(loop);
})();
