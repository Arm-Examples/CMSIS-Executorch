(()=>{
'use strict';
// Standalone UI integration fixture: NOT actual EEG data or board inference.
const $=id=>document.getElementById(id), INTERVAL=2500,DURATION=60000;
const predFixture=[0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0];
const cueFixture =[0,0,1,1,0,1,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,0];
let source="mock";
let correctPredictions=0,scoredPredictions=0;
function updateTotals(){setText('accuracyValue',scoredPredictions?Math.round(100*correctPredictions/scoredPredictions)+'%':'—');setText('correctValue',correctPredictions+' / '+scoredPredictions);setText('accuracyLabel',source==='mock'?'Accuracy':'Accuracy');}
// Only score a label explicitly paired with THIS prediction's EEG window.
// Hardware adapter must supply ground_truth: 0|1 on the prediction message.
// A separately displayed cue can belong to a different window and is never used here.
function scorePrediction(message){if(message.ground_truth!==0&&message.ground_truth!==1)return;scoredPredictions++;if(message.class_id===message.ground_truth)correctPredictions++;updateTotals();}
let playing=false, elapsed=0,lastFrame=null,nextPrediction=0,fixtureIndex=0,lastPredictionTime=-1,lastClass=null, cursorFrac=.5,dwell=0,hitCount=0,currentHit=null,lastViz=0;
const panel=$('targetArea'),mouse=$('mouse'),person=$('person'),truth=$('groundTruth'),predict=$('predictText');
function setText(id,s){$(id).textContent=s}
function humanCue(label){person.classList.remove('raiseL','raiseR');if(label===0)person.classList.add('raiseL');if(label===1)person.classList.add('raiseR');truth.textContent=label===0?'LEFT':label===1?'RIGHT':'—';setText('cueArrow',label===0?'←':label===1?'→':'↔');document.querySelector('.cue-direction').setAttribute('data-direction',label===0?'left':label===1?'right':'none');}
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
function reset(){document.querySelector('.current-pred').setAttribute('data-direction','none');correctPredictions=0;scoredPredictions=0;updateTotals();playing=false;elapsed=0;lastFrame=null;nextPrediction=0;fixtureIndex=0;lastPredictionTime=-1;lastClass=null;cursorFrac=.5;dwell=0;hitCount=0;currentHit=null;$('playBtn').textContent='▶ Play';$('playBtn').classList.remove('running');setText('clock','00:00 / 01:00');$('progress').style.width='0%';$('predictionAge').style.width='0%';setText('predTitle','Waiting');setText('predDesc','No predictions yet');setText('directionIcon','↔');$('directionIcon').style.color='#b3c7dd';predict.textContent='AWAITING PREDICTION';predict.className='predict wait';$('leftTarget').classList.remove('active','hit');$('rightTarget').classList.remove('active','hit');$('popup').classList.remove('show');humanCue(null);updateCursor();}
function updateCursor(){let w=panel.getBoundingClientRect().width;mouse.style.left=(cursorFrac*w)+'px';}
function processFixture(){if(source!=="mock")return;if(elapsed>=nextPrediction){while(elapsed>=nextPrediction&&fixtureIndex<predFixture.length){const i=fixtureIndex++;feedPrediction({class_id:predFixture[i],ground_truth:cueFixture[i],timestamp_ms:nextPrediction});humanCue(cueFixture[i]);nextPrediction+=INTERVAL;}}}
function updateMotion(dt){if(!playing||lastClass===null||elapsed-lastPredictionTime>INTERVAL+900)return;
 const w=panel.clientWidth, speed=156; // px/sec; smooth per-frame motion, predictions update 0.4 Hz.
 cursorFrac=Math.min(.78,Math.max(.22,cursorFrac+(lastClass===0?-1:1)*(speed*(dt/1000))/w));
 const target=lastClass===0?'left':'right';const targetEl= $(target+'Target');const tr=targetEl.getBoundingClientRect(), pr=panel.getBoundingClientRect(),pos=pr.left+cursorFrac*w;
 if(pos>tr.left&&pos<tr.right){if(currentHit!==target){currentHit=target;dwell=0}else dwell+=dt; if(dwell>650&&!targetEl.classList.contains('hit')){targetEl.classList.add('hit');hitCount++;showHit(target);}}else{currentHit=null;dwell=0;$('leftTarget').classList.remove('hit');$('rightTarget').classList.remove('hit')}
}
function showHit(target){const el=$('popup');el.textContent=target.toUpperCase()+' target reached · '+hitCount;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),1200)}
function render(dt){updateCursor();$('progress').style.width=(source==='mock'?elapsed/DURATION*100:0)+'%';setText('clock',mmss(elapsed)+(source==='mock'?' / 01:00':' · LIVE'));const age=lastPredictionTime<0?0:Math.min(100,(elapsed-lastPredictionTime)/INTERVAL*100);$('predictionAge').style.width=age+'%';if(lastClass!==null){predict.className='predict '+(lastClass===0?'left':'right');predict.textContent=(lastClass===0?'← LEFT':'→ RIGHT');}}
function updateStale(){if(source==='hardware'&&lastPredictionTime>=0&&elapsed-lastPredictionTime>INTERVAL+900){predict.className='predict wait';predict.textContent='STALE BOARD OUTPUT · CURSOR HELD';}}
function loop(now){if(lastFrame===null)lastFrame=now;const dt=Math.min(now-lastFrame,100);lastFrame=now;if(playing){elapsed=source==="mock"?Math.min(DURATION,elapsed+dt):elapsed+dt;processFixture();updateMotion(dt);render(dt);updateStale();if(source==="mock"&&elapsed>=DURATION){playing=false;$('playBtn').textContent='▶ Replay';$('playBtn').classList.remove('running');predict.textContent='SESSION COMPLETE';}}requestAnimationFrame(loop)}
$('playBtn').addEventListener('click',()=>{if(source==='mock'&&elapsed>=DURATION)reset();playing=!playing;$('playBtn').textContent=playing?'Ⅱ Pause':'▶ Play';$('playBtn').classList.toggle('running',playing);if(playing&&elapsed===0&&source==='mock'){processFixture();render(0);}});
$('resetBtn').addEventListener('click',reset);
$('drawerBtn').addEventListener('click',()=>{const isOpen=$('drawer').classList.toggle('open');$('drawerBtn').setAttribute('aria-expanded',String(isOpen));$('drawerBtn').querySelector('span span').textContent=isOpen?'(expanded)':'';});
window.addEventListener('resize',updateCursor);
// Prototype hook. The bridge can call window.ArmEEGDemo.onBoardPrediction({class_id:0|1})
window.ArmEEGDemo={
 onBoardPrediction:feedPrediction,
 onGroundTruth(label){if(label===0||label===1||label===null)humanCue(label)},
 setSource(value){if(value!=='mock'&&value!=='hardware')throw new Error('Source must be mock or hardware');reset();source=value;updateTotals();document.querySelector('.pill.mock').textContent=value==='mock'?'Mock mode':'Hardware mode';setText('boardStat',value==='mock'?'Board not connected':'Awaiting board connection');setText('truthInfo',value==='mock'?'Ground-truth cue':'Ground-truth cue');document.querySelector('.optional').textContent=value==='mock'?'Model output: every 2.5 s':'Board output · nominal 2.5 s';realEEG=null;setText('mapSource',value==='mock'?'EEG amplitude · relative RMS':'Awaiting EEG samples');document.querySelector('.trace-note').textContent=value==='mock'?'':'Awaiting EEG samples';drawScalp();drawSignals();},
 setConnection(connected){setText('boardStat',connected?'Board connected':'Board disconnected');if(!connected){lastClass=null;lastPredictionTime=-1;setText('predTitle','Waiting');setText('predDesc','No board connection');predict.textContent='AWAITING CONNECTION';predict.className='predict wait';$('leftTarget').classList.remove('active');$('rightTarget').classList.remove('active');}},
 reset:reset,
 getState:()=>({source,playing,elapsed_ms:elapsed,current_class:lastClass,cursor_x:cursorFrac})
};
// Real EEG adapter accepts channel-keyed arrays; map uses RMS, independent of cue/class.
let realEEG=null;
const montage={"Fz":[110,48],"FC3":[58,83],"FC1":[84,83],"FCz":[110,83],"FC2":[136,83],"FC4":[162,83],"C5":[32,119],"C3":[58,119],"C1":[84,119],"Cz":[110,119],"C2":[136,119],"C4":[162,119],"C6":[188,119],"CP3":[58,155],"CP1":[84,155],"CPz":[110,155],"CP2":[136,155],"CP4":[162,155],"P1":[84,190],"Pz":[110,190],"P2":[136,190],"POz":[110,218]};
Object.entries(montage).forEach(([name,xy])=>{const circle=document.createElementNS('http://www.w3.org/2000/svg','circle');circle.id='heat-'+name;circle.setAttribute('cx',xy[0]);circle.setAttribute('cy',xy[1]);circle.setAttribute('r','45');circle.setAttribute('fill','url(#heat)');$('heatField').appendChild(circle)});
function samplesFor(name,i){i=Object.keys(montage).indexOf(name);if(i<0)return [];if(source==='hardware'&&!realEEG)return [];if(realEEG)return realEEG[name]||[];return Array.from({length:160},(_,j)=>{const t=elapsed/1000*2+j*.085;return (1+.5*Math.sin(elapsed/3000+i))*(3.1*Math.sin(t*(1.8+i*.4)+i)+1.6*Math.sin(t*(4.5+i*.12)+i*.4)+.75*Math.sin(t*9+i*1.7))})}
function drawScalp(){const vals=Object.keys(montage).map((name,i)=>{const a=samplesFor(name,i);return a.length?Math.sqrt(a.reduce((s,x)=>s+x*x,0)/a.length):0});const max=Math.max(...vals,1e-9);Object.keys(montage).forEach((name,i)=>{const c=$('heat-'+name);c.setAttribute('opacity',String(vals[i]/max));c.setAttribute('r',String(28+26*vals[i]/max))})}
window.ArmEEGDemo.onEEGFrame=channels=>{if(!channels||typeof channels!=='object')return;const clean={};for(const name of Object.keys(montage)){const a=channels[name];if(Array.isArray(a)&&a.length&&a.every(Number.isFinite))clean[name]=a.slice(-500)}if(!Object.keys(clean).length)return;realEEG=clean;setText('mapSource','Live EEG · relative channel RMS');document.querySelector('.trace-note').textContent='';drawScalp();drawSignals()};
$('participant').addEventListener('change',()=>{realEEG=null;reset();setText('mapSource',source==='mock'?'EEG amplitude · relative RMS':'Awaiting EEG samples');document.querySelector('.trace-note').textContent=source==='mock'?'':'Awaiting EEG samples';drawScalp();drawSignals()});
const colors=['#9f93c8','#779fcb','#80b9b7','#84b09c','#b1a1c8','#b7a47c'];const names=Object.keys(montage);const traces=$('traces');const traceColumns=Array.from({length:3},(_,i)=>{const col=document.createElement('div');col.className='trace-column';traces.appendChild(col);return col});names.forEach((n,i)=>{const tr=document.createElement('div');tr.className='trace';tr.id='trace-'+n;tr.innerHTML='<div class="trace-label">'+n+'</div><svg viewBox="0 0 600 19" preserveAspectRatio="none"><path stroke="'+colors[i%colors.length]+'" stroke-width="1.5" fill="none" id="wave'+i+'"></path></svg>';traceColumns[i<8?0:i<15?1:2].appendChild(tr)});
function drawSignals(){if(!$('drawer').classList.contains('open'))return;names.forEach((n,i)=>{const out=[],a=samplesFor(n,i);const hasSignal=a.length>=2&&Math.max(...a)>Math.min(...a);$('trace-'+n).style.display=hasSignal?'flex':'none';if(!hasSignal){$('wave'+i).setAttribute('d','');return;}const peak=Math.max(...a.map(Math.abs),Number.MIN_VALUE);for(let j=0;j<160;j++){const x=j/159*600,y=9.5-(a[Math.floor(j/160*a.length)]||0)/peak*7;out.push((j?'L':'M')+x.toFixed(1)+' '+y.toFixed(1))};$('wave'+i).setAttribute('d',out.join(' '))})}

setInterval(()=>{if(playing){drawScalp();drawSignals()}},70);reset();drawScalp();drawSignals();requestAnimationFrame(loop);
})();
