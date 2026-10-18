/* Session summary view: renders an EdgeSessionSummary snapshot. renderHTML() is pure (Node-testable);
 * create() binds it to the #sessionSummary overlay. Styles live in summary.css under #sessionSummary only.
 * Presentation only: provenance (model hash, session id, seed, round trips, full channel list) stays in the snapshot. */
(function(root){'use strict';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const UNAVAILABLE='Unavailable',TOP_CHANNELS=3;
const pct=x=>x===null||x===undefined||!Number.isFinite(x)?UNAVAILABLE:(Math.round(x*1000)/10).toFixed(1)+'%';
const pctWhole=x=>x===null||x===undefined||!Number.isFinite(x)?UNAVAILABLE:Math.round(x*100)+'%';
const num=x=>Number.isFinite(x)?x.toLocaleString('en-US'):UNAVAILABLE;
const plural=(n,word)=>num(n)+' '+word+(n===1?'':'s');
const uv=x=>Number.isFinite(x)?(x>=10?x.toFixed(1):x.toFixed(2)):UNAVAILABLE;
function duration(ms){if(!Number.isFinite(ms)||ms<0)return UNAVAILABLE;const s=Math.floor(ms/1000),h=Math.floor(s/3600),m=Math.floor(s%3600/60),r=s%60,p=n=>String(n).padStart(2,'0');return (h?h+':':'')+p(m)+':'+p(r)+(h?'':' min');}
const STATUS={completed:'Run complete',empty:'Run complete',interrupted:'Run interrupted',paused:'Run paused'};
function tile(value,label,detail,cls=''){return `<div class="ss-tile ${cls}"><strong>${esc(value)}</strong><span>${esc(label)}</span>${detail?`<small>${esc(detail)}</small>`:''}</div>`;}
/* Cell labels name the ACTUAL class from the class metadata (short names), never an assumed class order. */
function cellLabel(actual,predicted){return (actual.index===predicted.index?'Correct ':'Incorrect ')+String(actual.short).toLowerCase();}
function renderMatrix(summary){
 const m=summary.metrics,classes=summary.classes;
 // Display order follows the metadata `order` (e.g. Left, Right); counts are looked up by class identity, not position.
 const shown=[...classes].sort((a,b)=>(a.order-b.order)||(a.index-b.index)),pos=c=>classes.findIndex(x=>x.index===c.index);
 const head=`<thead><tr><th scope="col" class="ss-corner"><span>Actual ↓</span><span>Predicted →</span></th>${shown.map(c=>`<th scope="col">${esc(c.display)}</th>`).join('')}</tr></thead>`;
 const rows=shown.map(a=>{const r=pos(a),pc=m.perClass[r];return `<tr><th scope="row"><span>${esc(a.display)}</span><small>${pc.total?esc(pctWhole(pc.accuracy))+' correct':UNAVAILABLE}</small></th>${shown.map(p=>{const v=m.confusion[r][pos(p)],diag=a.index===p.index;return `<td class="${diag?'ss-diag':'ss-off'}" title="Actual ${esc(a.display)}, predicted ${esc(p.display)}: ${v}"><strong>${v}</strong><small>${esc(cellLabel(a,p))}</small></td>`;}).join('')}</tr>`;}).join('');
 const classNote=!summary.session.classNamesVerified&&classes.length?`<p class="ss-scope">Class names not verified; indices shown.</p>`:'';
 return `<section class="ss-matrix-panel" aria-label="Confusion matrix"><h3 class="ss-h3">Confusion Matrix</h3><div class="ss-matrix-wrap"><table class="ss-matrix" aria-label="Confusion matrix: actual class rows, predicted class columns">${head}<tbody>${rows}</tbody></table></div>${classNote}</section>`;
}
const SEP=' | ';
/* The ranking column; its supporting line and note (renderEEGFoot) sit directly beneath the ranking and wrap as needed. */
function renderEEG(e){
 const title=`<h3 class="ss-h3">Top EEG Channels</h3><div class="ss-eeg-label">RMS (${esc(e?.units||'µV')})</div>`;
 if(!e||!e.available)return `<section class="ss-eeg" aria-label="Top EEG channels">${title}<p class="ss-eeg-unavailable">EEG amplitude unavailable${e?.reason?`${SEP}${esc(e.reason)}`:''}</p>${renderEEGFoot(e)}</section>`;
 const top=e.channels.slice(0,TOP_CHANNELS),max=Math.max(...top.map(c=>c.rms),Number.MIN_VALUE);
 const rows=top.map((c,i)=>`<li class="ss-rank"><span class="ss-rank-n">${i+1}</span><span class="ss-rank-name">${esc(c.name)}</span><span class="ss-bar" role="img" aria-label="${esc(c.name)} ${esc(uv(c.rms))} ${esc(e.units)}"><span style="width:${(c.rms/max*100).toFixed(1)}%"></span></span><span class="ss-rank-value">${esc(uv(c.rms))} ${esc(e.units)}</span></li>`).join('');
 return `<section class="ss-eeg" aria-label="Top EEG channels">${title}<ol class="ss-ranks">${rows}</ol>${renderEEGFoot(e)}</section>`;
}
function renderEEGFoot(e){
 const note=`<p class="ss-scope">Amplitude ranking does not indicate neural activation.</p>`;
 if(!e||!e.available)return `<div class="ss-foot">${note}</div>`;
 const parts=['Prerecorded EEG',plural(e.recordings.used,'recording')];
 if(e.artifact.known&&e.artifact.flagged)parts.push(`${num(e.artifact.flagged)} artifact-flagged`);
 if(e.recordings.missing)parts.push(`${num(e.recordings.missing)} without EEG`);
 if(e.excludedChannels.length)parts.push(`${plural(e.excludedChannels.length,'channel')} excluded`);
 return `<div class="ss-foot"><p class="ss-eeg-meta">${esc(parts.join(SEP))}</p>${note}</div>`;
}
/* final=true: a finalized snapshot (New Run offered). final=false: a live view while paused or stopped;
 * resumable=true adds the Resume action (only when the controller can safely continue). */
function renderHTML(summary,{final=true,resumable=false}={}){
 const s=summary.session,m=summary.metrics,status=STATUS[s.status]||esc(s.status);
 // Model, experiment and source stay in the snapshot (summary.session); the panel shows only the title and the run status.
 const head=`<header class="ss-head"><h2 id="ssTitle">Session Summary</h2><span class="ss-pill ss-status-${esc(s.status)}">${status}</span></header>`;
 const i=s.interruption,interruption=s.status==='interrupted'&&i?`<div class="ss-alert" role="alert">${esc(i.message||'A hardware request did not resolve.')}${i.unresolvedRequest!==undefined&&i.unresolvedRequest!==null?`${SEP}Request ${esc(i.unresolvedRequest)}${i.trialId?` (${esc(i.trialId)})`:''} not counted.`:''}</div>`:'';
 const note=s.note?`<div class="ss-alert ss-alert-soft">${esc(s.note)}</div>`:'';
 const tiles=`<section class="ss-tiles" aria-label="Session overview">${tile(pct(m.accuracy),'Overall accuracy',m.total?`${num(m.correct)} correct${SEP}${num(m.incorrect)} incorrect`:'no completed trials','ss-hero')}${tile(num(m.total),'Trials','')}${tile(m.duration?duration(m.duration.ms):UNAVAILABLE,'Duration',m.duration?'Includes pauses':'')}</section>`;
 const line=m.total?`<p class="ss-line">${plural(m.unique.count,'unique recording')}${m.repeated?`${SEP}${plural(m.repeated,'repeat evaluation')}`:''}</p>`:'';
 const body=`<section class="ss-grid"><div class="ss-col">${renderMatrix(summary)}</div><div class="ss-col">${renderEEG(summary.eeg)}</div></section>`;
 // One primary action, centered: New Run on a finalized snapshot, Resume while paused; none when the run cannot continue.
 const button=final?`<button class="ss-btn ss-primary" id="ssNewRun" type="button">New Run</button>`:resumable?`<button class="ss-btn ss-primary" id="ssResume" type="button">Resume</button>`:'';
 const actions=button?`<footer class="ss-actions">${button}</footer>`:'';
 return `<div class="ss-body">${head}${interruption}${note}${tiles}${line}${body}${actions}</div>`;
}
function renderEnding({request=null,trialId=null,message=null}={}){
 const waiting=request!==null&&request!==undefined?`Waiting for hardware request ${esc(request)}${trialId?` (${esc(trialId)})`:''} to finish or reach its timeout. No new trials are scheduled.`:'Finishing the current visualization. No new trials are scheduled.';
 return `<div class="ss-body"><header class="ss-head"><h2 id="ssTitle">Session Summary</h2><span class="ss-pill ss-status-ending">Ending run…</span></header><p class="ss-sub">${message?esc(message):waiting}</p></div>`;
}
function renderMessage(title,text){return `<div class="ss-body"><header class="ss-head"><h2 id="ssTitle">${esc(title)}</h2></header><p class="ss-sub">${esc(text)}</p></div>`;}
/* Binds the renderer to the right-panel container. No focus management, backdrop or close control: the panel is
 * part of the page and the prediction elements stay mounted beneath it. */
function create(container,{onNewRun=()=>{},onResume=()=>{}}={}){
 if(!container)throw Error('Session summary container missing');
 container.addEventListener('click',event=>{const id=event.target?.closest?.('button')?.id;if(id==='ssNewRun')onNewRun();else if(id==='ssResume')onResume();});
 const raf=typeof requestAnimationFrame==='function'?requestAnimationFrame:f=>f();
 const api={
  render(summary,options){container.innerHTML=renderHTML(summary,options);api.show();},
  ending(detail){container.innerHTML=renderEnding(detail);api.show();},
  message(title,text){container.innerHTML=renderMessage(title,text);api.show();},
  show(){if(!container.hidden&&container.classList.contains('is-visible'))return;container.hidden=false;raf(()=>{if(!container.hidden)container.classList.add('is-visible');});},
  hide(){container.classList.remove('is-visible');container.hidden=true;},
  clear(){container.innerHTML='';api.hide();},
  get visible(){return !container.hidden;}
 };
 return api;
}
root.EdgeSummaryView={renderHTML,renderMatrix,renderEEG,renderEEGFoot,renderEnding,renderMessage,create,cellLabel,TOP_CHANNELS,SEP,format:{pct,pctWhole,num,duration,uv,esc}};
if(typeof module!=='undefined')module.exports=root.EdgeSummaryView;
})(globalThis);
