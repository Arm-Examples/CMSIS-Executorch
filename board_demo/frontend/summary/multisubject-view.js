/* Presentation adapter only: observe the verified multi-subject controller and use
 * its existing End Run / Reset / Play actions. No transport or inference hooks. */
(()=>{'use strict';
 if(!window.EdgeMultisubjectMode)return;
 const $=id=>document.getElementById(id),end=$('endRunBtn'),fresh=$('newRunBtn');
 const view=EdgeSummaryView.create($('sessionSummary'),{onNewRun:()=>fresh.click(),onResume:()=>$('playBtn').click()});
 end.hidden=false;fresh.hidden=false;
 let owner=null,rows=null,startedAt=null,final=null,ending=false,lastKey='',acc=null,processed=0;
 const classes={0:{name:'left_hand',display:'Left hand',short:'Left',order:0},1:{name:'right_hand',display:'Right hand',short:'Right',order:1}};
 function clear(c){owner=c;rows=c?.completedRecords;startedAt=null;final=null;ending=false;lastKey='';processed=0;acc=EdgeEEGOverview.createAccumulator({units:'µV',unitsVerified:true});view.clear();}
 function snapshot(c,status){
  const records=c.completedRecords;
  for(;processed<records.length;processed++){
   const r=records[processed];if(acc.has(r.trial_id))continue;
   let samples=null;try{samples=c.dataset.channels(r.trial);}catch{}
   acc.add(r.trial_id,samples,{artifact:r.artifact});
  }
  const roundTrips=new Map(records.map(r=>[r.session+':'+r.request,r.timing.host_round_trip_ms]));
  return EdgeSessionSummary.build({modelId:c.state().model_id,modelHash:c.firmware?.model_sha256,modelHashVerified:!!c.ready,sessionId:c.session,seed:c.seed,source:'live-hardware',status,startedAt,endedAt:Date.now(),classLabels:classes,classNamesVerified:true,
   interruption:status==='interrupted'?{message:c.failed?.message||'Hardware request unresolved',unresolvedRequest:c.pending?.request}:null},
   EdgeSessionSummary.adaptStreamingRecords(records,{dataset:c.dataset,roundTrips}),{eeg:acc.result()});
 }
 end.addEventListener('click',async()=>{
  const c=owner;if(!c||ending||final)return;ending=true;end.disabled=true;fresh.disabled=true;view.ending({request:c.pending?.request});
  try{await window.EdgeMultisubjectSession.endRun();if(owner===c){final=snapshot(c,c.pending||c.failed?'interrupted':c.completedRecords.length?'completed':'empty');view.render(final);}}
  catch(e){view.message('Run interrupted',e.message);}
  finally{ending=false;}
 });
 fresh.addEventListener('click',()=>{if(owner&&!ending)$('resetBtn').click();});
 function observe(){
  const c=window.EdgeStreamingController;
  if(c!==owner||c&&c.completedRecords!==rows)clear(c);
  end.disabled=!c?.ready||!!c?.failed||ending||!!final||!!c?.ended;
  fresh.disabled=!c||ending||!!c.pending;
  if(c){
   const s=c.state();if(!s.paused&&startedAt===null)startedAt=Date.now();
   if(!ending&&!final){
    const key=[s.paused,s.error,s.ended,c.completedRecords.length,startedAt!==null].join(':');
    if(key!==lastKey){lastKey=key;
     if(s.ended){final=snapshot(c,s.error?'interrupted':'completed');view.render(final);}
     else if((startedAt!==null||c.completedRecords.length)&&(s.paused||s.error))view.render(snapshot(c,s.error?'interrupted':'paused'),{final:false,resumable:!!c.ready&&!c.failed});
     else view.hide();
    }
    if(s.paused&&startedAt!==null&&!s.error&&$('playBtn').textContent==='▶ Play')$('playBtn').textContent='▶ Resume';
   }
  }
  requestAnimationFrame(observe);
 }
 clear(null);requestAnimationFrame(observe);
})();
