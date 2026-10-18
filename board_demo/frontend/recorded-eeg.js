/* Display-only prerecorded EEG. UART predictions, timing and task labels are never changed. */
(()=>{
 'use strict';
 let current=null;
 const identity=t=>t&&`S${t.subject}/${t.session}/run${t.run}/cue${t.cue_sample}`;
 function resolve(message){
  const bundle=window.EdgeRecordedEEGData,profile=window.EdgeTrialMetadata;
  if(!bundle||!profile||message?.metadata_profile!==bundle.profile_id||profile.id!==bundle.profile_id||
     profile.model_sha256!==bundle.model_sha256||profile.inputs_sha256!==bundle.inputs_sha256||
     !Number.isInteger(message.trial)||message.trial<0)return null;
  const expected=profile.trials[message.trial],key=identity(expected),row=bundle.trials[key];
  if(!expected||!row||row.trial_id!==key||row.uart_trial!==message.trial||row.ground_truth!==profile.labels[message.trial]||
     !['subject','session','run','cue_sample','start','stop','artifact'].every(k=>row.source_trial[k]===expected[k])||
     bundle.fs_hz!==250||bundle.window_samples!==500||bundle.units!=='µV'||bundle.channel_order.length!==22||
     !bundle.channel_order.every(name=>Array.isArray(row.channels[name])&&row.channels[name].length===500&&row.channels[name].every(Number.isFinite)))return null;
  return row;
 }
 function clear(reason='Prerecorded EEG · awaiting verified trial'){
  current=null;window.ArmEEGDemo?.clearEEG?.(reason);
 }
 window.EdgeRecordedEEG={
  resolve,
  showTrial(message){
   const row=resolve(message);
   if(!row){clear('Prerecorded EEG unavailable · unverified trial');return false;}
   current={trial_id:row.trial_id,uart_trial:row.uart_trial,origin:'prerecorded',units:'µV',fs_hz:250,samples:500};
   window.ArmEEGDemo.onEEGFrame(row.channels,{origin:'prerecorded',trial_id:row.trial_id,uart_trial:row.uart_trial,units:'µV'});
   return true;
  },
  clear,
  getState:()=>current&&({...current})
 };
})();
