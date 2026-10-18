/* Incremental UART framing, shared by the browser and offline tests. */
(function(root){
'use strict';
function createParser(emit, error=()=>{}) {
 let buffer='', overflow=false;
 function line(){
  const s=buffer.trim(); buffer='';
  if(overflow){overflow=false;return;}
  if(!s)return;
  if(s.startsWith('EEG_RESULT')){
   const p=s.split(/\s+/), nums=p.slice(1).map(Number);
   if(p.length!==5||!nums.every(Number.isFinite)||!Number.isSafeInteger(nums[0])||nums[0]<0||![0,1].includes(nums[3])){error('Malformed EEG_RESULT: '+s);return;}
   emit({type:'prediction',trial:nums[0],logits:nums.slice(1,3),class_id:nums[3],raw_line:s});
  }else if(s.startsWith('Execution_result:')) emit({type:'execution',pass:/^Execution_result:\s*PASS(?:\s+\([^\r\n]*\))?$/.test(s),raw_line:s});
  else if(s.startsWith('EEG_ERROR')) emit({type:'execution',pass:false,raw_line:s});
 }
 return {push(text){for(const c of text){if(c==='\n'||c==='\r'||c==='\x04'){line();if(c==='\x04')emit({type:'eot'});}else if(!overflow){buffer+=c;if(buffer.length>4096){buffer='';overflow=true;error('UART line exceeds 4096 characters');}}}},finish:line};
}
root.EdgeUART={createParser};
if(typeof module!=='undefined')module.exports=root.EdgeUART;
})(globalThis);
