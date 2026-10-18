import fs from 'node:fs/promises';
export async function page(url,width=1512,height=950){
 const info=await (await fetch('http://127.0.0.1:9228/json/new?about:blank',{method:'PUT'})).json();
 const ws=new WebSocket(info.webSocketDebuggerUrl);await new Promise((ok,no)=>{ws.onopen=ok;ws.onerror=no;});
 let id=0;const pending=new Map(),events=[];
 ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.no(new Error(JSON.stringify(m.error))):p.ok(m.result);}}else events.push(m);};
 const rpc=(method,params={})=>new Promise((ok,no)=>{const n=++id;const timer=setTimeout(()=>{pending.delete(n);no(new Error('Timeout '+method));},15000);pending.set(n,{ok,no,timer});ws.send(JSON.stringify({id:n,method,params}));});
 const evaluate=async expression=>{const v=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(v.exceptionDetails)throw new Error(JSON.stringify(v.exceptionDetails));return v.result.value;};
 await rpc('Page.enable');await rpc('Runtime.enable');await rpc('Log.enable');await rpc('Network.enable');await rpc('Network.setCacheDisabled',{cacheDisabled:true});await rpc('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
 await rpc('Page.navigate',{url});
 for(let i=0;i<100;i++){if(await evaluate('document.readyState === "complete" && !!window.ArmEEGDemo'))break;await new Promise(ok=>setTimeout(ok,100));if(i===99)throw new Error('GUI did not load '+url);}
 return {id:info.id,rpc,evaluate,events,async screenshot(path){const {data}=await rpc('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(path,Buffer.from(data,'base64'));},async close(){await fetch('http://127.0.0.1:9228/json/close/'+info.id);ws.close();}};
}
