/* Select presentation bridge without modifying the validated Model 2 bridge. */
(() => {
 const script=document.createElement('script');
 script.src=window.EdgeMultisubjectMode?'multisubject/bridge.js':'streaming/bridge.js';
 script.onerror=()=>{const status=document.getElementById('transportStatus');if(status)status.textContent='Frontend bridge failed to load; reload before connecting UART';};
 document.body.appendChild(script);
})();
