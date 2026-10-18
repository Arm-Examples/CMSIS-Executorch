/* Keep the mobile hardware controls outside the design's filtered shell.
 * A backdrop-filter makes that shell the containing block for fixed descendants.
 * Moving this one integration widget preserves Cindy's responsive panel layout.
 */
(()=>{
 'use strict';
 const dock=document.getElementById('hardwareControls'),home=dock.parentElement;
 const mobile=window.matchMedia('(max-width:720px)');
 const place=()=>{(mobile.matches?document.body:home).appendChild(dock);};
 mobile.addEventListener('change',place);place();
})();
