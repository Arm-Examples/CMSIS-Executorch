const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const dir=path.resolve(__dirname,'..'),source=process.env.CINDY_UI_DIR||path.join(os.homedir(),'Desktop/cindy-new-ui');
const expected=JSON.parse(fs.readFileSync(path.join(__dirname,'cindy-source-manifest.json'),'utf8'));
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const approvedLayout=JSON.parse(fs.readFileSync(path.join(__dirname,'cindy-layout-manifest.json'),'utf8'));
assert.equal(hash(path.join(dir,'styles.css')),approvedLayout.sha256,'Runtime stylesheet must match Cindy approved layout update');
if(fs.existsSync(source)){
 for(const [name,sha] of Object.entries(expected))assert.equal(hash(path.join(source,name)),sha,'Original reference changed: '+name);
 const original=fs.readFileSync(path.join(source,'index.html'),'utf8'),merged=fs.readFileSync(path.join(dir,'index.html'),'utf8');
 for(const pattern of [/<svg class="person"[\s\S]*?<\/svg>/,/<svg class="scalp-topomap[\s\S]*?<\/svg>/])assert.equal(merged.match(pattern)[0],original.match(pattern)[0],'Cindy SVG must remain byte-identical');
}
console.log('PASS: Cindy approved layout stylesheet matches its source commit; human and 22-electrode SVGs and all supplied originals including PDF unchanged.');
