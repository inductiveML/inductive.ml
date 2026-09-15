import {readFile,writeFile,mkdir,stat,copyFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
const root=resolve(import.meta.dir,'..'),assets=resolve(root,'public/monarch');
const manifest=JSON.parse(await readFile(resolve(assets,'model-manifest.json'),'utf8'));
const sources=JSON.parse(await readFile(resolve(root,'src/monarch/source-manifest.json'),'utf8'));
for(const file of sources.files){
 const bytes=await readFile(resolve(root,'src/monarch/vendor/monarch',file.path));
 if(createHash('sha256').update(bytes).digest('hex')!==file.sha256)throw new Error(`Vendored research source changed: ${file.path}`);
}
// Large transport chunks stay out of Git. Restore them from the pinned local
// research blob when building a fresh checkout, and reject partial deployments.
let weights:Buffer|undefined;
for(const chunk of manifest.chunks){
 const path=resolve(root,'public'+chunk.url);let valid=false;
 try{valid=(await stat(path)).size===chunk.bytes&&createHash('sha256').update(await readFile(path)).digest('hex')===chunk.sha256;}catch{}
 if(valid)continue;
 if(!weights){const source=process.env.MONARCH_SOURCE_DIR??resolve(root,'../monarch');weights=await readFile(resolve(source,'experiments/flash_core_split_n_b1_4/data/weights/weights.bin'));if(createHash('sha256').update(weights).digest('hex')!==manifest.weightSha256)throw new Error('Canonical model digest mismatch');}
 const index=manifest.chunks.indexOf(chunk),offset=manifest.chunks.slice(0,index).reduce((n:number,c:any)=>n+c.bytes,0);
 await mkdir(resolve(assets,'weights'),{recursive:true});await writeFile(path,weights.subarray(offset,offset+chunk.bytes));
}
const build=await Bun.build({entrypoints:[resolve(root,'src/monarch/worker.ts')],target:'browser',format:'esm',minify:true});
if(!build.success)throw new Error(build.logs.map(x=>x.message).join('\n'));
await writeFile(resolve(assets,'worker.js'),await build.outputs[0].text());
await copyFile(resolve(root,'src/monarch/source-manifest.json'),resolve(assets,'source-manifest.json'));
console.log(`MONARCH: ${manifest.chunks.length} verified model chunks; worker ${build.outputs[0].size} bytes.`);
