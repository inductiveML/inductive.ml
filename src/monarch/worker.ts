import './vendor/monarch/experiments/kernel_efficiency_b1_8/src/page/hooks.ts';
import {configureCandidates} from './vendor/monarch/experiments/kernel_efficiency_b1_8/src/page/plans.ts';
import {initFused, prime, decodePipelined, deviceNeedsFor} from './vendor/monarch/experiments/kernel_efficiency_b1_8/src/page/fused.ts';
import {deviceForRuntime, readBuffer} from './vendor/monarch/experiments/flash_core_split_n_b1_4/src/page/device.ts';
import {Tokenizer} from './vendor/tokenizers.mjs';
import model from './model-config.json';
import fixture from './benchmark-fixture.json';

const send=(type:string,data:Record<string,unknown>={})=>postMessage({type,...data});
const pause=()=>new Promise(resolve=>setTimeout(resolve,0));
const digest=async(bytes:ArrayBuffer)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
let session:any=null,tokenizer:any=null,busy=false,stopped=false,controller:AbortController|null=null,hasTimestamps=false;
let deviceDescription='WebGPU device';
function safety(){
 const hooks=(globalThis as any).__probeHooks;
 if(hooks.errors().length||hooks.lost().length)throw new Error('The GPU reported an error. Unload the model and try again.');
}
async function support(){
 if(!navigator.gpu)return {supported:false,reason:'WebGPU is unavailable. Open this page in a recent Chrome or Edge browser on a supported computer.'};
 const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
 if(!adapter)return {supported:false,reason:'No WebGPU adapter is available in this browser.'};
 const missing=['shader-f16','subgroups'].filter(x=>!adapter.features.has(x as GPUFeatureName));
 if(missing.length)return {supported:false,reason:`This decoder needs ${missing.join(' and ')}. Your browser does not expose ${missing.length===1?'this feature':'these features'}; the model has not been downloaded.`};
 const info=adapter.info;
 if(info.subgroupMinSize!==32||info.subgroupMaxSize!==32)return {supported:false,reason:`This kernel is compiled for 32-lane subgroups. This adapter reports ${info.subgroupMinSize ?? '?'}–${info.subgroupMaxSize ?? '?'}. No compatible kernel is available here.`};
 if(adapter.limits.maxComputeInvocationsPerWorkgroup<1024||adapter.limits.maxComputeWorkgroupSizeX<1024)return {supported:false,reason:'This GPU does not support the 1,024-thread workgroups used by MIX_M40. The model has not been downloaded.'};
 if(adapter.limits.maxStorageBufferBindingSize<model.runtime.weightBytes)return {supported:false,reason:'This adapter cannot bind the 169 MB model buffer required by this decoder.'};
 if(info.isFallbackAdapter)return {supported:false,reason:'A hardware WebGPU adapter is required; a software adapter was detected.'};
 hasTimestamps=adapter.features.has('timestamp-query');
 deviceDescription=[info.description||info.vendor,info.architecture].filter(Boolean).join(' · ')||'WebGPU hardware adapter';
 return {supported:true,device:deviceDescription,timestamps:hasTimestamps,subgroup:32};
}
async function downloadWeights(){
 const response=await fetch('/monarch/model-manifest.json',{signal:controller!.signal});if(!response.ok)throw new Error('The model manifest could not be loaded.');
 const manifest=await response.json();
 if(manifest.weightSha256!=='ca1fea89fd9f3ca7e5d6d5720705c96f457a92edd0cfc811cf8d9391ac784cb0'||manifest.weightBytes!==model.runtime.weightBytes)throw new Error('The published model manifest does not match the tested decoder.');
 const combined=new Uint8Array(manifest.weightBytes);let offset=0,cache:Cache|null=null;
 try{cache=await caches.open('monarch-weights-ca1fea89-v1');}catch{/* Storage access is optional. */}
 for(const chunk of manifest.chunks){
  if(stopped)throw new DOMException('Stopped','AbortError');
  let bytes:ArrayBuffer|null=null,cached=false;
  const hit=await cache?.match(chunk.url).catch(()=>undefined);
  if(hit){const candidate=await hit.arrayBuffer();if(candidate.byteLength===chunk.bytes&&await digest(candidate)===chunk.sha256){bytes=candidate;cached=true;}else await cache?.delete(chunk.url);}
  if(!bytes){
   const r=await fetch(chunk.url,{signal:controller!.signal});if(!r.ok||!r.body)throw new Error(`Model download failed (${r.status}). Try loading again.`);
   const data=new Uint8Array(chunk.bytes),reader=r.body.getReader();let n=0,last=0;
   while(true){const {value,done}=await reader.read();if(done)break;if(n+value.byteLength>data.byteLength)throw new Error('Unexpected model chunk size.');data.set(value,n);n+=value.byteLength;
    if(performance.now()-last>80){send('progress',{stage:'Downloading weights',loaded:offset+n,total:manifest.weightBytes});last=performance.now();}}
   bytes=data.buffer;if(n!==chunk.bytes||await digest(bytes)!==chunk.sha256)throw new Error('Model integrity check failed. Please retry the download.');
   try{await cache?.put(chunk.url,new Response(bytes,{headers:{'Content-Type':'application/octet-stream'}}));}catch{/* A full cache must not prevent inference. */}
  }
  combined.set(new Uint8Array(bytes),offset);offset+=bytes.byteLength;
  send('progress',{stage:cached?'Reading verified cached weights':'Downloading weights',loaded:offset,total:manifest.weightBytes});
 }
 if(offset!==manifest.weightBytes||await digest(combined.buffer)!==manifest.weightSha256)throw new Error('The complete model checksum did not match.');
 return combined;
}
async function load(){
 const info=await support();send('support',info);if(!info.supported)return;
 controller=new AbortController();stopped=false;
 const config={...model.runtime,extraFeatures:model.runtime.extraFeatures.filter(f=>f!=='timestamp-query'||hasTimestamps)};
 send('progress',{stage:'Preparing your GPU',loaded:0,total:model.runtime.weightBytes});
 await deviceForRuntime(deviceNeedsFor(config as any),'high-performance',32);
 const bytes=await downloadWeights();
 send('progress',{stage:'Loading tokenizer',loaded:bytes.length,total:bytes.length});
 const [tj,tc]=await Promise.all(['tokenizer.json','tokenizer_config.json'].map(async n=>{const r=await fetch('/monarch/'+n,{signal:controller!.signal});if(!r.ok)throw new Error('Tokenizer download failed.');return r.json();}));
 tokenizer=new Tokenizer(tj,tc);
 configureCandidates([model.candidate] as any);
 send('progress',{stage:'Compiling the 72-dispatch decoder',loaded:bytes.length,total:bytes.length});
 const url=URL.createObjectURL(new Blob([bytes]));
 try{const initialized=await initFused({...config,weightsUrl:url} as any);session=initialized.session;}finally{URL.revokeObjectURL(url);}
 session.runtime.setArm('MIX_M40');safety();
 // A real fixed continuation checks the shipped bundle before enabling the UI.
 send('progress',{stage:'Checking a known 32-token continuation',loaded:bytes.length,total:bytes.length});
 await prime(session,{tokenIds:Uint32Array.from(fixture.prompt),promptLength:fixture.prompt.length,seed:111008,freerun:true});
 const seed=new Uint32Array(await readBuffer(session.gpu.device,session.runtime.buffers.sampled,(fixture.prompt.length-1)*4,4))[0];
 const test=await decode(32,fixture.prompt.length);
 if(seed!==fixture.continuation[0]||test.sampled.some((id:number,i:number)=>id!==fixture.continuation[i+1]))throw new Error('The decoder failed its known-answer check on this GPU. This device is not supported by the current kernel.');
 safety();send('loaded',{device:deviceDescription,timestamps:hasTimestamps,verifiedTokens:32,dispatches:session.runtime.dispatchCount,weightBytes:model.runtime.weightBytes});
}
function decode(tokens:number,position:number){return decodePipelined(session,{tokens,startPosition:position,startStep:position,readbackEvery:32,ringTokens:64,timestamps:hasTimestamps,freerun:true,seed:111008});}
function textOf(ids:number[]){return ids.length?tokenizer.decode(ids,{skip_special_tokens:true}):'';}
async function generate(prompt:string,limit:number){
 if(!prompt.trim())throw new Error('Enter a prompt first.');
 const formatted=`<|startoftext|><|im_start|>system\nYou are a helpful assistant trained by Liquid AI.<|im_end|>\n<|im_start|>user\n${prompt.trim()}<|im_end|>\n<|im_start|>assistant\n`;
 const ids=tokenizer.encode(formatted,{add_special_tokens:false}).ids;
 if(ids.length>1024)throw new Error(`Your prompt is ${ids.length.toLocaleString()} tokens. Shorten it to fit the demo’s 1,024-token prompt limit.`);
 const eos=tokenizer.token_to_id('<|im_end|>'),generated:number[]=[];const start=performance.now();
 send('generating',{promptTokens:ids.length});
 await prime(session,{tokenIds:Uint32Array.from(ids),promptLength:ids.length,seed:111008,freerun:true});
 const first=new Uint32Array(await readBuffer(session.gpu.device,session.runtime.buffers.sampled,(ids.length-1)*4,4))[0];
 const firstAt=performance.now(),ttft=firstAt-start;let computed=0,gpuMs=0;
 if(first!==eos)generated.push(first);
 send('text',{text:textOf(generated),tokens:generated.length,tps:null,ttft});
 let ended=first===eos,position=ids.length;
 while(!stopped&&!ended&&generated.length<limit){
  const r=await decode(32,position);position+=32;computed+=32;gpuMs+=Math.max(0,r.gpuBusyMs);
  for(const id of r.sampled){if(id===eos){ended=true;break;}if(generated.length>=limit)break;generated.push(id);}
  safety();const elapsed=performance.now()-firstAt;
  send('text',{text:textOf(generated),tokens:generated.length,tps:generated.length>1?(generated.length-1)*1000/elapsed:null,ttft,gpuTps:hasTimestamps?computed*1000/gpuMs:null});
  await pause();
 }
 safety();send('generated',{text:textOf(generated),tokens:generated.length,tps:generated.length>1?(generated.length-1)*1000/(performance.now()-firstAt):null,ttft,stopped,promptTokens:ids.length,computedTokens:computed+1});
}
async function benchmark(depth:number){
 if(![192,1024].includes(depth))throw new Error('Unsupported benchmark context.');
 const all=[...fixture.prompt,...fixture.continuation],start=depth-128,runs:any[]=[];
 const prepare=()=>prime(session,{tokenIds:Uint32Array.from(all.slice(0,start)),promptLength:start,seed:111008,freerun:true});
 send('benchmarkProgress',{completed:0,total:4,stage:'Warming the decoder'});
 await prepare();await decode(64,start);
 for(let i=0;i<4&&!stopped;i++){
  await prepare();send('benchmarkProgress',{completed:i,total:4,stage:`Measuring run ${i+1} of 4`});
  const r=await decode(256,start);safety();
  if(runs.length&&r.sampled.some((id:number,j:number)=>id!==runs[0].ids[j]))throw new Error('Benchmark continuations differed between repeats. No valid result is reported.');
  runs.push({wallMs:r.wallMs,tps:256000/r.wallMs,gpuBusyMs:hasTimestamps?r.gpuBusyMs:null,ids:r.sampled});
  send('benchmarkProgress',{completed:i+1,total:4,stage:`Run ${i+1}: ${(256000/r.wallMs).toFixed(0)} tok/s`,tps:256000/r.wallMs});await pause();
 }
 if(stopped){send('cancelled');return;}
 const times=runs.map(r=>r.wallMs/256).sort((a,b)=>a-b),ms=(times[1]+times[2])/2;
 send('benchmarkResult',{depth,tokens:256,runs:4,ms,tps:1000/ms,runTps:runs.map(r=>r.tps),spreadPct:100*(times[3]-times[0])/ms,device:deviceDescription,measuredAt:new Date().toISOString(),timestamps:hasTimestamps,exactRepeats:true,protocol:'4 × 256 free-running tokens; 64-token warmup; prefill excluded; 32-token readback through a 64-token ring; wall timer includes encoding and queue completion. Context is the center of the 256-token window. This quick browser run is not the original controlled lab confirmation.'});
}
onmessage=async({data})=>{
 if(data.type==='cancel'){stopped=true;controller?.abort();return;}
 if(data.type==='support'){try{send('support',await support());}catch(e){send('support',{supported:false,reason:String(e)});}return;}
 if(busy)return;busy=true;stopped=false;
 try{if(data.type==='load')await load();else if(data.type==='generate'&&session)await generate(String(data.prompt??''),Math.min(512,Math.max(32,Number(data.limit)||256)));else if(data.type==='benchmark'&&session)await benchmark(Number(data.depth));}
 catch(e){send('error',{message:e instanceof Error?e.message:String(e),fatal:!session||Boolean((globalThis as any).__probeHooks.lost().length)});}
 finally{busy=false;}
};
