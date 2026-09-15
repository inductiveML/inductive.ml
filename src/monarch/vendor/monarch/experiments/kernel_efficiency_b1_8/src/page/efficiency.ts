import type {Candidate} from './plans.ts';
import {DEQUANT_F16,type KernelSource} from '../../../flash_core_split_n_b1_4/src/page/wgsl.ts';
export function magicDot(kernel:KernelSource,enabled:boolean):KernelSource{
 if(!enabled)return kernel;let code=kernel.code;
 for(const name of ['dot8','dot8h']){
  if(!code.includes('fn '+name+'('))continue;
  const half=name==='dot8h',type=half?'f16':'f32';
  const pairs=Array.from({length:4},(_,i)=>{
   const bits=`(((packed >> ${i*4}u)&0x000f000fu)|0x64006400u)`;
   const val=half?`bitcast<vec2<f16>>(${bits})`:`unpack2x16float(${bits})`;
   return `let p${i}=${val}-vec2<${type}>(${type}(1024.0)+zp);`;
  }).join('\n');
  code=replaceFunction(code,name,`fn ${name}(packed:u32,zp:${type},a0:vec4<${type}>,a1:vec4<${type}>)->${type}{${pairs}return dot(vec4<${type}>(p0.x,p1.x,p2.x,p3.x),a0)+dot(vec4<${type}>(p0.y,p1.y,p2.y,p3.y),a1);}`);
 }
 return {...kernel,code};
}
export function maskedDot(kernel:KernelSource,enabled:boolean):KernelSource{
 if(!enabled||!kernel.code.includes('fn dot8('))return kernel;
 const code=replaceFunction(kernel.code,'dot8',`fn dot8(packed:u32,zp:f32,a0:vec4<f32>,a1:vec4<f32>)->f32{
  let mask=vec4<u32>(15u,240u,3840u,61440u);
  let factor=vec4<f32>(1.0,16.0,256.0,4096.0);let inv=vec4<f32>(1.0,0.0625,0.00390625,0.000244140625);
  let lo=vec4<f32>(vec4<u32>(packed)&mask)-vec4<f32>(zp)*factor;
  let hi=vec4<f32>(vec4<u32>(packed>>16u)&mask)-vec4<f32>(zp)*factor;
  return dot(lo,a0*inv)+dot(hi,a1*inv);
 }`);
 return {...kernel,code};
}
function replaceFunction(code:string,name:string,body:string):string {
 const begin=code.indexOf('fn '+name+'(');if(begin<0)throw new Error('Missing function '+name);
 let at=code.indexOf('{',begin)+1,depth=1;
 while(depth&&at<code.length){if(code[at]==='{')depth++;if(code[at]==='}')depth--;at++;}
 if(depth)throw new Error('Unclosed function '+name);
 return code.slice(0,begin)+body+code.slice(at);
}
function privateReads(code:string):string {
 let count=0;code=code.replace(/\bstage\[o(?:\s*\+\s*(\d+)u)?\]/g,(_m,n:string)=>{count++;return `laneBlock[${n??'0'}u]`;});
 if(count!==32)throw new Error('Private input source seam '+count);
 return code+'\nvar<private> laneBlock:array<f32,32>;\n';
}
export function privateMatvec(kernel:KernelSource,mode:string,preScale:number,enabled:boolean):KernelSource {
 if(!enabled)return kernel;
 if(!kernel.code.includes('const COLS: u32 = 1024u;'))throw new Error('Private matvec requires 1024 columns');
 const norm=mode==='matvec_residual'?'':`var square=0.0;for(var i=lane;i<COLS;i+=SG){let v=residual[query*COLS+i]*${preScale.toFixed(1)};square+=v*v;}let inv=inverseSqrt(subgroupAdd(square)/f32(COLS)+EPSILON);`;
 const value=mode==='matvec_residual'?'f32(source[query*COLS+at])':`(residual[query*COLS+at]*${preScale.toFixed(1)})*inv*gamma[at]`;
 const code=replaceFunction(kernel.code,'stageInput',`fn stageInput(query:u32,local:u32){let lane=local%SG;${norm}for(var i=0u;i<32u;i++){let at=lane*32u+i;laneBlock[i]=${value};}}`);
 return {...kernel,code:privateReads(code)};
}
export function preNormalizedMatvec(kernel:KernelSource,enabled:boolean):KernelSource{
 if(!enabled)return kernel;
 return {...kernel,code:replaceFunction(kernel.code,'stageInput',`fn stageInput(query:u32,local:u32){for(var i=local;i<COLS;i+=WG){stage[i]=residual[query*COLS+i];}workgroupBarrier();}`)};
}
export function convProjectionKernel(kernel:KernelSource,enabled:boolean):KernelSource{
 if(!enabled)return kernel;
 let code=kernel.code;
 const main=`@compute @workgroup_size(WG)
 fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(subgroup_invocation_id) lane:u32){
 let query=wid.y;if(query>=state.queries){return;}stageInput(query,local);
 let rowBase=wid.x*(ROWS_PER_WG/3u)+(local/SG)*(ROWS_PER_SG/3u);
 for(var r=0u;r<ROWS_PER_SG/3u;r++){
  let want=rowBase+r;let channel=min(want,1023u);
  let b=f32(f16(rowSum(channel,lane)));let c=f32(f16(rowSum(channel+1024u,lane)));let x=f32(f16(rowSum(channel+2048u,lane)));
  if(lane==0u&&want<1024u){
   let s1=f32(cache[channel*3u+1u]);let s2=f32(cache[channel*3u+2u]);let bx=b*x;
   let w0=bitcast<f32>(blob[channel*3u]);let w1=bitcast<f32>(blob[channel*3u+1u]);let w2=bitcast<f32>(blob[channel*3u+2u]);
   let y=w0*s1+w1*s2+w2*bx;destination[query*1024u+channel]=f16(c*y);
   cache[channel*3u]=f16(0.0);cache[channel*3u+1u]=f16(s2);cache[channel*3u+2u]=f16(bx);
  }
 }
 }`;
 const at=code.indexOf('@compute');if(at<0)throw new Error('Conv projection source seam');
 code=code.slice(0,at)+main+'\n@group(0) @binding(18) var<storage,read> blob:array<u32>;\n@group(0) @binding(9) var<storage,read_write> cache:array<f16>;\n';
 return {...kernel,code,bindings:[...kernel.bindings,{binding:18,name:'blob',access:'read'},{binding:9,name:'cache',access:'read_write'}]};
}
export function inputNormKernel(workgroupSize:number,preScale:number):KernelSource{
 return {workgroupSize,bindings:[{binding:0,name:'residual',access:'read'},{binding:1,name:'gamma',access:'read'},{binding:2,name:'projection',access:'read_write'}],code:`enable subgroups;
 const WG:u32=${workgroupSize}u;
 @group(0) @binding(0) var<storage,read> residual:array<f32>;
 @group(0) @binding(1) var<storage,read> gamma:array<f32>;
 @group(0) @binding(2) var<storage,read_write> projection:array<f32>;
 var<workgroup> squares:array<f32,${workgroupSize/32}>;
 @compute @workgroup_size(WG) fn main(@builtin(local_invocation_index) local:u32){
  var sum=0.0;for(var i=local;i<1024u;i+=WG){let v=residual[i]*${preScale.toFixed(1)};sum+=v*v;}
  let sg=subgroupAdd(sum);if(local%32u==0u){squares[local/32u]=sg;}workgroupBarrier();
  var total=0.0;for(var g=0u;g<WG/32u;g++){total+=squares[g];}
  let inv=inverseSqrt(total/1024.0+0.00001);
  for(var i=local;i<1024u;i+=WG){projection[i]=(residual[i]*${preScale.toFixed(1)})*inv*gamma[i];}
 }`};
}
export function swizzleStage(kernel:KernelSource,enabled:boolean):KernelSource {
 if(!enabled)return kernel;
 let count=0;
 const code=kernel.code.replace(/\bstage\[([^\]]+)\]/g,(_m,index:string)=>{count++;return `stage[stageIndex(${index})]`;});
 if(count<8)throw new Error('Stage swizzle source seam');
 return {...kernel,code:code+'\nfn stageIndex(i:u32)->u32{return i ^ ((i >> 5u) & 31u);}\n'};
}
export function efficientMlp(kernel:KernelSource,c:Candidate):KernelSource{
 let code=kernel.code;
 if(c.mlpVecLoad){
  if(c.mlpPair||c.projectPair)throw new Error('Vector-load arm uses unpaired baseline arithmetic');
  let count=0;
  code=code.replace(/var part: f32 = 0\.0;/g,m=>'let packed = weights[base / 4u];\n      '+m);
  code=code.replace(/blob\[base(?: \+ ([123])u)?\]/g,(_m,w:string)=>{count++;return 'packed.'+['x','y','z','w'][Number(w??0)];});
  if(count!==8)throw new Error('MLP vector load seam '+count);
  code+='\n@group(0) @binding(19) var<storage,read> weights:array<vec4<u32>>;\n';
  kernel={...kernel,bindings:[...kernel.bindings,{binding:19,name:'weights',access:'read'}]};
 }
 if(c.normOnceMlp){
  code=replaceFunction(code,'stageBlock',`fn stageBlock(local:u32,group:u32){
   let rowBegin=group*HIDDEN/SLICES;let rowEnd=(group+1u)*HIDDEN/SLICES;
   for(var i=rowBegin+local;i<rowEnd;i+=WG){residual_out[i]=residual_in[i];}
   for(var i=local;i<HIDDEN;i+=WG){stage[i]=projection[i];}workgroupBarrier();
  }`);
  code+='\n@group(0) @binding(8) var<storage,read> projection:array<f32>;\n';
  kernel={...kernel,bindings:[...kernel.bindings,{binding:8,name:'projection',access:'read'}]};
 }
 if(c.mlpPrivate){
  if(c.mlpPair||!code.includes('const FOLD_SLICES: u32 = 0u;'))throw new Error('Private MLP requires a plain no-fold input');
  code=replaceFunction(code,'stageBlock',`fn stageBlock(local:u32,group:u32){
   let lane=local%SG;var square=0.0;
   for(var i=lane;i<HIDDEN;i+=SG){let v=residual_in[i];square+=v*v;}
   let inv=inverseSqrt(subgroupAdd(square)/f32(HIDDEN)+EPSILON);
   let rowBegin=group*HIDDEN/SLICES;let rowEnd=(group+1u)*HIDDEN/SLICES;
   for(var i=rowBegin+local;i<rowEnd;i+=WG){residual_out[i]=residual_in[i];}
   for(var i=0u;i<32u;i++){let at=lane*32u+i;laneBlock[i]=residual_in[at]*inv*bitcast<f32>(blob[NORM+at]);}
  }`);
  code=privateReads(code);
 }
 if(c.mlpPair){
  const begin=code.indexOf('fn rowSum('),end=code.indexOf('const KQUANT:',begin);
  if(begin<0||end<0)throw new Error('PAIR source seam');
  const dots=Array.from({length:4},(_,w)=>{
   const at=w*8;
   const a=Array.from({length:4},(_,j)=>`stage[o+${at+j}u]`).join(','),b=Array.from({length:4},(_,j)=>`stage[o+${at+j+4}u]`).join(',');
   return `let x${w}=vec4<f32>(${a}); let y${w}=vec4<f32>(${b});\npart.x += dot8(blob[base+${w}u],8.0,x${w},y${w});\npart.y += dot8(blob[upbase+${w}u],8.0,x${w},y${w});`;
  }).join('\n');
  code=code.slice(0,begin)+`fn rowPair(row:u32,lane:u32)->vec2<f32>{
 var acc=vec2<f32>(0.0);
 for(var i=0u;i<BLOCK_ITERS;i++){
 let block=lane+i*SG;
 if(block<BLOCKS){let base=QUANT+row*BLOCKS*4u+block*4u;let upbase=base+FFN*BLOCKS*4u;let o=block*32u;var part=vec2<f32>(0.0);
 ${dots}
 acc += part*vec2<f32>(bitcast<f32>(blob[SCALES+row*BLOCKS+block]),bitcast<f32>(blob[SCALES+(FFN+row)*BLOCKS+block]));}
 }
 return subgroupAdd(acc);
}\n`+code.slice(end);
  code=code.replace('let gate = rowSum(rowBase + r, lane);\n    let up = rowSum(FFN + rowBase + r, lane);','let pair = rowPair(rowBase + r, lane);\n    let gate = pair.x; let up = pair.y;');
 }
 if(c.projectPair){
  const begin=code.indexOf('fn project('),end=code.indexOf('@compute',begin);if(begin<0||end<0)throw new Error('PROJECT source seam');
  const dots=Array.from({length:4},(_,w)=>{
   const a=Array.from({length:4},(_,j)=>`hidden_slice[o+${w*8+j}u]`).join(','),b=Array.from({length:4},(_,j)=>`hidden_slice[o+${w*8+j+4}u]`).join(',');
   return `let x${w}=vec4<f32>(${a});let y${w}=vec4<f32>(${b});part.x+=dot8(blob[base+${w}u],8.0,x${w},y${w});part.y+=dot8(blob[base2+${w}u],8.0,x${w},y${w});`;
  }).join('\n');
  code=code.slice(0,begin)+`fn project(local:u32,group:u32){
 let sliceBase=group*HIDDEN;
 for(var row=local;row<HIDDEN;row+=2u*WG){var acc=vec2<f32>(0.0);
 for(var b=0u;b<BLOCKS_PER_SLICE;b++){let base=KQUANT+(sliceBase+row)*WORDS_PER_ROW_SLICE+b*4u;let base2=base+WG*WORDS_PER_ROW_SLICE;let o=b*32u;var part=vec2<f32>(0.0);${dots}
 acc+=part*vec2<f32>(bitcast<f32>(blob[KSCALES+(sliceBase+row)*BLOCKS_PER_SLICE+b]),bitcast<f32>(blob[KSCALES+(sliceBase+row+WG)*BLOCKS_PER_SLICE+b]));}
 partials_out[sliceBase+row]=acc.x;partials_out[sliceBase+row+WG]=acc.y;}
}\n`+code.slice(end);
 }
 if(c.mlpDotIlp){
  if(!c.mlpPair||c.projectPair||c.mlpVecLoad)throw new Error('MLP ILP requires paired gate and plain down');
  if(c.mlpDotIlp==='gate'||c.mlpDotIlp==='both'){
   let count=0;
   code=code.replace('var part=vec2<f32>(0.0);','var gatePart=vec4<f32>(0.0);var upPart=vec4<f32>(0.0);');
   code=code.replace(/part\.([xy]) \+= dot8\(blob\[(base|upbase)\+(\d)u\],8\.0,x(\d),y(\d)\);/g,(_m,xy:string,_base:string,w:string,x:string,y:string)=>{
    if(w!==x||w!==y)throw new Error('Gate ILP lane mismatch');count++;return `${xy==='x'?'gatePart':'upPart'}.${['x','y','z','w'][Number(w)]}=dot8(blob[${xy==='x'?'base':'upbase'}+${w}u],8.0,x${w},y${w});`;
   });
   if(count!==8)throw new Error('Gate ILP source seam '+count);
   code=code.replace('acc += part*vec2<f32>','let part=vec2<f32>((gatePart.x+gatePart.y)+(gatePart.z+gatePart.w),(upPart.x+upPart.y)+(upPart.z+upPart.w));acc += part*vec2<f32>');
  }
  if(c.mlpDotIlp==='down'||c.mlpDotIlp==='both'){
   const expr=Array.from({length:4},(_,w)=>{
    const a=Array.from({length:4},(_,j)=>`hidden_slice[o+${w*8+j}u]`).join(',');
    const b=Array.from({length:4},(_,j)=>`hidden_slice[o+${w*8+j+4}u]`).join(',');
    return `dot8(blob[base+${w}u],8.0,vec4<f32>(${a}),vec4<f32>(${b}))`;
   }).join(',');
   code=replaceFunction(code,'project',`fn project(local:u32,group:u32){
    let sliceBase=group*HIDDEN;
    for(var row=local;row<HIDDEN;row+=WG){var acc=0.0;
     for(var b=0u;b<BLOCKS_PER_SLICE;b++){let base=KQUANT+(sliceBase+row)*WORDS_PER_ROW_SLICE+b*4u;let o=b*32u;
      let parts=vec4<f32>(${expr});let part=(parts.x+parts.y)+(parts.z+parts.w);
      acc+=part*bitcast<f32>(blob[KSCALES+(sliceBase+row)*BLOCKS_PER_SLICE+b]);
     }partials_out[sliceBase+row]=acc;
    }
   }`);
  }
 }
 if(c.mlpHalfProduct){
  code+=`
fn dot8halfProduct(packed:u32,zp:f32,a0:vec4<f32>,a1:vec4<f32>)->f32{
   let lo=vec4<f16>(f16(packed&15u),f16((packed>>4u)&15u),f16((packed>>8u)&15u),f16((packed>>12u)&15u))-vec4<f16>(f16(zp));
   let hi=vec4<f16>(f16((packed>>16u)&15u),f16((packed>>20u)&15u),f16((packed>>24u)&15u),f16((packed>>28u)&15u))-vec4<f16>(f16(zp));
   let p0=vec4<f32>(lo*vec4<f16>(a0));let p1=vec4<f32>(hi*vec4<f16>(a1));
   return dot(p0,vec4<f32>(1.0))+dot(p1,vec4<f32>(1.0));
  }`;
  const rowBegin=code.indexOf(c.mlpPair?'fn rowPair(':'fn rowSum('),projectBegin=code.indexOf('fn project('),mainBegin=code.indexOf('@compute',projectBegin);
  if(rowBegin<0||projectBegin<0||mainBegin<0)throw new Error('Half product source seam');
  if(c.mlpHalfProduct==='down'||c.mlpHalfProduct==='both')code=code.slice(0,projectBegin)+code.slice(projectBegin,mainBegin).replaceAll('dot8(','dot8halfProduct(')+code.slice(mainBegin);
  if(c.mlpHalfProduct==='gate'||c.mlpHalfProduct==='both')code=code.slice(0,rowBegin)+code.slice(rowBegin,projectBegin).replaceAll('dot8(','dot8halfProduct(')+code.slice(projectBegin);
 }
 if(c.mlpHalfDot){
  code+=DEQUANT_F16+`
fn dot8half(packed:u32,zp:f32,a0:vec4<f32>,a1:vec4<f32>)->f32{
   return f32(dot8h(packed,f16(zp),vec4<f16>(a0),vec4<f16>(a1)));
  }`;
  const rowBegin=code.indexOf(c.mlpPair?'fn rowPair(':'fn rowSum('),projectBegin=code.indexOf('fn project('),mainBegin=code.indexOf('@compute',projectBegin);
  if(rowBegin<0||projectBegin<0||mainBegin<0)throw new Error('Half dot source seam');
  if(c.mlpHalfDot==='down'||c.mlpHalfDot==='both')code=code.slice(0,projectBegin)+code.slice(projectBegin,mainBegin).replaceAll('dot8(','dot8half(')+code.slice(mainBegin);
  if(c.mlpHalfDot==='gate'||c.mlpHalfDot==='both')code=code.slice(0,rowBegin)+code.slice(rowBegin,projectBegin).replaceAll('dot8(','dot8half(')+code.slice(projectBegin);
 }
 if(c.mlpSoA){
  if(c.mlpVecLoad||c.mlpSlices===160)throw new Error('SoA requires ordinary whole quant blocks and no competing weight binding');
  const rowBegin=code.indexOf(c.mlpPair?'fn rowPair(':'fn rowSum('),rowEnd=code.indexOf('const KQUANT:',rowBegin);
  const projectBegin=code.indexOf('fn project('),projectEnd=code.indexOf('@compute',projectBegin);
  if(rowBegin<0||rowEnd<0||projectBegin<0||projectEnd<0)throw new Error('SoA source seam');
  if(c.mlpSoA==='down'||c.mlpSoA==='both'){
   let count=0;const body=code.slice(projectBegin,projectEnd).replace(/blob\[(base|base2)(?:\s*\+\s*(\d)u)?\]/g,(_m,base:string,n:string)=>{count++;return `down_soa[((group*BLOCKS_PER_SLICE+b)*4u+${n??0}u)*HIDDEN+row${base==='base2'?'+WG':''}]`;});
   if(count!==(c.projectPair?8:4))throw new Error('SoA down reads '+count);
   code=code.slice(0,projectBegin)+body+code.slice(projectEnd);
   code+='\n@group(0) @binding(20) var<storage,read> down_soa:array<u32>;\n';
   kernel={...kernel,bindings:[...kernel.bindings,{binding:20,name:'source',access:'read'}]};
  }
  if(c.mlpSoA==='gate'||c.mlpSoA==='both'){
   let count=0;const body=code.slice(rowBegin,rowEnd).replace(/blob\[(base|upbase)(?:\s*\+\s*(\d)u)?\]/g,(_m,base:string,n:string)=>{count++;return `gate_soa[(${base==='upbase'?'row+FFN':'row'})*BLOCKS*4u+${n??0}u*BLOCKS+block]`;});
   if(count!==(c.mlpPair?8:4))throw new Error('SoA gate reads '+count);
   code=code.slice(0,rowBegin)+body+code.slice(rowEnd);
   code+='\n@group(0) @binding(19) var<storage,read> gate_soa:array<u32>;\n';
   kernel={...kernel,bindings:[...kernel.bindings,{binding:19,name:'weights',access:'read'}]};
  }
 }
 if(c.mlpTranspose){
  if(c.mlpStageSwizzle||c.mlpPrivate)throw new Error('Transposed stage cannot combine with alternate stage addressing');
  let count=0;
  code=code.replace(/\bstage\[o(?:\s*\+\s*(\d+)u)?\]/g,(_m,n:string)=>{count++;return `stage[block+${Number(n??0)*32}u]`;});
  if(count!==32)throw new Error('Transposed MLP input seam '+count);
  code=code.replace(/\bstage\[i\]/g,'stage[((i&31u)<<5u)+(i>>5u)]');
 }
 return magicDot(maskedDot(swizzleStage({...kernel,code},c.mlpStageSwizzle??false),c.maskMlp??false),c.magicMlp??false);
}
export function efficientFold(kernel:KernelSource,c:Candidate):KernelSource{
 if(c.foldSubgroup){
  const begin=kernel.code.indexOf('var<workgroup> part:');if(begin<0)throw new Error('Fold subgroup seam');
  const code=kernel.code.slice(0,begin)+`var<workgroup> part:array<f32,${kernel.workgroupSize}>;
   @compute @workgroup_size(WG)
   fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) local:u32){
    let slot=local%ROWS_PER_WG;let lane=local/ROWS_PER_WG;let row=wid.x*ROWS_PER_WG+slot;
    var acc=0.0;if(row<HIDDEN){for(var s=lane;s<FOLD_SLICES;s+=LANES){acc+=partials_in[s*HIDDEN+row];}}
    for(var delta=ROWS_PER_WG;delta<32u;delta*=2u){acc+=subgroupShuffleXor(acc,delta);}
    if(local%32u<ROWS_PER_WG){part[(local/32u)*ROWS_PER_WG+slot]=acc;}workgroupBarrier();
    if(local<ROWS_PER_WG&&row<HIDDEN){var sum=residual_out[row];for(var sg=0u;sg<WG/32u;sg++){sum+=part[sg*ROWS_PER_WG+local];}residual_out[row]=sum;}
   }`;
  return {...kernel,code};
 }
 if(!c.foldFast)return kernel;
 const begin=kernel.code.indexOf('var<workgroup> part:');
 if(begin<0)throw new Error('FOLD source seam');
 // Preserve the existing dispatch geometry. Only ROWS_PER_WG lanes do useful work;
 // each sums the original LANES strided subsequences in the same order.
 const code=kernel.code.slice(0,begin)+`@compute @workgroup_size(WG)
 fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) local:u32){
 if(local>=ROWS_PER_WG){return;}let row=wid.x*ROWS_PER_WG+local;if(row>=HIDDEN){return;}
 var sum=residual_out[row];for(var lane=0u;lane<LANES;lane++){var acc=0.0;for(var s=lane;s<FOLD_SLICES;s+=LANES){acc+=partials_in[s*HIDDEN+row];}sum+=acc;}residual_out[row]=sum;
 }`;
 return {...kernel,code};
}
