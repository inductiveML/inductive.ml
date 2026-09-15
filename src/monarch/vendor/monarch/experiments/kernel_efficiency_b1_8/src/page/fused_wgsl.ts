export * from '../../../flash_core_split_n_b1_4/src/page/fused_wgsl.ts';
import {attnCoreFlashKernel as original} from '../../../flash_core_split_n_b1_4/src/page/fused_wgsl.ts';
let keyPadding=0;
let flashMode='carried';
export function setKeyPadding(padding:number){if(![0,2,4,8].includes(padding))throw new Error('Unregistered padding');keyPadding=padding;}
export function setFlashMode(mode:string){if(!['carried','subgroup','vec4','both'].includes(mode))throw new Error('Unregistered flash variant');flashMode=mode;}
function once(code:string,from:string,to:string){if(code.split(from).length!==2)throw new Error('Flash source seam drift: '+from.slice(0,70));return code.replace(from,to);}
export function attnCoreFlashKernel(...args:Parameters<typeof original>):ReturnType<typeof original>{
 const kernel=original(...args);let code=kernel.code;
 const shape=args[0],stride=shape.headDim+keyPadding;
 if(keyPadding){code=once(code,`var<workgroup> ktile: array<f16, ${64*shape.headDim}>;`,`var<workgroup> ktile: array<f16, ${64*stride}>;`);
 code=once(code,'ktile[p * HEAD_DIM + local]',`ktile[p * ${stride}u + local]`);code=once(code,'ktile[local * HEAD_DIM + d]',`ktile[local * ${stride}u + d]`);}
 if(flashMode==='subgroup'||flashMode==='both'){
 code=once(code,`    reduce[local] = 0.0;
    workgroupBarrier();
    if (local < HEAD_DIM) { reduce[local] = qk[off + local] * qk[off + local]; }
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
      if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
      workgroupBarrier();
    }
    let inv = inverseSqrt(reduce[0] / f32(HEAD_DIM) + EPSILON);`,
 `    let square = qk[off + local] * qk[off + local];
    let subgroupSquare = subgroupAdd(square);
    if (local % 32u == 0u) { reduce[local / 32u] = subgroupSquare; }
    workgroupBarrier();
    let inv = inverseSqrt((reduce[0] + reduce[1]) / f32(HEAD_DIM) + EPSILON);`);
 code=once(code,`      reduce[local] = sv[g];
      workgroupBarrier();
      for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
        workgroupBarrier();
      }
      peak[g] = reduce[0];`,
 `      let subgroupPeak = subgroupMax(sv[g]);
      if (local % 32u == 0u) { reduce[local / 32u] = subgroupPeak; }
      workgroupBarrier();
      peak[g] = max(reduce[0], reduce[1]);`);
 code=once(code,`      reduce[local] = e;
      workgroupBarrier();
      for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
        workgroupBarrier();
      }
      l[g] = l[g] * corr[g] + reduce[0];`,
 `      let subgroupDenom = subgroupAdd(e);
      if (local % 32u == 0u) { reduce[local / 32u] = subgroupDenom; }
      workgroupBarrier();
      l[g] = l[g] * corr[g] + (reduce[0] + reduce[1]);`);
 }
 if(flashMode==='vec4'||flashMode==='both'){
 const group=shape.heads/shape.kvHeads,at=keyPadding?`${stride}u`:'HEAD_DIM';
 code=once(code,`      var a: array<f32, ${group}>;
      for (var g = 0u; g < GROUP; g = g + 1u) { a[g] = 0.0; }
      for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
        let kd = f32(ktile[local * ${at} + d]);
        for (var g = 0u; g < GROUP; g = g + 1u) { a[g] = a[g] + qk[g * HEAD_DIM + d] * kd; }
      }
      for (var g = 0u; g < GROUP; g = g + 1u) { sv[g] = a[g] * SCALE; }`,
 `      var a: array<vec4<f32>, ${group}>;
      for (var g = 0u; g < GROUP; g = g + 1u) { a[g] = vec4<f32>(0.0); }
      for (var d = 0u; d < HEAD_DIM; d = d + 4u) {
        let b = local * ${at} + d;
        let kd = vec4<f32>(f32(ktile[b]), f32(ktile[b+1u]), f32(ktile[b+2u]), f32(ktile[b+3u]));
        for (var g = 0u; g < GROUP; g = g + 1u) {
          let q = g * HEAD_DIM + d;
          a[g] = a[g] + vec4<f32>(qk[q],qk[q+1u],qk[q+2u],qk[q+3u]) * kd;
        }
      }
      for (var g = 0u; g < GROUP; g = g + 1u) { sv[g] = ((a[g].x+a[g].y)+(a[g].z+a[g].w)) * SCALE; }`);
 }
 return {...kernel,code};
}

import {mlpFusedKernel as originalMlp} from '../../../flash_core_split_n_b1_4/src/page/fused_wgsl.ts';
import {DEQUANT_F16} from '../../../flash_core_split_n_b1_4/src/page/wgsl.ts';
let mlpMode='carried';
export function setMlpMode(mode:string){if(!['carried','subgroup','packed','both'].includes(mode))throw new Error('Unknown MLP mode');mlpMode=mode;}
export function mlpFusedKernel(...args:Parameters<typeof originalMlp>):ReturnType<typeof originalMlp>{
 const halfBlock=args[2].slices===160;
 const kernel=originalMlp(args[0],args[1],halfBlock?{...args[2],slices:80}:args[2]);let code=kernel.code;
 if(halfBlock){
  const wg=args[2].workgroupSize;
  code=once(code,'const SLICES: u32 = 80u;','const SLICES: u32 = 160u;');
  code=once(code,'const ROWS_PER_SLICE: u32 = 32u;','const ROWS_PER_SLICE: u32 = 16u;');
  code=once(code,`const SLICE_ITERS: u32 = ${32/(wg/32)}u;`,`const SLICE_ITERS: u32 = ${16/(wg/32)}u;`);
  code=once(code,'var<workgroup> hidden_slice: array<f32, 32>;','var<workgroup> hidden_slice: array<f32, 16>;');
  code=once(code,'KQUANT + (sliceBase + row) * WORDS_PER_ROW_SLICE + b * 4u','KQUANT + ((group / 2u) * HIDDEN + row) * 4u + (group % 2u) * 2u');
  code=once(code,'KSCALES + (sliceBase + row) * BLOCKS_PER_SLICE + b','KSCALES + (group / 2u) * HIDDEN + row');
  const begin=code.indexOf('fn project('),end=code.indexOf('@compute',begin);let body=code.slice(begin,end);let removed=0;
  body=body.replace(/      part = part \+ dot8\(blob\[base \+ [23]u\], 8\.0,[\s\S]*?\);/g,()=>{removed++;return '';});
  if(removed!==2)throw new Error('Half-block source seam drift');code=code.slice(0,begin)+body+code.slice(end);
 }

 // B1.4's blob packs each 32-column K block separately across output rows.
 // A larger logical slice spans several such original blocks, never a new blob layout.
 if(!halfBlock&&args[2].slices!==80){
  if(![20,40].includes(args[2].slices))throw new Error('Unregistered MLP slices');
  code=once(code,'KQUANT + (sliceBase + row) * WORDS_PER_ROW_SLICE + b * 4u','KQUANT + ((group * BLOCKS_PER_SLICE + b) * HIDDEN + row) * 4u');
  code=once(code,'KSCALES + (sliceBase + row) * BLOCKS_PER_SLICE + b','KSCALES + (group * BLOCKS_PER_SLICE + b) * HIDDEN + row');
 }
 if(mlpMode==='subgroup'||mlpMode==='both'){
  code=once(code,'  reduce[local] = sum;',`  let sgSum = subgroupAdd(sum);
  if (local % SG == 0u) { reduce[local / SG] = sgSum; }`);
  code=once(code,`  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(reduce[0] / f32(HIDDEN) + EPSILON);`,
 `  var squareSum = 0.0;
  for (var subgroup = 0u; subgroup < WG / SG; subgroup++) { squareSum += reduce[subgroup]; }
  let inv = inverseSqrt(squareSum / f32(HIDDEN) + EPSILON);`);
 }
 if(mlpMode==='packed'||mlpMode==='both'){
  let count=0;
  code=code.replace(/part = part \+ dot8\((blob\[[^\]]+\]), 8\.0,\s*(vec4<f32>\([^;]+?)\);/g,(_m,weight,input)=>{count++;const halves=input.replaceAll('vec4<f32>','vec4<f16>').replace(/(stage|hidden_slice)\[[^\]]+\]/g,(at:string)=>`f16(${at})`);return `part = part + f32(dot8h(${weight}, f16(8.0), ${halves}));`;});
  if(count!==8)throw new Error('MLP dot source seam drift '+count);
  code+='\n'+DEQUANT_F16;
 }
 return {...kernel,code};
}
