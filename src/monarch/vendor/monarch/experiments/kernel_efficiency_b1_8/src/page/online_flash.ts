import type {KernelSource,RuntimeShape} from '../../../flash_core_split_n_b1_4/src/page/wgsl.ts';
import {STATE_STRUCT} from '../../../flash_core_split_n_b1_4/src/page/wgsl.ts';
import type {AttnCoreOffsets} from '../../../flash_core_split_n_b1_4/src/page/fused_wgsl.ts';
export function subgroupMerge(kernel:KernelSource,splits:number,enabled:boolean):KernelSource{
 if(!enabled)return kernel;if(splits>32)throw new Error('Subgroup merge supports at most 32 splits');
 const begin=kernel.code.indexOf('@compute');if(begin<0)throw new Error('Merge source seam');
 const code=kernel.code.slice(0,begin)+`@compute @workgroup_size(WG)
 fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(subgroup_invocation_id) lane:u32){
  let first=(wid.y*SPLITS*HEADS+wid.x)*RECORD;
  var mine=NEG_INF;if(lane<SPLITS){mine=flash[first+lane*HEADS*RECORD+HEAD_DIM];}
  let peak=subgroupMax(mine);var weight=0.0;var den=0.0;
  if(lane<SPLITS){weight=exp(mine-peak);den=weight*flash[first+lane*HEADS*RECORD+HEAD_DIM+1u];}
  let denominator=subgroupAdd(den);var numerator=0.0;
  for(var s=0u;s<SPLITS;s++){numerator+=subgroupShuffle(weight,s)*flash[first+s*HEADS*RECORD+local];}
  destination[(wid.y*HEADS+wid.x)*HEAD_DIM+local]=f16(numerator/denominator);
 }`;
 return {...kernel,code};
}
export function onlineFlash(shape:RuntimeShape,offsets:AttnCoreOffsets,splits:number):KernelSource{
 if(shape.headDim!==64||shape.heads/shape.kvHeads!==2)throw new Error('Online flash requires 64-d GQA2');
 return {workgroupSize:64,bindings:[
  {binding:0,name:'blob',access:'read'},{binding:5,name:'state',access:'read'},
  {binding:6,name:'key_cache',access:'read_write'},{binding:7,name:'value_cache',access:'read_write'},
  {binding:8,name:'projection',access:'read'},{binding:10,name:'flash',access:'read_write'},
 ],code:`enable f16;
 enable subgroups;
 ${STATE_STRUCT}
 const SPLITS:u32=${splits}u;const MAX_POSITIONS:u32=${shape.maxPositions}u;
 @group(0) @binding(0) var<storage,read> blob:array<u32>;
 @group(0) @binding(5) var<storage,read> state:State;
 @group(0) @binding(6) var<storage,read_write> key_cache:array<f16>;
 @group(0) @binding(7) var<storage,read_write> value_cache:array<f16>;
 @group(0) @binding(8) var<storage,read> projection:array<f32>;
 @group(0) @binding(10) var<storage,read_write> flash:array<f32>;
 @compute @workgroup_size(64)
 fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(subgroup_invocation_id) lane:u32){
  let kvHead=wid.x/SPLITS;let split=wid.x%SPLITS;let group=local/32u;let head=kvHead*2u+group;
  let position=subgroupBroadcastFirst(state.position)+wid.y;let total=position+1u;
  let blockLen=(total+SPLITS-1u)/SPLITS;let start=split*blockLen;let stop=min(start+blockLen,total);let owner=(total-1u)/blockLen;
  let base=wid.y*2048u;
  var qlo=projection[base+head*64u+lane];var qhi=projection[base+head*64u+lane+32u];
  let qinv=inverseSqrt(subgroupAdd(qlo*qlo+qhi*qhi)/64.0+${shape.epsilon});
  qlo=qlo*qinv*bitcast<f32>(blob[${offsets.qNorm}u+lane]);qhi=qhi*qinv*bitcast<f32>(blob[${offsets.qNorm}u+lane+32u]);
  let co=bitcast<f32>(blob[${offsets.cos}u+position*32u+lane]);let si=bitcast<f32>(blob[${offsets.sin}u+position*32u+lane]);
  let qrot=qlo*co-qhi*si;qhi=f32(f16(qhi*co+qlo*si));qlo=f32(f16(qrot));
  var klo=projection[base+1024u+kvHead*64u+lane];var khi=projection[base+1024u+kvHead*64u+lane+32u];
  let kinv=inverseSqrt(subgroupAdd(klo*klo+khi*khi)/64.0+${shape.epsilon});
  klo=klo*kinv*bitcast<f32>(blob[${offsets.kNorm}u+lane]);khi=khi*kinv*bitcast<f32>(blob[${offsets.kNorm}u+lane+32u]);
  let krot=klo*co-khi*si;khi=f32(f16(khi*co+klo*si));klo=f32(f16(krot));
  let vlo=f32(f16(projection[base+1536u+kvHead*64u+lane]));let vhi=f32(f16(projection[base+1536u+kvHead*64u+lane+32u]));
  if(split==owner&&group==0u){let at=(kvHead*MAX_POSITIONS+position)*64u+lane;
   key_cache[at]=f16(klo);key_cache[at+32u]=f16(khi);value_cache[at]=f16(vlo);value_cache[at+32u]=f16(vhi);
  }
  var peak=-3.0e38;var denom=0.0;var nlo=0.0;var nhi=0.0;
  for(var p=start;p<stop;p++){
   var kl=klo;var kh=khi;var vl=vlo;var vh=vhi;
   if(p!=position){let at=(kvHead*MAX_POSITIONS+p)*64u+lane;kl=f32(key_cache[at]);kh=f32(key_cache[at+32u]);vl=f32(value_cache[at]);vh=f32(value_cache[at+32u]);}
   let score=subgroupAdd(qlo*kl+qhi*kh)*0.125;
   let next=max(peak,score);let alpha=exp(peak-next);let beta=exp(score-next);
   denom=denom*alpha+beta;nlo=nlo*alpha+vl*beta;nhi=nhi*alpha+vh*beta;peak=next;
  }
  let rec=((wid.y*SPLITS+split)*16u+head)*68u;
  flash[rec+lane]=nlo;flash[rec+lane+32u]=nhi;
  if(lane==0u){flash[rec+64u]=peak;flash[rec+65u]=denom;}
 }`};
}
