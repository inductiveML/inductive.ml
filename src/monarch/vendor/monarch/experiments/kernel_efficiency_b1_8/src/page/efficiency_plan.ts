export * from '../../../flash_core_split_n_b1_4/src/pure/plan.ts';
import type {PlannedDispatch as PriorDispatch} from '../../../flash_core_split_n_b1_4/src/pure/plan.ts';
import type {Candidate} from './plans.ts';
import type {ModelShape} from '../../../flash_core_split_n_b1_4/src/pure/shape.ts';
export type PlannedDispatch=Omit<PriorDispatch,'kind'> & {kind:PriorDispatch['kind']|'input_norm'|'conv_proj_core'};
export function hasNormOnce(c:Candidate,d:{kind:string}):boolean{
 return d.kind==='mlp_fused'?c.normOnceMlp??false:
  ['norm_matvec','attn_proj','norm_head'].includes(d.kind)?(c.normOnceMatvec||(d.kind==='norm_head'&&c.normOnceHead))??false:false;
}
export function efficiencyPlan(raw:readonly PriorDispatch[],c:Candidate,shape:ModelShape,position:number,queries:number):PlannedDispatch[]{
 const result:PlannedDispatch[]=[];
 for(const d of raw){
  if(c.flashOnline&&d.kind==='attn_core_flash'){
   if(queries!==1)throw new Error('Online flash accounting requires one decode query');
   const splits=c.flashBlocks!;
   // Logical shader payload, not measured DRAM traffic. Each q-head subgroup reads its
   // own Q and current K/V in every split, and reads its own copy of previous KV rows.
   const projectionRead=shape.heads*splits*shape.headDim*4*3;
   const cacheRead=shape.heads*2*position*shape.headDim*2;
   const payloadWrite=shape.heads*splits*(shape.headDim+2)*4+shape.kvHeads*2*shape.headDim*2;
   result.push({...d,index:result.length,readBytes:projectionRead+cacheRead,writeBytes:payloadWrite});continue;
  }
  if(c.convFuse&&d.kind==='conv_core')continue;
  if(c.convFuse&&d.kind==='norm_matvec'){
   result.push({...d,index:result.length,kind:'conv_proj_core',constantBytes:d.constantBytes+12288,readBytes:d.readBytes+6144,writeBytes:8192});continue;
  }
  if(hasNormOnce(c,d))result.push({...d,index:result.length,name:d.name+'.input_norm',kind:'input_norm',weight:null,reads:[],constantBytes:4096,readBytes:4096,writeBytes:4096,workgroups:1,site:null,geometry:null,fusion:{...d.fusion,slices:0,foldSlices:0}});
  result.push({...d,index:result.length});
 }
 return result;
}

function landingIndex(plan: readonly PlannedDispatch[], layer: number, prefix: string, label: string): number {
  let last = -1;
  for (const dispatch of plan) {
    if (dispatch.layer === layer && dispatch.name.startsWith(prefix)) {
      last = dispatch.index;
    }
  }
  if (last < 0) {
    throw new Error(`${label}: layer ${layer} has no ${prefix} dispatch`);
  }
  const block = plan[last];
  if (block === undefined) {
    throw new Error(`${label}: dispatch ${last} is missing from the plan`);
  }
  if (block.fusion.slices === 0) {
    return last;
  }
  for (let index = last + 1; index < plan.length; index += 1) {
    const dispatch = plan[index];
    if (dispatch === undefined) {
      throw new Error(`${label}: gap at dispatch ${index}`);
    }
    if (dispatch.fusion.foldSlices > 0) {
      return index;
    }
  }
  throw new Error(`${label}: layer ${layer} publishes ${block.fusion.slices} slices that nothing folds`);
}

// Every arm of this unit ends its attention layer with a split-N out-projection that writes the
// residual in place, so this is the out-projection's own index on all four arms. The deferred
// branch is still taken from the metadata, not assumed away.
export function attentionLandingIndex(plan: readonly PlannedDispatch[], layer: number): number {
  return landingIndex(plan, layer, `L${layer}.attn.`, "attentionLandingIndex");
}

// The MLP's landing point, which is where N1 and the carried fused block are comparable: on an N1
// arm the down projection writes the residual in place and lands there, and on a carried arm the
// fused block publishes eighty slices that the next layer's fold consumes.
export function mlpLandingIndex(plan: readonly PlannedDispatch[], layer: number): number {
  return landingIndex(plan, layer, `L${layer}.mlp.`, "mlpLandingIndex");
}
