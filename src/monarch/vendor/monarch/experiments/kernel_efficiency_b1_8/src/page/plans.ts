import {planOptionsFor,CARRIED_TUNING,type PlanTuning} from '../../../flash_core_split_n_b1_4/src/pure/plan_options.ts';
import type {ModelShape} from '../../../flash_core_split_n_b1_4/src/pure/shape.ts';
import type {MatvecGeometry,MatvecSite} from '../../../flash_core_split_n_b1_4/src/pure/plan.ts';
import type {DequantVariant} from '../../../flash_core_split_n_b1_4/src/page/wgsl.ts';
type MlpDotIlp = 'gate' | 'down' | 'both';
export interface Candidate {key:string;mlpDotIlp?:MlpDotIlp;mlpSoA?:MlpDotIlp;mlpHalfDot?:MlpDotIlp;mlpHalfProduct?:MlpDotIlp;mlpTranspose?:boolean;magicMlp?:boolean;magicMatvec?:boolean;foldLanes?:number;foldSubgroup?:boolean;noGpuTimestamps?:boolean;tokensPerPass?:number;maskMlp?:boolean;maskMatvec?:boolean;mlpVecLoad?:boolean;mergeSubgroup?:boolean;convFuse?:boolean;flashOnline?:boolean;flashBlocks?:number;normOnceHead?:boolean;normOnceMatvec?:boolean;normOnceMlp?:boolean;mlpPrivate?:boolean;matvecPrivate?:boolean;mlpStageSwizzle?:boolean;matvecStageSwizzle?:boolean;mlpPair?:boolean;projectPair?:boolean;foldFast?:boolean;flash:boolean;unfused:boolean;padding:number;flashMode:string;normSubgroup?:boolean;mlpMode?:string;mlpSlices?:number;mlpWorkgroup?:number;matvec:Partial<Record<MatvecSite,{geometry:MatvecGeometry;variant:DequantVariant}>>;}
let candidates:Candidate[]=[];
export function configureCandidates(value:Candidate[]){if(!value.length)throw new Error('No registered candidates');candidates=value;}
export function candidateOf(key:string){const c=candidates.find(c=>c.key===key);if(!c)throw new Error('Unknown candidate '+key);return c;}
export function armPlanOptions(shape:ModelShape,position:number,flashPosition:number,tuning:PlanTuning){
 return candidates.map(c=>{
  const options=planOptionsFor(shape,1,position,flashPosition,c.unfused?'A2N1':c.flash?'A2N2':'A2',tuning);
  const carried={...options.geometry.carried};for(const [site,value] of Object.entries(c.matvec))carried[site as MatvecSite]=value.geometry;
  return {key:c.key,options:{...options,flash:{...options.flash,blocks:c.flashBlocks??options.flash.blocks},slices:{...options.slices,mlp:c.mlpSlices??options.slices.mlp},geometry:{...options.geometry,foldGeometry:{...options.geometry.foldGeometry,threadsPerRow:c.foldLanes??options.geometry.foldGeometry.threadsPerRow},carried,blockWorkgroupSize:c.mlpWorkgroup??options.geometry.blockWorkgroupSize}}};
 });
}

export function candidateMaxSlices(){return Math.max(80,...candidates.map(c=>c.mlpSlices??80));}
