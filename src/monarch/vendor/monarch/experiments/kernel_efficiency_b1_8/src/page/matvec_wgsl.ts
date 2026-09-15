import {matvecKernel as original} from '../../../flash_core_split_n_b1_4/src/page/wgsl.ts';
export function matvecKernel(args:Parameters<typeof original>,subgroupNorm:boolean):ReturnType<typeof original>{
 const kernel=original(...args);if(!subgroupNorm||args[0]==='matvec_residual')return kernel;
 let code=kernel.code;
 const old=`  reduce[local] = sum;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(reduce[0] / f32(COLS) + EPSILON);`;
 if(code.split(old).length!==2)throw new Error('Matvec RMS source seam drift');
 code=code.replace(old,`  let sgSum = subgroupAdd(sum);
  if (local % SG == 0u) { reduce[local / SG] = sgSum; }
  workgroupBarrier();
  var squareSum = 0.0;
  for (var subgroup = 0u; subgroup < WG / SG; subgroup++) { squareSum += reduce[subgroup]; }
  let inv = inverseSqrt(squareSum / f32(COLS) + EPSILON);`);
 return {...kernel,code};
}
