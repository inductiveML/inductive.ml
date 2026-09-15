import type {KernelSource} from '../../../flash_core_split_n_b1_4/src/page/wgsl.ts';

// Integer-only startup permutation. Each destination word has exactly one writer.
export function quantLayoutKernel(hidden:number,ffn:number,kind:'gate'|'down'):KernelSource {
 const rows=kind==='gate'?2*ffn:hidden,blocks=kind==='gate'?hidden/32:ffn/32;
 const index=kind==='gate'
  ? 'let row=i/(BLOCKS*4u);let word=(i/BLOCKS)%4u;let block=i%BLOCKS;let at=(row*BLOCKS+block)*4u+word;'
  : 'let block=i/(ROWS*4u);let word=(i/ROWS)%4u;let row=i%ROWS;let at=(block*ROWS+row)*4u+word;';
 return {workgroupSize:256,bindings:[{binding:0,name:'weights',access:'read'},{binding:1,name:'destination',access:'read_write'}],code:`
  const ROWS:u32=${rows}u;const BLOCKS:u32=${blocks}u;
  @group(0) @binding(0) var<storage,read> input_words:array<u32>;
  @group(0) @binding(1) var<storage,read_write> output_words:array<u32>;
  @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
   let i=gid.x;if(i>=ROWS*BLOCKS*4u){return;}${index}output_words[i]=input_words[at];
  }`};
}
