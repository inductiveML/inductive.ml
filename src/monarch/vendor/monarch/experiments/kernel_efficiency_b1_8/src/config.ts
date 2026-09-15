// B1.8 reuses the pinned B1.4 instruments and precision. Decoder arms are gated by P0.
export {PRECISION,FUSION,PROMPT,CHROMIUM,PINS,SHAPE,KERNELS,RUNTIME,ORT,LEVERS,PROTOCOL,P0A,P0B,P0C,SENTINEL} from '../../flash_core_split_n_b1_4/src/config.ts';
import {SERVER as PRIOR_SERVER} from '../../flash_core_split_n_b1_4/src/config.ts';
export const SERVER={...PRIOR_SERVER,port:47320} as const;
export const UNIT={name:'kernel_efficiency_b1_8',spec:'MONARCH B1.8 DECODE_MAX_TPS',priorClosureCell:'O6.8',closureCell:'O6.9',closureId:'B1_8_KERNEL_EFFICIENCY',referenceUnit:'attention_bandwidth_b1_3',weightsUnit:'flash_core_split_n_b1_4',instrumentUnit:'webgpu_pricing_probe_b1_0'} as const;

export const ARMS=[{key:"A2"},{key:"G"},{key:"F"},{key:"GF"},{key:"R"}] as const;

import {DEVICE_MINIMUMS as PRIOR_MINIMUMS} from '../../flash_core_split_n_b1_4/src/config.ts';
const {maxComputeWorkgroupStorageSizeFlash,...webgpuMinimums}=PRIOR_MINIMUMS;
// Flash storage is another requirement on the same WebGPU limit, not a limit name.
export const DEVICE_MINIMUMS={...webgpuMinimums,maxComputeWorkgroupStorageSize:Math.max(webgpuMinimums.maxComputeWorkgroupStorageSize,maxComputeWorkgroupStorageSizeFlash)};
