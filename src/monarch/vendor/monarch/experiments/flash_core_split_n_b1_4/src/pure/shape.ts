// Model geometry and the int4 byte model. One source of truth for the real LFM2.5-230M shape,
// for the synthetic fixed-byte stacks of P3, and for every byte figure the desk model prices.

export type LayerKind = "conv" | "attention";

export interface ModelShape {
  readonly layerTypes: readonly LayerKind[];
  readonly hidden: number;
  readonly ffn: number;
  readonly vocab: number;
  readonly heads: number;
  readonly kvHeads: number;
  readonly headDim: number;
  readonly convCache: number;
  readonly quantBits: number;
  readonly quantBlock: number;
}

export interface QuantMatrixBytes {
  readonly quant: number;
  readonly scales: number;
  readonly zeroPoints: number;
  readonly total: number;
}

// MatMulNBits storage: rows x ceil(cols/block) blocks of `bits`-packed weights, one fp32 scale
// per block, and (only when the exporter emitted them) one `bits`-packed zero point per block.
export function quantMatrixBytes(rows: number, cols: number, zeroPoints: boolean, bits: number, block: number): QuantMatrixBytes {
  if (!Number.isInteger(rows) || rows <= 0 || !Number.isInteger(cols) || cols <= 0) {
    throw new Error(`quantMatrixBytes: bad shape ${rows}x${cols}`);
  }
  if (cols % block !== 0) {
    throw new Error(`quantMatrixBytes: cols ${cols} not a multiple of block ${block}`);
  }
  if (bits !== 4) {
    throw new Error(`quantMatrixBytes: only 4-bit packing is registered, got ${bits}`);
  }
  const blocks = cols / block;
  const quant = (rows * cols * bits) / 8;
  const scales = rows * blocks * 4;
  const zp = zeroPoints ? rows * Math.ceil(blocks / 2) : 0;
  return { quant, scales, zeroPoints: zp, total: quant + scales + zp };
}

export interface WeightMatrix {
  readonly name: string;
  readonly layer: number;
  readonly role: string;
  readonly rows: number;
  readonly cols: number;
  readonly zeroPoints: boolean;
  readonly bytes: QuantMatrixBytes;
}

// Attention sits at every second layer starting at index 2, which reproduces LFM2.5-230M's
// registered layer_types at L = 14 and extends it to the P3 depths.
export function layerTypesFor(layers: number): readonly LayerKind[] {
  if (!Number.isInteger(layers) || layers < 3) {
    throw new Error(`layerTypesFor: layers must be an integer >= 3, got ${layers}`);
  }
  const types: LayerKind[] = [];
  for (let index = 0; index < layers; index += 1) {
    types.push(index >= 2 && index % 2 === 0 ? "attention" : "conv");
  }
  return types;
}

export function attentionLayerCount(layers: number): number {
  return layerTypesFor(layers).filter((kind) => kind === "attention").length;
}

export function layerMatrices(shape: ModelShape, layer: number): readonly WeightMatrix[] {
  const kind = shape.layerTypes[layer];
  if (kind === undefined) {
    throw new Error(`layerMatrices: layer ${layer} out of range`);
  }
  const { hidden, ffn, kvHeads, headDim, quantBits: bits, quantBlock: block } = shape;
  const kv = kvHeads * headDim;
  const make = (role: string, name: string, rows: number, cols: number): WeightMatrix => ({
    name,
    layer,
    role,
    rows,
    cols,
    zeroPoints: false,
    bytes: quantMatrixBytes(rows, cols, false, bits, block),
  });
  const mlp: readonly WeightMatrix[] = [
    make("mlp.gate_proj", `/model/layers.${layer}/mlp/gate_proj/MatMul_Q4`, ffn, hidden),
    make("mlp.up_proj", `/model/layers.${layer}/mlp/up_proj/MatMul_Q4`, ffn, hidden),
    make("mlp.down_proj", `/model/layers.${layer}/mlp/down_proj/MatMul_Q4`, hidden, ffn),
  ];
  if (kind === "conv") {
    return [
      make("conv.in_proj", `/model/layers.${layer}/conv/in_proj/MatMul_Q4`, 3 * hidden, hidden),
      make("conv.out_proj", `/model/layers.${layer}/conv/out_proj/MatMul_Q4`, hidden, hidden),
      ...mlp,
    ];
  }
  return [
    make("attn.q_proj", `/model/layers.${layer}/attn/q_proj/MatMul_Q4`, hidden, hidden),
    make("attn.k_proj", `/model/layers.${layer}/attn/k_proj/MatMul_Q4`, kv, hidden),
    make("attn.v_proj", `/model/layers.${layer}/attn/v_proj/MatMul_Q4`, kv, hidden),
    make("attn.o_proj", `/model/layers.${layer}/attn/o_proj/MatMul_Q4`, hidden, hidden),
    ...mlp,
  ];
}

export function headMatrix(shape: ModelShape): WeightMatrix {
  return {
    name: "logits",
    layer: -1,
    role: "lm_head",
    rows: shape.vocab,
    cols: shape.hidden,
    zeroPoints: true,
    bytes: quantMatrixBytes(shape.vocab, shape.hidden, true, shape.quantBits, shape.quantBlock),
  };
}

export function allMatrices(shape: ModelShape): readonly WeightMatrix[] {
  const out: WeightMatrix[] = [];
  for (let layer = 0; layer < shape.layerTypes.length; layer += 1) {
    out.push(...layerMatrices(shape, layer));
  }
  out.push(headMatrix(shape));
  return out;
}

export interface WeightByteBreakdown {
  readonly matmul: number;
  readonly head: number;
  readonly layerNorms: number;
  readonly convWeights: number;
  readonly headNorms: number;
  readonly finalNorm: number;
  // Sub-kilobyte per-head q/k norms; B1.0's graph table drops tensors below 1 KB, so this is
  // reported separately to keep the cross-check against graph_lfm2_5_230m.json exact.
  readonly layerOtherAtLeast1KiB: number;
  readonly embeddingRow: number;
  readonly ropeRows: number;
  readonly total: number;
}

export function weightByteBreakdown(shape: ModelShape): WeightByteBreakdown {
  const matrices = allMatrices(shape);
  const head = headMatrix(shape).bytes.total;
  const matmul = matrices.reduce((sum, matrix) => sum + matrix.bytes.total, 0);
  const layers = shape.layerTypes.length;
  const layerNorms = layers * 2 * shape.hidden * 4;
  const convWeights = shape.layerTypes.filter((kind) => kind === "conv").length * shape.hidden * shape.convCache * 4;
  const headNorms = shape.layerTypes.filter((kind) => kind === "attention").length * 2 * shape.headDim * 4;
  const finalNorm = shape.hidden * 4;
  // One embedding row: packed nibbles + its fp32 scales + its packed zero points.
  const blocks = shape.hidden / shape.quantBlock;
  const embeddingRow = shape.hidden / 2 + blocks * 4 + Math.ceil(blocks / 2);
  // Two rotary tables, one row each, fp32 over half the head dimension.
  const ropeRows = shape.layerTypes.includes("attention") ? 2 * (shape.headDim / 2) * 4 : 0;
  return {
    matmul,
    head,
    layerNorms,
    convWeights,
    headNorms,
    finalNorm,
    layerOtherAtLeast1KiB: layerNorms + convWeights,
    embeddingRow,
    ropeRows,
    total: matmul + layerNorms + convWeights + headNorms + finalNorm + embeddingRow + ropeRows,
  };
}

export const LFM2_5_230M: ModelShape = {
  layerTypes: layerTypesFor(14),
  hidden: 1024,
  ffn: 2560,
  vocab: 65536,
  heads: 16,
  kvHeads: 8,
  headDim: 64,
  convCache: 3,
  quantBits: 4,
  quantBlock: 32,
};


// KV cache traffic read by the attention kernels at a given position, plus the conv state. P3
// sweeps position, and this is the only term in the token that moves when it does.
export function cacheBytesAtPosition(shape: ModelShape, position: number, kvBytesPerValue: number, convBytesPerValue: number): number {
  const attn = shape.layerTypes.filter((kind) => kind === "attention").length;
  const conv = shape.layerTypes.length - attn;
  const kv = attn * 2 * shape.kvHeads * position * shape.headDim * kvBytesPerValue;
  const convState = conv * shape.hidden * shape.convCache * convBytesPerValue;
  return kv + convState;
}
