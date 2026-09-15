// The fused blocks: one dispatch per operator instead of two or three and, for the attention arms
// that keep the block whole, one dispatch per attention layer instead of eighteen for the stack.
// Everything here exists to delete dispatch boundaries, so the two things that make a boundary
// necessary are removed first.
//
// 1. TEN STORAGE BUFFERS. This adapter caps maxStorageBuffersPerShaderStage at 10 and a fused
//    attention block otherwise needs fourteen (two matrices x {packed, scales}, three gammas, two
//    rotary tables, residual in/out, partials in/out, two caches, state). The whole weight file is
//    bound once as `blob: array<u32>` and every section's offset is baked into the module at
//    build time; f32 sections are read with bitcast. Sections are 256 B aligned so a byte offset
//    divided by four is exact. That takes every fused kernel to eight bindings or fewer.
//
// 2. THE CROSS-WORKGROUP SUM. Splitting an output projection's reduction dimension across
//    workgroups leaves partial sums that some other workgroup has to add, and WGSL has no f32
//    atomic and no device-scope release the next dispatch could wait on inside a pass. So the sum
//    is not done here at all: it is deferred to the next norm, which already reads the whole
//    residual. Each block therefore does three things, in this order:
//      stage    x[i] = residual_in[i] + sum over the pending partial slices, then RMSNorm x gamma
//               into workgroup memory -- reads only;
//      forward  residual_out[i] = x[i] for a contiguous row range no other workgroup writes;
//      publish  its own output as one fresh partial slice per workgroup, into partials_out.
//    Reads land on residual_in/partials_in and writes on residual_out/partials_out, so the two
//    pairs ping-pong and no dispatch reads a buffer it is concurrently writing. Slices are summed
//    in index order everywhere, so every deferred sum is bitwise reproducible.
//
// Activations round to f16 at exactly the points the unfused kernels round them -- the q/k/v heads,
// the attention output, the SwiGLU intermediate -- because f16 is the registered activation
// precision and the KV cache dtype. The current position then scores against its own k and v
// exactly as it will when a later token reads them back from the cache.
//
// 3. THE GQA CORE, shared by A1's block and A2's core dispatch, is what makes those two arms the
//    same lever measured two ways. F2's block owns one q head, so it projects the k and v rows of
//    its group once per q head and its walk over the cache reads every cached k and v once per q
//    head: 3.93 MB of duplicated reads per token at this shape. The shared core owns one KV head
//    and every q head that shares it, so each of those bytes is read once. The price is one score
//    row per q head in workgroup memory, because the saving is only real if every q head's scores
//    come out of ONE pass over the key cache -- a second pass would re-read it and hand the
//    duplication straight back. That is what the workgroup storage budget is spent on here.
import {
  DEQUANT,
  HEADER,
  STATE_STRUCT,
  declare,
  type BindingSlot,
  type KernelSource,
  type RuntimeShape,
  type Workgroups,
} from "./wgsl.ts";

// Word offsets into the blob. Every field is `byteOffset / 4` of a section the layout placed, so
// the kernels index `blob` directly and no dispatch reads a uniform.

// What the GQA core alone reads: the two head norms and the rotary tables. A2's core dispatch has
// no matrix of its own, so this is the whole of it; A1's block extends it with the two matrices and
// the block norm.
export interface AttnCoreOffsets {
  readonly qNorm: number;
  readonly kNorm: number;
  readonly cos: number;
  readonly sin: number;
}

export interface MlpBlockOffsets {
  readonly norm: number;
  readonly gateUpQuant: number;
  readonly gateUpScales: number;
  readonly downKQuant: number;
  readonly downKScales: number;
}

export interface FusedGeometry {
  readonly workgroupSize: number;
  readonly subgroupSize: number;
  readonly slices: number;
  // Partial slices pending on entry, which this dispatch reduces while it stages. Zero at the top
  // of a token and after a fold; otherwise the slice count of the dispatch before it.
  readonly foldSlices: number;
}

export const FUSED_BINDING = {
  blob: 0,
  residualIn: 1,
  residualOut: 2,
  partialsIn: 3,
  partialsOut: 4,
  state: 5,
  keyCache: 6,
  valueCache: 7,
  // A2's core alone: the f32 q/k/v scratch its projection wrote, and the f16 attention output its
  // out-projection reads as a source. Neither is a block binding -- the core is not a fused block,
  // it neither stages a residual nor publishes a partial slice.
  projection: 8,
  destination: 9,
  // N2's partials: one record per (query, position block, q head). Index 10 is a binding NUMBER,
  // not an eleventh buffer -- the flash core binds six and the merge binds two, both inside the
  // adapter's ten storage buffers per stage.
  flash: 10,
} as const;

function checkGeometry(kind: string, shape: RuntimeShape, geometry: FusedGeometry, splitDimension: number): void {
  const { workgroupSize, subgroupSize, slices, foldSlices } = geometry;
  if (workgroupSize % subgroupSize !== 0) {
    throw new Error(`${kind}: workgroup ${workgroupSize} is not a multiple of subgroup ${subgroupSize}`);
  }
  if (splitDimension % slices !== 0) {
    throw new Error(`${kind}: ${splitDimension} is not divisible by ${slices} slices`);
  }
  const rowsPerSlice = splitDimension / slices;
  const rowsAtOnce = workgroupSize / subgroupSize;
  if (rowsPerSlice % rowsAtOnce !== 0) {
    throw new Error(`${kind}: ${rowsPerSlice} rows per slice is not a multiple of the ${rowsAtOnce} rows a workgroup sums at once`);
  }
  if (shape.hidden % shape.quantBlock !== 0) {
    throw new Error(`${kind}: hidden ${shape.hidden} is not a multiple of block ${shape.quantBlock}`);
  }
  if (!Number.isInteger(foldSlices) || foldSlices < 0) {
    throw new Error(`${kind}: fold slices must be a non-negative integer, got ${foldSlices}`);
  }
}

// The eight nibbles of one u32 against eight consecutive entries of a workgroup array.
function dotBlock(source: string, indent: string): string {
  const at = (index: number): string => (index === 0 ? `${source}[o]` : `${source}[o + ${index}u]`);
  const quad = (from: number): string => `vec4<f32>(${at(from)}, ${at(from + 1)}, ${at(from + 2)}, ${at(from + 3)})`;
  const lines: string[] = [];
  for (let word = 0; word < 4; word += 1) {
    const slot = word === 0 ? "base" : `base + ${word}u`;
    lines.push(`${indent}part = part + dot8(blob[${slot}], 8.0,\n${indent}  ${quad(word * 8)},\n${indent}  ${quad(word * 8 + 4)});`);
  }
  return lines.join("\n");
}

// One row of a row-major quantised matrix in the blob against the staged, normed input. Lane l
// covers blocks l, l+SG, ... so each lane's 16 B load is contiguous; the trip count is a
// compile-time constant, which keeps the closing subgroupAdd uniform.
function rowSumFn(quantWord: number, scaleWord: number, blocks: number, subgroupSize: number): string {
  return `
const QUANT: u32 = ${quantWord}u;
const SCALES: u32 = ${scaleWord}u;
const BLOCKS: u32 = ${blocks}u;
const BLOCK_ITERS: u32 = ${Math.ceil(blocks / subgroupSize)}u;

fn rowSum(row: u32, lane: u32) -> f32 {
  var acc: f32 = 0.0;
  for (var i = 0u; i < BLOCK_ITERS; i = i + 1u) {
    let block = lane + i * SG;
    if (block < BLOCKS) {
      let base = QUANT + row * BLOCKS * 4u + block * 4u;
      let o = block * 32u;
      var part: f32 = 0.0;
${dotBlock("stage", "      ")}
      acc = acc + part * bitcast<f32>(blob[SCALES + row * BLOCKS + block]);
    }
  }
  return subgroupAdd(acc);
}
`;
}

// The deferred half of the split-K projection: this workgroup's slice of the reduction dimension
// against every output row, written as one fresh partial slice. The matrix is read through its
// K-sliced repack, so the slice a workgroup wants is one linear run and consecutive lanes read
// consecutive rows of it. One thread owns a whole output row, so there is no reduction here at all.
//
// Workgroup g owns exactly one slice of the repacked K -- the fused MLP is the only block left in
// this unit, and its workgroup owns one contiguous run of intermediate rows.
function projectFn(kQuantWord: number, kScaleWord: number, blocksPerSlice: number, source: string): string {
  return `
const KQUANT: u32 = ${kQuantWord}u;
const KSCALES: u32 = ${kScaleWord}u;
const BLOCKS_PER_SLICE: u32 = ${blocksPerSlice}u;
const WORDS_PER_ROW_SLICE: u32 = ${blocksPerSlice * 4}u;

fn project(local: u32, group: u32) {
  let sliceBase = group * HIDDEN;
  for (var row = local; row < HIDDEN; row = row + WG) {
    var acc: f32 = 0.0;
    for (var b = 0u; b < BLOCKS_PER_SLICE; b = b + 1u) {
      let base = KQUANT + (sliceBase + row) * WORDS_PER_ROW_SLICE + b * 4u;
      let o = b * 32u;
      var part: f32 = 0.0;
${dotBlock(source, "      ")}
      acc = acc + part * bitcast<f32>(blob[KSCALES + (sliceBase + row) * BLOCKS_PER_SLICE + b]);
    }
    partials_out[sliceBase + row] = acc;
  }
}
`;
}

// Stage, forward, and norm: parts one and two of the deferred-reduction contract. The forward runs
// between the sum-of-squares barrier and the tree reduction, which is the one window in which
// `stage` holds x and nothing has scaled it yet. Workgroup g owns the contiguous row range
// [g*HIDDEN/SLICES, (g+1)*HIDDEN/SLICES), a partition of the residual, so the dispatch writes every
// row exactly once and no two workgroups race.
function stageBlockFn(shape: RuntimeShape, normWord: number, foldSlices: number): string {
  const source = foldSlices === 0
    ? "    let v = residual_in[i];"
    : `    var v = residual_in[i];
    for (var s = 0u; s < FOLD_SLICES; s = s + 1u) { v = v + partials_in[s * HIDDEN + i]; }`;
  return `
const NORM: u32 = ${normWord}u;
const FOLD_SLICES: u32 = ${foldSlices}u;
const EPSILON: f32 = ${shape.epsilon};

fn stageBlock(local: u32, group: u32) {
  var sum: f32 = 0.0;
  for (var i = local; i < HIDDEN; i = i + WG) {
${source}
    stage[i] = v;
    sum = sum + v * v;
  }
  reduce[local] = sum;
  workgroupBarrier();
  let rowBegin = (group * HIDDEN) / SLICES;
  let rowEnd = ((group + 1u) * HIDDEN) / SLICES;
  for (var i = rowBegin + local; i < rowEnd; i = i + WG) {
    residual_out[i] = stage[i];
  }
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(reduce[0] / f32(HIDDEN) + EPSILON);
  workgroupBarrier();
  for (var i = local; i < HIDDEN; i = i + WG) {
    stage[i] = stage[i] * inv * bitcast<f32>(blob[NORM + i]);
  }
  workgroupBarrier();
}
`;
}

function commonConstants(shape: RuntimeShape, geometry: FusedGeometry): string {
  return `const HIDDEN: u32 = ${shape.hidden}u;
const WG: u32 = ${geometry.workgroupSize}u;
const SG: u32 = ${geometry.subgroupSize}u;
const SLICES: u32 = ${geometry.slices}u;
const ROWS_AT_ONCE: u32 = ${geometry.workgroupSize / geometry.subgroupSize}u;
`;
}

function blockBindings(geometry: FusedGeometry): BindingSlot[] {
  const bindings: BindingSlot[] = [
    { binding: FUSED_BINDING.blob, name: "blob", access: "read" },
    { binding: FUSED_BINDING.residualIn, name: "residual_in", access: "read" },
    { binding: FUSED_BINDING.residualOut, name: "residual_out", access: "read_write" },
  ];
  if (geometry.foldSlices > 0) {
    bindings.push({ binding: FUSED_BINDING.partialsIn, name: "partials_in", access: "read" });
  }
  bindings.push({ binding: FUSED_BINDING.partialsOut, name: "partials_out", access: "read_write" });
  return bindings;
}

// The constants the GQA core reads, and the checks that make its indexing exact. `K_ROW` is where
// the one k row sits in `qk`: after the GROUP q rows, so a workgroup's q heads are contiguous from
// zero and the k row is found at a fixed offset regardless of how many q heads share it.
function attnCoreConstants(shape: RuntimeShape, offsets: AttnCoreOffsets, workgroupSize: number): string {
  const group = shape.heads / shape.kvHeads;
  return `const HEAD_DIM: u32 = ${shape.headDim}u;
const HALF_DIM: u32 = ${shape.headDim / 2}u;
const GROUP: u32 = ${group}u;
const GROUP_ROWS: u32 = ${group + 1}u;
const K_ROW: u32 = ${group * shape.headDim}u;
const KV_WIDTH: u32 = ${shape.kvHeads * shape.headDim}u;
const MAX_POSITIONS: u32 = ${shape.maxPositions}u;
const POS_GROUPS: u32 = ${workgroupSize / shape.headDim}u;
const SCALE: f32 = ${1 / Math.sqrt(shape.headDim)};
const Q_NORM: u32 = ${offsets.qNorm}u;
const K_NORM: u32 = ${offsets.kNorm}u;
const COS: u32 = ${offsets.cos}u;
const SIN: u32 = ${offsets.sin}u;
`;
}

function checkCoreGeometry(kind: string, shape: RuntimeShape, workgroupSize: number): number {
  if (shape.heads % shape.kvHeads !== 0) {
    throw new Error(`${kind}: ${shape.heads} q heads is not a multiple of ${shape.kvHeads} kv heads`);
  }
  if (workgroupSize % shape.headDim !== 0) {
    throw new Error(`${kind}: workgroup ${workgroupSize} is not a multiple of head dim ${shape.headDim}`);
  }
  if (shape.headDim % 2 !== 0) {
    throw new Error(`${kind}: head dim ${shape.headDim} does not split into the two halves the rotation pairs`);
  }
  return shape.heads / shape.kvHeads;
}

// The GROUP q rows and the one k row take the same per-head norm and the same non-interleaved
// rotation, in place in `qk`. The trip count is a compile-time constant, so every barrier is
// uniform; the tree reduction runs over the whole workgroup and sums exact zeros above HEAD_DIM,
// which is why the same source at WG = 256 and at WG = 64 returns bit-identical rows -- the flash
// core and the core it replaces therefore enter their softmax with the same q, k and v, and the
// merge is the only new numerics between them.
function headRowsFn(): string {
  return `
fn headRows(local: u32, position: u32) {
  for (var which = 0u; which < GROUP_ROWS; which = which + 1u) {
    let normBase = select(K_NORM, Q_NORM, which < GROUP);
    let off = which * HEAD_DIM;
    reduce[local] = 0.0;
    workgroupBarrier();
    if (local < HEAD_DIM) { reduce[local] = qk[off + local] * qk[off + local]; }
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
      if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
      workgroupBarrier();
    }
    let inv = inverseSqrt(reduce[0] / f32(HEAD_DIM) + EPSILON);
    workgroupBarrier();
    if (local < HEAD_DIM) { qk[off + local] = qk[off + local] * inv * bitcast<f32>(blob[normBase + local]); }
    workgroupBarrier();
    if (local < HALF_DIM) {
      let c = bitcast<f32>(blob[COS + position * HALF_DIM + local]);
      let s = bitcast<f32>(blob[SIN + position * HALF_DIM + local]);
      let lo = qk[off + local];
      let hi = qk[off + local + HALF_DIM];
      reduce[local] = lo * c - hi * s;
      reduce[local + HALF_DIM] = hi * c + lo * s;
    }
    workgroupBarrier();
    if (local < HEAD_DIM) { qk[off + local] = f32(f16(reduce[local])); }
    workgroupBarrier();
  }
}
`;
}

// The GQA core. One workgroup owns one KV head and the GROUP q heads that share it, which is the
// whole of the attention lever: the k row, the v row and every cached k and v position are read
// once per KV head instead of once per q head.
//
// The saving is only real if every q head's scores come out of ONE pass over the key cache, so the
// pass below loads each cached k row once and dots it against all GROUP q heads before moving on.
// That costs one score row per q head in workgroup memory, and that is what the storage budget is
// spent on -- a second pass would re-read the cache and hand the duplication straight back.
//
// Everything else per q head is a function-scope array: the running maximum, the softmax
// denominator, the weighted V sum in flight. Those are register resident and, being computed from
// workgroup-uniform data, identical in every lane, so they cost no workgroup memory and leave every
// barrier below in uniform control flow. The one that cannot be is the V fold, which is a real
// cross-lane reduction; the GROUP heads take turns through the single `vacc` array with barriers
// between them. The arithmetic is F2's, term for term and in the same order -- the whole difference
// between this and F2's block is which bytes are read, which is what isolates the byte lever.
//
// The caller stages `qk` (GROUP q rows then the k row, projected and unrotated) and `vcur`, and
// declares `reduce`, `scores`, `vacc`, EPSILON, WG and everything attnCoreConstants emits. `sink`
// writes one q head's output row from `acc`, `denom[g]`, `g` and `local`.
function attnCoreFn(group: number, sink: string): string {
  return `${headRowsFn()}
fn attnCore(local: u32, kvHead: u32, position: u32, queryIndex: u32) {
  let total = position + 1u;
  headRows(local, position);

  // One workgroup per KV head, so the current slot is written exactly once and read by nobody: the
  // walks below score and weight the current position against the k and v still in registers.
  if (local < HEAD_DIM) {
    key_cache[(kvHead * MAX_POSITIONS + position) * HEAD_DIM + local] = f16(qk[K_ROW + local]);
    value_cache[(kvHead * MAX_POSITIONS + position) * HEAD_DIM + local] = f16(vcur[local]);
  }

  var best: array<f32, ${group}>;
  for (var g = 0u; g < GROUP; g = g + 1u) { best[g] = -3.0e38; }
  for (var p = local; p < total; p = p + WG) {
    var acc: array<f32, ${group}>;
    for (var g = 0u; g < GROUP; g = g + 1u) { acc[g] = 0.0; }
    let base = (kvHead * MAX_POSITIONS + p) * HEAD_DIM;
    for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
      var kd: f32 = qk[K_ROW + d];
      if (p != position) { kd = f32(key_cache[base + d]); }
      for (var g = 0u; g < GROUP; g = g + 1u) { acc[g] = acc[g] + qk[g * HEAD_DIM + d] * kd; }
    }
    for (var g = 0u; g < GROUP; g = g + 1u) {
      let s = acc[g] * SCALE;
      scores[g * MAX_POSITIONS + p] = s;
      best[g] = max(best[g], s);
    }
  }

  var denom: array<f32, ${group}>;
  for (var g = 0u; g < GROUP; g = g + 1u) {
    reduce[local] = best[g];
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
      if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
      workgroupBarrier();
    }
    let maximum = reduce[0];
    workgroupBarrier();
    var sum: f32 = 0.0;
    for (var p = local; p < total; p = p + WG) {
      let e = exp(scores[g * MAX_POSITIONS + p] - maximum);
      scores[g * MAX_POSITIONS + p] = e;
      sum = sum + e;
    }
    reduce[local] = sum;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
      if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
      workgroupBarrier();
    }
    denom[g] = reduce[0];
    workgroupBarrier();
  }

  // One pass over the value cache for all GROUP heads, weights held in registers, then one fold per
  // head through the shared array in position-group order so the sum is fixed.
  let dim = local % HEAD_DIM;
  let posGroup = local / HEAD_DIM;
  var weighted: array<f32, ${group}>;
  for (var g = 0u; g < GROUP; g = g + 1u) { weighted[g] = 0.0; }
  for (var p = posGroup; p < total; p = p + POS_GROUPS) {
    var vd: f32 = vcur[dim];
    if (p != position) { vd = f32(value_cache[(kvHead * MAX_POSITIONS + p) * HEAD_DIM + dim]); }
    for (var g = 0u; g < GROUP; g = g + 1u) { weighted[g] = weighted[g] + scores[g * MAX_POSITIONS + p] * vd; }
  }
  for (var g = 0u; g < GROUP; g = g + 1u) {
    workgroupBarrier();
    vacc[posGroup * HEAD_DIM + dim] = weighted[g];
    workgroupBarrier();
    if (local < HEAD_DIM) {
      var acc: f32 = 0.0;
      for (var pg = 0u; pg < POS_GROUPS; pg = pg + 1u) { acc = acc + vacc[pg * HEAD_DIM + local]; }
      ${sink}
    }
  }
  workgroupBarrier();
}
`;
}

// A2. The core alone, between two ordinary matvecs. The projection that feeds it is `norm_projection`
// at the wide geometry -- an ordinary split-N matmul over 2048 rows at 512 workgroups, priced on the
// same bandwidth curve as the MLP's -- and the out-projection that follows is a wide split-N
// `matvec_residual` in place. This is the arm the standing correction authorises: one dispatch at
// 45 GB/s is dearer than three at the surface's rate, so the fusion is run backwards.
//
// It reads q, k and v from the f32 scratch the projection wrote rather than from an f16 one, and
// rounds only k and v, exactly where the fused block rounds them. q stays f32 into the score dot,
// which is what the block does too -- the block's q never leaves registers. So the arithmetic is
// unchanged and only the bytes move.
//
// It publishes no partial slice and stages no residual: it is not a fused block, and its output is
// the f16 source the out-projection reads. That is why the out-projection needs no deferred sum --
// a split-N matvec writes whole output rows.
export function attnCoreQkvKernel(
  shape: RuntimeShape,
  offsets: AttnCoreOffsets,
  workgroupSize: number,
): KernelSource {
  const group = checkCoreGeometry("attnCoreQkvKernel", shape, workgroupSize);
  const bindings: BindingSlot[] = [
    { binding: FUSED_BINDING.blob, name: "blob", access: "read" },
    { binding: FUSED_BINDING.state, name: "state", access: "read" },
    { binding: FUSED_BINDING.keyCache, name: "key_cache", access: "read_write" },
    { binding: FUSED_BINDING.valueCache, name: "value_cache", access: "read_write" },
    { binding: FUSED_BINDING.projection, name: "projection", access: "read" },
    { binding: FUSED_BINDING.destination, name: "destination", access: "read_write" },
  ];
  const sink = "destination[(queryIndex * HEADS + kvHead * GROUP + g) * HEAD_DIM + local] = f16(acc / denom[g]);";
  const code = `${HEADER}${STATE_STRUCT}
const HIDDEN: u32 = ${shape.hidden}u;
const HEADS: u32 = ${shape.heads}u;
const WG: u32 = ${workgroupSize}u;
const EPSILON: f32 = ${shape.epsilon};
const PROJ_ROWS: u32 = ${shape.hidden + 2 * shape.kvHeads * shape.headDim}u;
${attnCoreConstants(shape, offsets, workgroupSize)}
${declare(bindings, {})}

var<workgroup> reduce: array<f32, ${workgroupSize}>;
var<workgroup> scores: array<f32, ${group * shape.maxPositions}>;
var<workgroup> qk: array<f32, ${(group + 1) * shape.headDim}>;
var<workgroup> vcur: array<f32, ${shape.headDim}>;
var<workgroup> vacc: array<f32, ${workgroupSize}>;
${attnCoreFn(group, sink)}
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let kvHead = wid.x;
  let queryIndex = wid.y;
  let position = state.position + queryIndex;
  let base = queryIndex * PROJ_ROWS;
  if (local < HEAD_DIM) {
    for (var g = 0u; g < GROUP; g = g + 1u) {
      qk[g * HEAD_DIM + local] = projection[base + (kvHead * GROUP + g) * HEAD_DIM + local];
    }
    qk[K_ROW + local] = projection[base + HIDDEN + kvHead * HEAD_DIM + local];
    vcur[local] = f32(f16(projection[base + HIDDEN + KV_WIDTH + kvHead * HEAD_DIM + local]));
  }
  workgroupBarrier();
  attnCore(local, kvHead, position, queryIndex);
}
`;
  return { code, bindings, workgroupSize };
}

// One workgroup per KV head per query. Every workgroup is a real one, so the kernel needs no bounds
// guard -- which matters, because a guard read from `state` would leave its barriers non-uniform.
export function attnCoreQkvWorkgroups(shape: RuntimeShape, queries: number): Workgroups {
  return [shape.kvHeads, queries, 1];
}

// ---- N2. The split-position (flash-decoding) core ----

// The registered block count: one 64-position block per workgroup, capped at 32. `positions` is the
// largest total the run will reach, because S is fixed for a run -- the command buffer is recorded
// once and replayed, so the workgroup count cannot be re-chosen per token -- while the block LENGTH
// is derived on-device from the position and therefore grows inside the run.
export function flashSplitsFor(positions: number, tile: number, cap: number): number {
  if (!Number.isInteger(positions) || positions < 1) {
    throw new Error(`flashSplitsFor: positions must be a positive integer, got ${positions}`);
  }
  return Math.min(cap, Math.ceil(positions / tile));
}

// One tile of K and V per workgroup, held in f16 exactly as the cache holds them: 64 positions x 64
// dims x 2 B x 2 arrays = 16 384 B whatever the context length is. That is the whole structural
// difference from the core it replaces, which held one f32 score per position -- 20 480 B at
// maxPositions -- and so could not raise its occupancy past eight workgroups.
export const FLASH_TILE = 64;

// 64 accumulator words, the block maximum, the block denominator, and two words of padding that
// take the record to 272 B, a whole number of 16 B rows. Nothing reads the padding.
export const FLASH_RECORD_WORDS = 68;

export function flashRecordBytes(shape: RuntimeShape, splits: number, queries: number): number {
  return queries * splits * shape.heads * FLASH_RECORD_WORDS * 4;
}

function checkFlashGeometry(shape: RuntimeShape, workgroupSize: number, splits: number, maxTotal: number): number {
  const group = checkCoreGeometry("attnCoreFlashKernel", shape, workgroupSize);
  if (workgroupSize !== shape.headDim) {
    throw new Error(`attnCoreFlashKernel: the split-position core is one thread per head dim, so workgroup ${workgroupSize} must be ${shape.headDim}`);
  }
  if (FLASH_TILE !== workgroupSize) {
    throw new Error(`attnCoreFlashKernel: a ${FLASH_TILE}-position tile needs ${FLASH_TILE} threads, the workgroup has ${workgroupSize}`);
  }
  if (!Number.isInteger(splits) || splits < 1) {
    throw new Error(`attnCoreFlashKernel: splits must be a positive integer, got ${splits}`);
  }
  if (!Number.isInteger(maxTotal) || maxTotal < 1) {
    throw new Error(`attnCoreFlashKernel: maxTotal must be a positive integer, got ${maxTotal}`);
  }
  if (maxTotal > shape.maxPositions) {
    throw new Error(`attnCoreFlashKernel: maxTotal ${maxTotal} is past the ${shape.maxPositions}-position cache`);
  }
  return group;
}

// The tile trip count. It is a compile-time constant because the loop it bounds carries
// workgroupBarrier(), and WGSL's uniformity analysis will not accept a bound read from storage --
// `state.position` is uniform in fact but not in the analysis. So the run's LAST total is baked in
// and the tiles a shorter token does not need fall out through a zero-length staging loop, which
// carries no barrier and may therefore be bounded at runtime. Under the registered S rule the
// block length is at most one tile and this is 1.
function flashTileIters(splits: number, maxTotal: number): number {
  return Math.ceil(Math.ceil(maxTotal / splits) / FLASH_TILE);
}

// N2. The attention core split across position blocks. Workgroup (kvHead, s) scores this token's q
// against positions [s*len, min((s+1)*len, total)) of one KV head, where len = ceil(total/S), and
// publishes that block's unnormalised weighted-V sum with the maximum and denominator that scale
// it. Every cached k and v is read by exactly one workgroup, so the split costs no cache bytes; it
// costs one q read per block instead of one per KV head, and the merge that follows.
//
// The block holding the current position writes this token's k and v into the cache and then reads
// them back with the rest of its tile. That is safe without any barrier and without a special case:
// the blocks partition the positions, so no other workgroup reads that slot, and inside one
// workgroup thread `local` writes dim `local` and is the only thread that reads it back.
//
// The scores are bit-identical to the core this replaces -- same q, same k after the shared
// `headRows`, same ascending-d dot -- and the tile maximum, the rescale and the merge are the whole
// of the new numerics, which is what the registered core comparison measures.
export function attnCoreFlashKernel(
  shape: RuntimeShape,
  offsets: AttnCoreOffsets,
  workgroupSize: number,
  splits: number,
  maxTotal: number,
): KernelSource {
  const group = checkFlashGeometry(shape, workgroupSize, splits, maxTotal);
  const bindings: BindingSlot[] = [
    { binding: FUSED_BINDING.blob, name: "blob", access: "read" },
    { binding: FUSED_BINDING.state, name: "state", access: "read" },
    { binding: FUSED_BINDING.keyCache, name: "key_cache", access: "read_write" },
    { binding: FUSED_BINDING.valueCache, name: "value_cache", access: "read_write" },
    { binding: FUSED_BINDING.projection, name: "projection", access: "read" },
    { binding: FUSED_BINDING.flash, name: "flash", access: "read_write" },
  ];
  const code = `${HEADER}${STATE_STRUCT}
const HIDDEN: u32 = ${shape.hidden}u;
const HEADS: u32 = ${shape.heads}u;
const WG: u32 = ${workgroupSize}u;
const EPSILON: f32 = ${shape.epsilon};
const PROJ_ROWS: u32 = ${shape.hidden + 2 * shape.kvHeads * shape.headDim}u;
const SPLITS: u32 = ${splits}u;
const TILE: u32 = ${FLASH_TILE}u;
const TILE_ITERS: u32 = ${flashTileIters(splits, maxTotal)}u;
const RECORD: u32 = ${FLASH_RECORD_WORDS}u;
const NEG_INF: f32 = -3.0e38;
${attnCoreConstants(shape, offsets, workgroupSize)}
${declare(bindings, {})}

var<workgroup> ktile: array<f16, ${FLASH_TILE * shape.headDim}>;
var<workgroup> vtile: array<f16, ${FLASH_TILE * shape.headDim}>;
var<workgroup> qk: array<f32, ${(group + 1) * shape.headDim}>;
var<workgroup> vcur: array<f32, ${shape.headDim}>;
var<workgroup> reduce: array<f32, ${workgroupSize}>;
var<workgroup> sc: array<f32, ${group * FLASH_TILE}>;
${headRowsFn()}
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let kvHead = wid.x / SPLITS;
  let s = wid.x % SPLITS;
  let queryIndex = wid.y;
  let position = state.position + queryIndex;
  let total = position + 1u;
  let blockLen = (total + SPLITS - 1u) / SPLITS;
  let start = s * blockLen;
  let stop = min(start + blockLen, total);
  // The block that holds this token's own position. Under the registered S rule it is the last
  // one, but a short token under a large S leaves trailing blocks empty and it is not, so it is
  // computed rather than assumed.
  let owner = (total - 1u) / blockLen;

  // Every block needs q. Only the block that owns the current position needs this token's k and v,
  // and reading them in every block would multiply the projection scratch by S. The rows the other
  // blocks skip are written to zero rather than left to the implementation: they still take the
  // norm and the rotation, because dropping them would put a barrier under a non-uniform branch,
  // and nothing reads the result.
  let base = queryIndex * PROJ_ROWS;
  for (var g = 0u; g < GROUP; g = g + 1u) {
    qk[g * HEAD_DIM + local] = projection[base + (kvHead * GROUP + g) * HEAD_DIM + local];
  }
  if (s == owner) {
    qk[K_ROW + local] = projection[base + HIDDEN + kvHead * HEAD_DIM + local];
    vcur[local] = f32(f16(projection[base + HIDDEN + KV_WIDTH + kvHead * HEAD_DIM + local]));
  } else {
    qk[K_ROW + local] = 0.0;
    vcur[local] = 0.0;
  }
  workgroupBarrier();
  headRows(local, position);
  if (s == owner) {
    key_cache[(kvHead * MAX_POSITIONS + position) * HEAD_DIM + local] = f16(qk[K_ROW + local]);
    value_cache[(kvHead * MAX_POSITIONS + position) * HEAD_DIM + local] = f16(vcur[local]);
  }

  // The running state of the online softmax. Every one of these is computed from workgroup-uniform
  // data and is identical in every lane, so it lives in registers and costs no workgroup memory --
  // except vacc, which is per dim, and thread local owns dim local in every tile.
  var m: array<f32, ${group}>;
  var l: array<f32, ${group}>;
  var vacc: array<f32, ${group}>;
  var corr: array<f32, ${group}>;
  var peak: array<f32, ${group}>;
  for (var g = 0u; g < GROUP; g = g + 1u) { m[g] = NEG_INF; l[g] = 0.0; vacc[g] = 0.0; }

  for (var t = 0u; t < TILE_ITERS; t = t + 1u) {
    let tileStart = start + t * TILE;
    // TILE_ITERS is baked from the deepest total the run reaches, because one encoded command
    // buffer serves every token of a run that sweeps depth. At shallower positions the trailing
    // tiles are empty, and an empty tile must cost a branch rather than two full workgroup
    // reductions per q head -- otherwise the core's price would be set by the end of the run
    // instead of by the position it is at. The test is workgroup-uniform: start and stop come from
    // workgroup_id and a read-only storage load, and t is a uniform loop counter, so every barrier
    // below stays in uniform control flow.
    if (tileStart >= stop) { continue; }
    let len = min(TILE, stop - tileStart);

    // No barrier in this loop, so a runtime bound is legal.
    for (var p = 0u; p < len; p = p + 1u) {
      let row = (kvHead * MAX_POSITIONS + tileStart + p) * HEAD_DIM + local;
      ktile[p * HEAD_DIM + local] = key_cache[row];
      vtile[p * HEAD_DIM + local] = value_cache[row];
    }
    workgroupBarrier();

    // Thread local scores position local of the tile against all GROUP q heads in one pass
    // over its staged k row, in ascending d -- the same term order as the core this replaces.
    var sv: array<f32, ${group}>;
    for (var g = 0u; g < GROUP; g = g + 1u) { sv[g] = NEG_INF; }
    if (local < len) {
      var a: array<f32, ${group}>;
      for (var g = 0u; g < GROUP; g = g + 1u) { a[g] = 0.0; }
      for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
        let kd = f32(ktile[local * HEAD_DIM + d]);
        for (var g = 0u; g < GROUP; g = g + 1u) { a[g] = a[g] + qk[g * HEAD_DIM + d] * kd; }
      }
      for (var g = 0u; g < GROUP; g = g + 1u) { sv[g] = a[g] * SCALE; }
    }

    for (var g = 0u; g < GROUP; g = g + 1u) {
      reduce[local] = sv[g];
      workgroupBarrier();
      for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
        workgroupBarrier();
      }
      peak[g] = reduce[0];
      workgroupBarrier();
    }

    // The online update. An empty tile leaves peak at NEG_INF, so corr is exp(0) = 1 and the
    // denominator gains nothing; an empty BLOCK never leaves NEG_INF, and the merge prices its
    // weight at exp(NEG_INF - M) = 0 against any real maximum.
    for (var g = 0u; g < GROUP; g = g + 1u) {
      let next = max(m[g], peak[g]);
      corr[g] = exp(m[g] - next);
      var e: f32 = 0.0;
      if (local < len) { e = exp(sv[g] - next); }
      sc[g * TILE + local] = e;
      reduce[local] = e;
      workgroupBarrier();
      for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
        workgroupBarrier();
      }
      l[g] = l[g] * corr[g] + reduce[0];
      m[g] = next;
      workgroupBarrier();
    }

    // Thread local owns dim local and walks the tile's positions in index order, so the sum is
    // fixed and the staged V row it reads is contiguous across the workgroup.
    var w: array<f32, ${group}>;
    for (var g = 0u; g < GROUP; g = g + 1u) { w[g] = 0.0; }
    for (var p = 0u; p < len; p = p + 1u) {
      let vd = f32(vtile[p * HEAD_DIM + local]);
      for (var g = 0u; g < GROUP; g = g + 1u) { w[g] = w[g] + sc[g * TILE + p] * vd; }
    }
    for (var g = 0u; g < GROUP; g = g + 1u) { vacc[g] = vacc[g] * corr[g] + w[g]; }
    workgroupBarrier();
  }

  for (var g = 0u; g < GROUP; g = g + 1u) {
    let rec = ((queryIndex * SPLITS + s) * HEADS + kvHead * GROUP + g) * RECORD;
    flash[rec + local] = vacc[g];
    if (local == 0u) {
      flash[rec + HEAD_DIM] = m[g];
      flash[rec + HEAD_DIM + 1u] = l[g];
    }
  }
}
`;
  return { code, bindings, workgroupSize };
}

export function attnCoreFlashWorkgroups(shape: RuntimeShape, splits: number, queries: number): Workgroups {
  return [shape.kvHeads * splits, queries, 1];
}

// The log-sum-exp merge: one workgroup per q head, one thread per dim, blocks visited in ascending
// index so the sum is bitwise reproducible. It is emitted at every S, including S = 1, where it
// reduces to acc / l and the dispatch list therefore does not change with the block count.
export function attnMergeKernel(shape: RuntimeShape, workgroupSize: number, splits: number): KernelSource {
  if (workgroupSize !== shape.headDim) {
    throw new Error(`attnMergeKernel: one thread per head dim, so workgroup ${workgroupSize} must be ${shape.headDim}`);
  }
  if (!Number.isInteger(splits) || splits < 1) {
    throw new Error(`attnMergeKernel: splits must be a positive integer, got ${splits}`);
  }
  const bindings: BindingSlot[] = [
    { binding: FUSED_BINDING.destination, name: "destination", access: "read_write" },
    { binding: FUSED_BINDING.flash, name: "flash", access: "read" },
  ];
  const code = `${HEADER}
const HEADS: u32 = ${shape.heads}u;
const HEAD_DIM: u32 = ${shape.headDim}u;
const WG: u32 = ${workgroupSize}u;
const SPLITS: u32 = ${splits}u;
const RECORD: u32 = ${FLASH_RECORD_WORDS}u;
const NEG_INF: f32 = -3.0e38;

${declare(bindings, {})}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let head = wid.x;
  let queryIndex = wid.y;
  let first = (queryIndex * SPLITS * HEADS + head) * RECORD;
  var peak: f32 = NEG_INF;
  for (var s = 0u; s < SPLITS; s = s + 1u) {
    peak = max(peak, flash[first + s * HEADS * RECORD + HEAD_DIM]);
  }
  var num: f32 = 0.0;
  var den: f32 = 0.0;
  for (var s = 0u; s < SPLITS; s = s + 1u) {
    let rec = first + s * HEADS * RECORD;
    let weight = exp(flash[rec + HEAD_DIM] - peak);
    num = num + weight * flash[rec + local];
    den = den + weight * flash[rec + HEAD_DIM + 1u];
  }
  destination[(queryIndex * HEADS + head) * HEAD_DIM + local] = f16(num / den);
}
`;
  return { code, bindings, workgroupSize };
}

export function attnMergeWorkgroups(shape: RuntimeShape, queries: number): Workgroups {
  return [shape.heads, queries, 1];
}

// The fused MLP block, carried unchanged from B1.2. One workgroup owns ffn/slices rows of the
// SwiGLU intermediate and the matching K-slice of the down projection, so gate, up, the activation
// and that slice's whole contribution to the output are one dispatch.
//
// The slice count is chosen twice over: 2560/80 = 32 intermediate rows is exactly one quantisation
// block of the down projection's K, so a workgroup's slice is a whole number of blocks and no
// scale is shared across workgroups; and 80 workgroups of 256 threads is two per core on forty
// cores, which is what lifts the down projection off R1's 56-70 GB/s.
export function mlpFusedKernel(
  shape: RuntimeShape,
  offsets: MlpBlockOffsets,
  geometry: FusedGeometry,
): KernelSource {
  checkGeometry("mlpFusedKernel", shape, geometry, shape.ffn);
  const { workgroupSize, subgroupSize, slices } = geometry;
  const rowsPerSlice = shape.ffn / slices;
  if (rowsPerSlice % shape.quantBlock !== 0) {
    throw new Error(`mlpFusedKernel: ${rowsPerSlice} rows per slice is not a whole number of ${shape.quantBlock}-wide blocks`);
  }
  const bindings = blockBindings(geometry);
  const code = `${HEADER}${STATE_STRUCT}${DEQUANT}
${commonConstants(shape, geometry)}const FFN: u32 = ${shape.ffn}u;
const ROWS_PER_SLICE: u32 = ${rowsPerSlice}u;
const SLICE_ITERS: u32 = ${rowsPerSlice / (workgroupSize / subgroupSize)}u;

${declare(bindings, {})}

var<workgroup> stage: array<f32, ${shape.hidden}>;
var<workgroup> reduce: array<f32, ${workgroupSize}>;
var<workgroup> hidden_slice: array<f32, ${rowsPerSlice}>;
${stageBlockFn(shape, offsets.norm, geometry.foldSlices)}${rowSumFn(offsets.gateUpQuant, offsets.gateUpScales, shape.hidden / shape.quantBlock, subgroupSize)}${projectFn(offsets.downKQuant, offsets.downKScales, rowsPerSlice / shape.quantBlock, "hidden_slice")}
@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) local: u32,
  @builtin(subgroup_invocation_id) lane: u32,
) {
  let group = wid.x;
  stageBlock(local, group);
  let sub = local / SG;
  let rowBase = group * ROWS_PER_SLICE;
  for (var i = 0u; i < SLICE_ITERS; i = i + 1u) {
    let r = sub + i * ROWS_AT_ONCE;
    let gate = rowSum(rowBase + r, lane);
    let up = rowSum(FFN + rowBase + r, lane);
    if (lane == 0u) { hidden_slice[r] = f32(f16((gate / (1.0 + exp(-gate))) * up)); }
  }
  workgroupBarrier();
  project(local, group);
}
`;
  return { code, bindings, workgroupSize };
}

// The reduction that clears pending partials when the next dispatch is not a fused block: before an
// unfused norm matvec in a mixed arm, and before the head, which stages the residual across 4096
// workgroups and must not fold. It is the one dispatch allowed to work in place -- it needs no norm,
// so each workgroup reads and writes only the rows it owns and nothing another workgroup writes.
// That is why the residual ping-pong flips only on fused blocks, whose count is even in every arm.
//
// Rows are split across workgroups and the slice sum is split across `threadsPerRow` lanes of each
// row, so every lane carries work at any slice count and the lanes of one row read consecutive rows
// of consecutive slices. The per-row combine runs in lane order, so the sum is fixed.
export function foldKernel(
  shape: RuntimeShape,
  workgroupSize: number,
  threadsPerRow: number,
  foldSlices: number,
): KernelSource {
  if (workgroupSize % threadsPerRow !== 0) {
    throw new Error(`foldKernel: workgroup ${workgroupSize} is not a multiple of ${threadsPerRow} threads per row`);
  }
  if (foldSlices < 1) {
    throw new Error(`foldKernel: nothing to fold, got ${foldSlices} slices`);
  }
  const rowsPerWorkgroup = workgroupSize / threadsPerRow;
  const bindings: BindingSlot[] = [
    { binding: FUSED_BINDING.residualOut, name: "residual_out", access: "read_write" },
    { binding: FUSED_BINDING.partialsIn, name: "partials_in", access: "read" },
  ];
  const code = `${HEADER}
const HIDDEN: u32 = ${shape.hidden}u;
const WG: u32 = ${workgroupSize}u;
const LANES: u32 = ${threadsPerRow}u;
const ROWS_PER_WG: u32 = ${rowsPerWorkgroup}u;
const FOLD_SLICES: u32 = ${foldSlices}u;

${declare(bindings, {})}

var<workgroup> part: array<f32, ${workgroupSize}>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let slot = local % ROWS_PER_WG;
  let lane = local / ROWS_PER_WG;
  let row = wid.x * ROWS_PER_WG + slot;
  var acc: f32 = 0.0;
  if (row < HIDDEN) {
    for (var s = lane; s < FOLD_SLICES; s = s + LANES) { acc = acc + partials_in[s * HIDDEN + row]; }
  }
  part[local] = acc;
  workgroupBarrier();
  if (lane == 0u && row < HIDDEN) {
    var sum: f32 = residual_out[row];
    for (var l = 0u; l < LANES; l = l + 1u) { sum = sum + part[l * ROWS_PER_WG + slot]; }
    residual_out[row] = sum;
  }
}
`;
  return { code, bindings, workgroupSize };
}

export function fusedBlockWorkgroups(slices: number): Workgroups {
  return [slices, 1, 1];
}

export function foldWorkgroups(shape: RuntimeShape, workgroupSize: number, threadsPerRow: number): Workgroups {
  return [Math.ceil(shape.hidden / (workgroupSize / threadsPerRow)), 1, 1];
}
