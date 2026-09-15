// The unfused kernels: one WGSL module per dispatch that binds its matrices as separate buffers.
// Every module bakes its shapes in as compile-time constants so no dispatch reads a uniform, and
// the only mutable scalar state (position, step, rng) lives in one device-side buffer that the
// sampler advances, which is what lets a whole token be encoded without the host touching a buffer.
//
// Semantics are taken from the exported graph, node for node:
//   SimplifiedLayerNormalization  x / sqrt(mean(x^2) + eps) * gamma
//   MatMulNBits                   y[n] = sum_k (nibble(n,k) - zp(n,k/32)) * scale(n,k/32) * x[k]
//   conv block                    B,C,x = split(in_proj(h)); y = C * causal_conv(B*x)
//   attention                     per-head q_norm/k_norm, non-interleaved RoPE, causal GQA
//   final norm                    SkipSimplifiedLayerNormalization(x, x) = RMSNorm(2x) * gamma
//
// The matvec is the occupancy lever's whole surface. One workgroup owns
// (workgroupSize/subgroupSize) * rowsPerSubgroup output rows and stages the input once; dropping
// rowsPerSubgroup from four to one quadruples the workgroup count over the same matrix without
// changing a byte of what is read. C1 and M1 are exactly that substitution.
//
// Subgroup ops appear only in workgroup-uniform control flow: every loop that encloses one has a
// compile-time trip count, and the per-lane tail is masked with an `if` that contains no subgroup
// op. Dispatches are two-dimensional (x = tile, y = query); this unit decodes one query at a time,
// so y is always 1 and the y guard against state.queries is what keeps that honest.

// `norm_projection` is `norm_matvec` with an f32 sink instead of an f16 one. It exists because
// A2 lifts the q/k/v projection out of the attention block into an ordinary wide split-N matvec,
// and its core reads q, k and v before rounding them: the block computes them in f32 registers
// and rounds only k and v on the way into the cache, so a f16 scratch here would round q too and
// change the arm's arithmetic as well as its bandwidth.
//
// The mode IS the kernel class P0a measures a curve for, so the name comes from the plan rather
// than being re-declared here: a class the plan can name and the kernel cannot emit, or the
// reverse, would put a hole in the pricing table.
import {
  MATVEC_CLASSES,
  cellReachable,
  rowsPerWorkgroup as planRowsPerWorkgroup,
  type MatvecClass,
  type MatvecGeometry as PlanMatvecGeometry,
} from "../pure/plan.ts";

export type MatvecMode = MatvecClass;
export const MATVEC_MODES: readonly MatvecMode[] = MATVEC_CLASSES;

// The three dequantisation variants P0b measures against the carried kernel, plus the carried
// kernel itself. They differ only inside the inner product -- how the packed nibbles are loaded,
// what precision the eight-term dot runs in, and how many independent accumulators the block loop
// keeps -- so a variant is a kernel-internal substitution, not a lever: whichever P0b adopts is
// used by EVERY arm including the baseline.
//
//   carried              four u32 loads per block, f32 unpack and dot, one f32 accumulator.
//   v1_vec4_f16          one vec4<u32> load per block, f16 unpack and dot, one f32 accumulator.
//   v2_vec4_accumulator  carried loads and f32 dot, four independent f32 accumulators closed by
//                        one subgroupAdd on the vec4.
//   v3_both              the vec4<u32> f16 load and dot with the four-lane accumulator.
//
// None of them changes the staged input: staging is f32 in every variant, so the only rounding a
// variant introduces is inside the eight-term dot itself and the correctness comparison measures
// exactly that.
export const DEQUANT_VARIANTS = ["carried", "v1_vec4_f16", "v2_vec4_accumulator", "v3_both"] as const;
export type DequantVariant = (typeof DEQUANT_VARIANTS)[number];

function packedVec4(variant: DequantVariant): boolean {
  return variant === "v1_vec4_f16" || variant === "v3_both";
}

function vec4Accumulator(variant: DequantVariant): boolean {
  return variant === "v2_vec4_accumulator" || variant === "v3_both";
}

export type BindingName =
  | "weights"
  | "scales"
  | "zero_points"
  | "gamma"
  | "residual"
  | "source"
  | "destination"
  | "state"
  | "tokens"
  | "cos"
  | "sin"
  | "key_cache"
  | "value_cache"
  | "cache"
  | "logits"
  | "partials"
  | "sampled"
  // A2's f32 q/k/v scratch: written by the projection matvec, read by the core.
  | "projection"
  // N2's per-position-block partials: one 68-word record per (query, block, q head) holding the
  // block's unnormalised weighted-V sum, its running maximum and its softmax denominator. Written
  // by the split-position core, read by the merge that closes the log-sum-exp.
  | "flash"
  // The fused blocks (src/page/fused_wgsl.ts) bind the whole weight blob once instead of one
  // buffer per matrix, and read and write the residual and the deferred partials on opposite
  // parities of a ping-pong pair.
  | "blob"
  | "residual_in"
  | "residual_out"
  | "partials_in"
  | "partials_out";

export interface BindingSlot {
  readonly binding: number;
  readonly name: BindingName;
  readonly access: "read" | "read_write";
}

export interface KernelSource {
  readonly code: string;
  readonly bindings: readonly BindingSlot[];
  readonly workgroupSize: number;
}

export type Workgroups = readonly [number, number, number];

export interface RuntimeShape {
  readonly hidden: number;
  readonly ffn: number;
  readonly vocab: number;
  readonly heads: number;
  readonly kvHeads: number;
  readonly headDim: number;
  readonly convTaps: number;
  readonly quantBlock: number;
  readonly epsilon: number;
  readonly maxPositions: number;
  readonly maxQueries: number;
}

// The plan's geometry plus the one bound that belongs to the kernel rather than to the plan: the
// staged input lives in workgroup memory, so the widest matrix a module can contain is bounded by
// maxComputeWorkgroupStorageSize.
export interface MatvecGeometry extends PlanMatvecGeometry {
  readonly maxCols: number;
}

export function withMaxCols(geometry: PlanMatvecGeometry, maxCols: number): MatvecGeometry {
  return {
    workgroupSize: geometry.workgroupSize,
    subgroupSize: geometry.subgroupSize,
    rowsPerSubgroup: geometry.rowsPerSubgroup,
    subgroupsPerRow: geometry.subgroupsPerRow,
    maxCols,
  };
}

export const HEADER = "enable f16;\nenable subgroups;\n";

// The device-side scalar state, carried unchanged. S -- the split-position core's block count --
// is deliberately NOT here: one command buffer is recorded per run and replayed, so the core's
// workgroup count is baked at record time and S is a compile-time constant of the kernel. Only the
// block LENGTH varies inside a run, and that is derived on-device from `position`.
export const STATE_STRUCT = `
struct State {
  position: u32,
  step: u32,
  queries: u32,
  sample_mode: u32,
  rng: u32,
  freerun: u32,
  advance: u32,
  reserved: u32,
}
`;

const BINDING_TYPES: Readonly<Record<BindingName, string>> = {
  weights: "array<u32>",
  scales: "array<f32>",
  zero_points: "array<u32>",
  gamma: "array<f32>",
  residual: "array<f32>",
  source: "array<f16>",
  destination: "array<f16>",
  state: "State",
  tokens: "array<u32>",
  cos: "array<f32>",
  sin: "array<f32>",
  key_cache: "array<f16>",
  value_cache: "array<f16>",
  cache: "array<f16>",
  logits: "array<f32>",
  partials: "array<u32>",
  sampled: "array<u32>",
  blob: "array<u32>",
  residual_in: "array<f32>",
  residual_out: "array<f32>",
  partials_in: "array<f32>",
  partials_out: "array<f32>",
  projection: "array<f32>",
  flash: "array<f32>",
};

// The registered type of a binding, unless the kernel names another one for it. The only override
// in this unit is the packed-vec4 weight view P0b's V1 and V3 read, which is the same bytes at the
// same offsets seen as 16 B elements rather than 4 B ones.
export type BindingTypes = Readonly<Partial<Record<BindingName, string>>>;

export function declare(slots: readonly BindingSlot[], overrides: BindingTypes): string {
  return slots
    .map((s) => {
      const override = overrides[s.name];
      const type = override === undefined ? BINDING_TYPES[s.name] : override;
      return `@group(0) @binding(${s.binding}) var<storage, ${s.access}> ${s.name}: ${type};`;
    })
    .join("\n");
}

// Unpacks the eight 4-bit weights held in one u32 and dots them with eight staged inputs.
export const DEQUANT = `
fn dot8(packed: u32, zp: f32, a0: vec4<f32>, a1: vec4<f32>) -> f32 {
  let lo = vec4<f32>(f32(packed & 15u), f32((packed >> 4u) & 15u), f32((packed >> 8u) & 15u), f32((packed >> 12u) & 15u));
  let hi = vec4<f32>(f32((packed >> 16u) & 15u), f32((packed >> 20u) & 15u), f32((packed >> 24u) & 15u), f32((packed >> 28u) & 15u));
  let z = vec4<f32>(zp, zp, zp, zp);
  return dot(lo - z, a0) + dot(hi - z, a1);
}
`;

// The same eight nibbles in f16. Each is a small integer in [0, 15] and the zero point is one of
// them, so the difference is exact in f16; what the variant trades is the precision of the
// eight-term sum, and the correctness comparison against the carried kernel measures exactly that.
export const DEQUANT_F16 = `
fn dot8h(packed: u32, zp: f16, a0: vec4<f16>, a1: vec4<f16>) -> f16 {
  let lo = vec4<f16>(f16(packed & 15u), f16((packed >> 4u) & 15u), f16((packed >> 8u) & 15u), f16((packed >> 12u) & 15u));
  let hi = vec4<f16>(f16((packed >> 16u) & 15u), f16((packed >> 20u) & 15u), f16((packed >> 24u) & 15u), f16((packed >> 28u) & 15u));
  let z = vec4<f16>(zp, zp, zp, zp);
  return dot(lo - z, a0) + dot(hi - z, a1);
}
`;

// Counter-based PRNG for the Gumbel-max draw. Indexed by vocabulary id, so the draw a token gets
// does not depend on which dispatch or which lane happens to score it: the folded head scan and
// the standalone partial scan sample the same token from the same logits and the same rng.
export const GUMBEL = `
fn hash(seed: u32, index: u32) -> u32 {
  var x = seed ^ (index * 2654435761u);
  x = x ^ (x >> 16u);
  x = x * 2246822519u;
  x = x ^ (x >> 13u);
  x = x * 3266489917u;
  return x ^ (x >> 16u);
}

fn gumbel(seed: u32, index: u32) -> f32 {
  let u = (f32(hash(seed, index) >> 8u) + 0.5) * ${1 / 16777216};
  return -log(-log(u));
}
`;

function zeroPointFn(zeroPoints: boolean, blocks: number): string {
  if (!zeroPoints) {
    return "fn zeroPoint(row: u32, block: u32) -> f32 { return 8.0; }\n";
  }
  return `
const ZP_BYTES_PER_ROW: u32 = ${Math.ceil(blocks / 2)}u;
fn zeroPoint(row: u32, block: u32) -> f32 {
  let index = row * ZP_BYTES_PER_ROW + block / 2u;
  let byte = (zero_points[index / 4u] >> ((index % 4u) * 8u)) & 255u;
  if ((block & 1u) == 0u) { return f32(byte & 15u); }
  return f32(byte >> 4u);
}
`;
}

// Stages the residual for one query and RMSNorms it in place with the operator's gamma. This is
// what makes (norm -> matvec) one dispatch: the norm costs one pass over 4 KB of cached residual
// per workgroup instead of a second dispatch and a round trip through memory.
function stageResidual(preScale: number): string {
  return `
fn stageInput(query: u32, local: u32) {
  var sum: f32 = 0.0;
  for (var i = local; i < COLS; i = i + WG) {
    let v = residual[query * COLS + i] * ${preScale.toFixed(1)};
    stage[i] = v;
    sum = sum + v * v;
  }
  reduce[local] = sum;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(reduce[0] / f32(COLS) + EPSILON);
  workgroupBarrier();
  for (var i = local; i < COLS; i = i + WG) {
    stage[i] = stage[i] * inv * gamma[i];
  }
  workgroupBarrier();
}
`;
}

const STAGE_SOURCE = `
fn stageInput(query: u32, local: u32) {
  for (var i = local; i < COLS; i = i + WG) {
    stage[i] = f32(source[query * COLS + i]);
  }
  workgroupBarrier();
}
`;

// One row of a quantised matrix against the staged input vector.
//
// SUBGROUPS_PER_ROW subgroups cooperate on one row. Within a row the block index is
//
//   block = i * (SG * SUBGROUPS_PER_ROW) + part * SG + lane
//
// so each lane's 16 B load is contiguous, each subgroup spans a contiguous run of blocks, and the
// SUBGROUPS_PER_ROW runs tile the row. The trip count is a compile-time constant, which keeps the
// closing subgroupAdd uniform. At SUBGROUPS_PER_ROW = 1 the emitted source is B1.3's, character for
// character, so the baseline arm runs the kernel B1.3 measured.
function blockExpr(subgroupsPerRow: number): string {
  return subgroupsPerRow === 1 ? "lane + i * SG" : "slice * SG + lane + i * (SG * SUBGROUPS_PER_ROW)";
}

function stageAt(offset: number): string {
  return offset === 0 ? "stage[o]" : `stage[o + ${offset}u]`;
}

function stageQuad(offset: number): string {
  return `vec4<f32>(${[0, 1, 2, 3].map((k) => stageAt(offset + k)).join(", ")})`;
}

function stageQuadHalf(offset: number): string {
  return `vec4<f16>(${[0, 1, 2, 3].map((k) => `f16(${stageAt(offset + k)})`).join(", ")})`;
}

// The four eight-term dots of one 32-wide quantisation block, as expressions. `word` selects which
// of the block's four packed u32 the dot consumes; the packed-vec4 variants read all four out of
// one 16 B load, the carried form reads four 4 B ones.
function blockDot(variant: DequantVariant, word: number): string {
  const lo = word * 8;
  const hi = word * 8 + 4;
  if (packedVec4(variant)) {
    const lane = ["x", "y", "z", "w"][word];
    return `dot8h(packed.${lane}, zpv,\n        ${stageQuadHalf(lo)},\n        ${stageQuadHalf(hi)})`;
  }
  const source = word === 0 ? "weights[base]" : `weights[base + ${word}u]`;
  return `dot8(${source}, zpv,\n        ${stageQuad(lo)},\n        ${stageQuad(hi)})`;
}

function rowSumSource(variant: DequantVariant, subgroupsPerRow: number): string {
  const signature = subgroupsPerRow === 1 ? "row: u32, lane: u32" : "row: u32, slice: u32, lane: u32";
  const load = packedVec4(variant)
    ? `      let packed = weights[row * BLOCKS + block];
      let zpv = f16(zeroPoint(row, block));
      let o = block * 32u;`
    : `      let base = row * BLOCKS * 4u + block * 4u;
      let zpv = zeroPoint(row, block);
      let o = block * 32u;`;
  const dots = [0, 1, 2, 3].map((word) => blockDot(variant, word));
  let declaration: string;
  let accumulate: string;
  let close: string;
  if (vec4Accumulator(variant)) {
    declaration = "  var acc: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);";
    accumulate = `      let d = vec4<f32>(
        ${dots.map((dot) => (packedVec4(variant) ? `f32(${dot})` : dot)).join(",\n        ")});
      acc = acc + d * scales[row * BLOCKS + block];`;
    close = `  let total = subgroupAdd(acc);
  return total.x + total.y + total.z + total.w;`;
  } else {
    declaration = "  var acc: f32 = 0.0;";
    accumulate = `      var part: ${packedVec4(variant) ? "f16 = f16(0.0)" : "f32 = 0.0"};
${dots.map((dot) => `      part = part + ${dot};`).join("\n")}
      acc = acc + ${packedVec4(variant) ? "f32(part)" : "part"} * scales[row * BLOCKS + block];`;
    close = "  return subgroupAdd(acc);";
  }
  const rowSum = `
fn rowSum(${signature}) -> f32 {
${declaration}
  for (var i = 0u; i < BLOCK_ITERS; i = i + 1u) {
    let block = ${blockExpr(subgroupsPerRow)};
    if (block < BLOCKS) {
${load}
${accumulate}
    }
  }
${close}
}
`;
  if (subgroupsPerRow === 1) {
    return rowSum;
  }
  // The closing cross-subgroup reduction. Every cooperating subgroup publishes its partial, and
  // every one of them re-reads the whole group in ASCENDING slice order, so the sum a row gets does
  // not depend on which subgroup runs first. The trailing barrier keeps the next row's writes from
  // overtaking this row's reads; both barriers sit in workgroup-uniform control flow because the
  // row loop's trip count is a compile-time constant and its tail is clamped, not skipped.
  return `${rowSum}
fn rowTotal(row: u32, subgroup: u32, slice: u32, lane: u32) -> f32 {
  let mine = rowSum(row, slice, lane);
  if (lane == 0u) { cross[subgroup] = mine; }
  workgroupBarrier();
  let base = subgroup - slice;
  var total: f32 = 0.0;
  for (var p = 0u; p < SUBGROUPS_PER_ROW; p = p + 1u) {
    total = total + cross[base + p];
  }
  workgroupBarrier();
  return total;
}
`;
}

function tilesFor(mode: MatvecMode, rows: number, shape: RuntimeShape, geometry: MatvecGeometry): number {
  const emits = mode === "norm_swiglu" ? shape.ffn : rows;
  return Math.ceil(emits / planRowsPerWorkgroup(geometry));
}

export function matvecWorkgroups(
  mode: MatvecMode,
  rows: number,
  shape: RuntimeShape,
  geometry: MatvecGeometry,
  queries: number,
): Workgroups {
  return [tilesFor(mode, rows, shape, geometry), queries, 1];
}

// One int4 matrix-vector kernel for every mode the plan names. The mode decides three things and
// nothing else: which buffer the staged input comes from, which buffer the result lands in, and
// whether the row loop emits one value or folds a gate and an up projection into one. The variant
// decides nothing outside the inner product.
export function matvecKernel(
  mode: MatvecMode,
  rows: number,
  cols: number,
  zeroPoints: boolean,
  geometry: MatvecGeometry,
  shape: RuntimeShape,
  preScale: number,
  variant: DequantVariant,
): KernelSource {
  const { workgroupSize, subgroupSize, rowsPerSubgroup, subgroupsPerRow } = geometry;
  if (workgroupSize % subgroupSize !== 0) {
    throw new Error(`matvecKernel: workgroup ${workgroupSize} is not a multiple of subgroup ${subgroupSize}`);
  }
  if (cols % shape.quantBlock !== 0) {
    throw new Error(`matvecKernel: cols ${cols} is not a multiple of block ${shape.quantBlock}`);
  }
  if (cols > geometry.maxCols) {
    throw new Error(`matvecKernel: cols ${cols} exceeds the registered stage width ${geometry.maxCols}`);
  }
  if (mode === "norm_swiglu" && rows !== 2 * shape.ffn) {
    throw new Error(`matvecKernel: swiglu expects ${2 * shape.ffn} rows, got ${rows}`);
  }
  if (mode === "norm_projection" && rows % shape.headDim !== 0) {
    throw new Error(`matvecKernel: a q/k/v projection emits whole heads, ${rows} rows is not a multiple of ${shape.headDim}`);
  }
  if (!cellReachable(geometry, cols, shape.quantBlock)) {
    throw new Error(
      `matvecKernel: ${subgroupsPerRow} subgroups of ${subgroupSize} lanes cannot each take a block of a ` +
        `${cols}-wide row`,
    );
  }
  const rowsPerWorkgroup = planRowsPerWorkgroup(geometry);
  const subgroups = workgroupSize / subgroupSize;
  const blocks = cols / shape.quantBlock;
  const emits = mode === "norm_swiglu" ? shape.ffn : rows;
  const cooperative = subgroupsPerRow > 1;
  // Cooperating subgroups own one row between them, so a workgroup covers exactly its row groups.
  if (cooperative && rowsPerSubgroup !== 1) {
    throw new Error(`matvecKernel: cooperating subgroups own one row each, got ${rowsPerSubgroup} per subgroup`);
  }

  const bindings: BindingSlot[] = [
    { binding: 0, name: "weights", access: "read" },
    { binding: 1, name: "scales", access: "read" },
  ];
  if (zeroPoints) {
    bindings.push({ binding: 2, name: "zero_points", access: "read" });
  }
  if (mode === "matvec_residual") {
    bindings.push({ binding: 4, name: "residual", access: "read_write" });
    bindings.push({ binding: 5, name: "source", access: "read" });
  } else {
    bindings.push({ binding: 3, name: "gamma", access: "read" });
    bindings.push({ binding: 4, name: "residual", access: "read" });
  }
  if (mode === "norm_head") {
    bindings.push({ binding: 16, name: "logits", access: "read_write" });
  } else if (mode === "norm_projection") {
    bindings.push({ binding: 17, name: "projection", access: "read_write" });
  } else if (mode !== "matvec_residual") {
    bindings.push({ binding: 6, name: "destination", access: "read_write" });
  }
  bindings.push({ binding: 7, name: "state", access: "read" });
  // V1 and V3 read the same weight bytes at the same offsets through a 16 B element type. The
  // dynamic offset a section is bound at is 256-aligned and the row stride is a whole number of
  // blocks, so every index the kernel forms is 16 B aligned.
  const overrides: BindingTypes = packedVec4(variant) ? { weights: "array<vec4<u32>>" } : {};

  const emit = mode === "norm_head"
    ? "logits[query * EMITS + row] = value;"
    : mode === "norm_projection"
      ? "projection[query * EMITS + row] = value;"
      : mode === "matvec_residual"
        ? "residual[query * EMITS + row] = residual[query * EMITS + row] + value;"
        : "destination[query * EMITS + row] = f16(value);";

  // A cooperating call has to run on every subgroup of the row group, because the reduction inside
  // it carries barriers; only one of them writes.
  const call = cooperative ? "rowTotal(ROWEXPR, subgroup, slice, lane)" : "rowSum(ROWEXPR, lane)";
  const sum = (rowExpr: string): string => call.replace("ROWEXPR", rowExpr);
  const writer = cooperative ? "lane == 0u && slice == 0u" : "lane == 0u";

  const body = mode === "norm_swiglu"
    ? `
  for (var r = 0u; r < ROWS_PER_SG; r = r + 1u) {
    let want = rowBase + r;
    let row = min(want, EMITS - 1u);
    let gate = ${sum("row")};
    let up = ${sum("row + EMITS")};
    if (${writer} && want < EMITS) {
      destination[query * EMITS + row] = f16((gate / (1.0 + exp(-gate))) * up);
    }
  }
`
    : `  for (var r = 0u; r < ROWS_PER_SG; r = r + 1u) {
    let want = rowBase + r;
    let row = min(want, EMITS - 1u);
    let value = ${sum("row")};
    if (${writer} && want < EMITS) {
      ${emit}
    }
  }
`;

  const dequant = packedVec4(variant) ? DEQUANT_F16 : DEQUANT;
  const crossDecl = cooperative ? `var<workgroup> cross: array<f32, ${subgroups}>;\n` : "";
  const rowBase = cooperative
    ? "let rowBase = wid.x * ROWS_PER_WG + (subgroup / SUBGROUPS_PER_ROW) * ROWS_PER_SG;"
    : "let rowBase = wid.x * ROWS_PER_WG + (local / SG) * ROWS_PER_SG;";
  const sliceDecl = cooperative
    ? "  let subgroup = local / SG;\n  let slice = subgroup % SUBGROUPS_PER_ROW;\n"
    : "";
  // Emitted only where it is read, so a one-subgroup-per-row cell -- which is every cell the
  // carried arms run -- generates the source B1.3 measured, byte for byte.
  const cooperativeConst = cooperative ? `const SUBGROUPS_PER_ROW: u32 = ${subgroupsPerRow}u;\n` : "";

  const code = `${HEADER}${STATE_STRUCT}${dequant}
const COLS: u32 = ${cols}u;
const BLOCKS: u32 = ${blocks}u;
const BLOCK_ITERS: u32 = ${Math.ceil(blocks / (subgroupSize * subgroupsPerRow))}u;
const WG: u32 = ${workgroupSize}u;
const SG: u32 = ${subgroupSize}u;
${cooperativeConst}const ROWS_PER_SG: u32 = ${rowsPerSubgroup}u;
const ROWS_PER_WG: u32 = ${rowsPerWorkgroup}u;
const EMITS: u32 = ${emits}u;
const EPSILON: f32 = ${shape.epsilon};

${declare(bindings, overrides)}

var<workgroup> stage: array<f32, ${cols}>;
var<workgroup> reduce: array<f32, ${workgroupSize}>;
${crossDecl}${zeroPointFn(zeroPoints, blocks)}${mode === "matvec_residual" ? STAGE_SOURCE : stageResidual(preScale)}${rowSumSource(variant, subgroupsPerRow)}
@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) local: u32,
  @builtin(subgroup_invocation_id) lane: u32,
) {
  let query = wid.y;
  if (query >= state.queries) { return; }
  stageInput(query, local);
${sliceDecl}  ${rowBase}
${body}}
`;
  return { code, bindings, workgroupSize };
}

// Embedding lookup: one workgroup per query token, dequantising a GatherBlockQuantized row
// (bits 4, block 32, gather axis 0, quantize axis 1) straight into the f32 residual stream.
export function embedKernel(shape: RuntimeShape, workgroupSize: number): KernelSource {
  const blocks = shape.hidden / shape.quantBlock;
  const bindings: BindingSlot[] = [
    { binding: 0, name: "weights", access: "read" },
    { binding: 1, name: "scales", access: "read" },
    { binding: 2, name: "zero_points", access: "read" },
    { binding: 4, name: "residual", access: "read_write" },
    { binding: 7, name: "state", access: "read" },
    { binding: 8, name: "tokens", access: "read" },
  ];
  const code = `${HEADER}${STATE_STRUCT}
const HIDDEN: u32 = ${shape.hidden}u;
const BLOCK: u32 = ${shape.quantBlock}u;
const BLOCKS: u32 = ${blocks}u;
const U32_PER_ROW: u32 = ${shape.hidden / 8}u;
const ZP_BYTES_PER_ROW: u32 = ${Math.ceil(blocks / 2)}u;
const WG: u32 = ${workgroupSize}u;

${declare(bindings, {})}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let query = wid.x;
  if (query >= state.queries) { return; }
  let token = tokens[state.step + query];
  for (var i = local; i < HIDDEN; i = i + WG) {
    let block = i / BLOCK;
    let packed = weights[token * U32_PER_ROW + i / 8u];
    let nibble = (packed >> ((i % 8u) * 4u)) & 15u;
    let zpIndex = token * ZP_BYTES_PER_ROW + block / 2u;
    let zpByte = (zero_points[zpIndex / 4u] >> ((zpIndex % 4u) * 8u)) & 255u;
    var zpv: u32 = zpByte & 15u;
    if ((block & 1u) == 1u) { zpv = zpByte >> 4u; }
    residual[query * HIDDEN + i] = (f32(nibble) - f32(zpv)) * scales[token * BLOCKS + block];
  }
}
`;
  return { code, bindings, workgroupSize };
}

export function embedWorkgroups(queries: number): Workgroups {
  return [queries, 1, 1];
}

// Depthwise causal conv over B*x, gated by C, with the rolling state in a persistent buffer.
// One thread owns one channel and walks the queries in order, so the 1..64 query path is exact
// without a second dispatch and without ever materialising a [queries, hidden] convolution.
export function convKernel(shape: RuntimeShape, workgroupSize: number): KernelSource {
  if (shape.convTaps !== 3) {
    throw new Error(`convKernel: registered for 3 taps, got ${shape.convTaps}`);
  }
  const bindings: BindingSlot[] = [
    { binding: 3, name: "gamma", access: "read" },
    { binding: 5, name: "source", access: "read" },
    { binding: 6, name: "destination", access: "read_write" },
    { binding: 7, name: "state", access: "read" },
    { binding: 9, name: "cache", access: "read_write" },
  ];
  const code = `${HEADER}${STATE_STRUCT}
const HIDDEN: u32 = ${shape.hidden}u;
const TAPS: u32 = ${shape.convTaps}u;
const WG: u32 = ${workgroupSize}u;

${declare(bindings, {})}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let channel = gid.x;
  if (channel >= HIDDEN) { return; }
  let w0 = gamma[channel * TAPS];
  let w1 = gamma[channel * TAPS + 1u];
  let w2 = gamma[channel * TAPS + 2u];
  var s1 = f32(cache[channel * TAPS + 1u]);
  var s2 = f32(cache[channel * TAPS + 2u]);
  for (var q = 0u; q < state.queries; q = q + 1u) {
    let base = q * 3u * HIDDEN;
    let bx = f32(source[base + channel]) * f32(source[base + 2u * HIDDEN + channel]);
    let y = w0 * s1 + w1 * s2 + w2 * bx;
    destination[q * HIDDEN + channel] = f16(f32(source[base + HIDDEN + channel]) * y);
    s1 = s2;
    s2 = bx;
  }
  cache[channel * TAPS] = f16(0.0);
  cache[channel * TAPS + 1u] = f16(s1);
  cache[channel * TAPS + 2u] = f16(s2);
}
`;
  return { code, bindings, workgroupSize };
}

export function convWorkgroups(shape: RuntimeShape, workgroupSize: number): Workgroups {
  return [Math.ceil(shape.hidden / workgroupSize), 1, 1];
}

// Sampling in two dispatches over the last query's logits. Greedy is a plain argmax; T = 1 is the
// Gumbel-max trick with a counter-based PRNG, which draws exactly from softmax(logits) with no
// normalising pass. Partials carry (bitcast score, index) pairs.
export function samplePartialKernel(shape: RuntimeShape, workgroupSize: number, partials: number): KernelSource {
  if (shape.vocab % partials !== 0) {
    throw new Error(`samplePartialKernel: vocab ${shape.vocab} is not a multiple of ${partials}`);
  }
  const bindings: BindingSlot[] = [
    { binding: 7, name: "state", access: "read" },
    { binding: 15, name: "partials", access: "read_write" },
    { binding: 16, name: "logits", access: "read" },
  ];
  const code = `${HEADER}${STATE_STRUCT}
const VOCAB: u32 = ${shape.vocab}u;
const WG: u32 = ${workgroupSize}u;
const PER_GROUP: u32 = ${shape.vocab / partials}u;

${declare(bindings, {})}

var<workgroup> bestValue: array<f32, ${workgroupSize}>;
var<workgroup> bestIndex: array<u32, ${workgroupSize}>;
${GUMBEL}
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let offset = (state.queries - 1u) * VOCAB;
  var value: f32 = -3.0e38;
  var index: u32 = 0u;
  for (var i = wid.x * PER_GROUP + local; i < (wid.x + 1u) * PER_GROUP; i = i + WG) {
    var s = logits[offset + i];
    if (state.sample_mode == 1u) { s = s + gumbel(state.rng, i); }
    if (s > value) { value = s; index = i; }
  }
  bestValue[local] = value;
  bestIndex[local] = index;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      if (bestValue[local + stride] > bestValue[local]) {
        bestValue[local] = bestValue[local + stride];
        bestIndex[local] = bestIndex[local + stride];
      }
    }
    workgroupBarrier();
  }
  if (local == 0u) {
    partials[wid.x * 2u] = bitcast<u32>(bestValue[0]);
    partials[wid.x * 2u + 1u] = bestIndex[0];
  }
}
`;
  return { code, bindings, workgroupSize };
}

// The closing dispatch of a token: reduce the partials, write the sampled id where the next
// embedding lookup will read it, and advance position/step/rng on the device. Nothing here
// returns to the host, which is what allows a whole run to be encoded ahead of time.
export function sampleFinalKernel(shape: RuntimeShape, workgroupSize: number, partials: number): KernelSource {
  const bindings: BindingSlot[] = [
    { binding: 7, name: "state", access: "read_write" },
    { binding: 8, name: "tokens", access: "read_write" },
    { binding: 15, name: "partials", access: "read" },
    { binding: 17, name: "sampled", access: "read_write" },
  ];
  const code = `${HEADER}${STATE_STRUCT}
const WG: u32 = ${workgroupSize}u;
const PARTIALS: u32 = ${partials}u;
const MAX_POSITIONS: u32 = ${shape.maxPositions}u;

${declare(bindings, {})}

var<workgroup> bestValue: array<f32, ${workgroupSize}>;
var<workgroup> bestIndex: array<u32, ${workgroupSize}>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_index) local: u32) {
  var value: f32 = -3.0e38;
  var index: u32 = 0u;
  for (var i = local; i < PARTIALS; i = i + WG) {
    let candidate = bitcast<f32>(partials[i * 2u]);
    if (candidate > value) { value = candidate; index = partials[i * 2u + 1u]; }
  }
  bestValue[local] = value;
  bestIndex[local] = index;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      if (bestValue[local + stride] > bestValue[local]) {
        bestValue[local] = bestValue[local + stride];
        bestIndex[local] = bestIndex[local + stride];
      }
    }
    workgroupBarrier();
  }
  if (local == 0u) {
    let token = bestIndex[0];
    let last = state.step + state.queries - 1u;
    sampled[last] = token;
    if (state.freerun == 1u && last + 1u < MAX_POSITIONS) {
      tokens[last + 1u] = token;
    }
    if (state.advance == 1u) {
      state.position = state.position + state.queries;
      state.step = state.step + state.queries;
      state.rng = state.rng * 1664525u + 1013904223u;
    }
  }
}
`;
  return { code, bindings, workgroupSize };
}

export function samplePartialWorkgroups(partials: number): Workgroups {
  return [partials, 1, 1];
}

export function sampleFinalWorkgroups(): Workgroups {
  return [1, 1, 1];
}
