// The registered dispatch plan, as an ordered list per token, for any of the four arms. This module
// is the single source of truth: P0 prices from it, the runtime encodes exactly this list, and the
// context sweep re-derives it at each cache depth.
//
// The deferred-reduction contract, which is what makes a fused block sound on a memory model with
// no device-scope visibility guarantee and no f32 atomics:
//
//   1. every workgroup stages x[i] = residual_in[i] + sum of the pending partial slices -- read
//      only, and it needs the whole vector because a norm follows;
//   2. it writes residual_out[i] = residual_in[i] + that same sum, for a strided row slice no
//      other workgroup of the dispatch writes;
//   3. it writes its own output as one fresh slice of partials_out.
//
// Reads land on residual_in / partials_in, writes on residual_out / partials_out, so both
// ping-pong and no dispatch reads a buffer it is concurrently writing. Sums run in slice index
// order in every consumer, so every reduction is bitwise reproducible.
//
// B1.4 keeps that contract only for the arms that keep B1.3's fused MLP. N1 removes it entirely:
// every matvec owns whole output rows and reads the whole input vector, so nothing defers a sum,
// no partial slice is ever written, and the `fold` dispatch disappears from the plan on its own --
// `emitFold` finds nothing pending. The residual parity therefore never flips on an N1 arm, and
// flips fourteen times (once per fused MLP) on the arms that keep it. Both are even, which is the
// condition for encoding one command buffer and replaying it every token. See
// `fusedBlocksPerToken`.
//
// Every dispatch carries the workgroup count and workgroup size it will actually be encoded with,
// because the adopted cost model is
//
//   t_AR ~= sum over dispatches of max(lambda_kind, bytes_i / BW_kernel(class_i, bytes_i, wg_i, size_i))
//
// and a plan that does not carry its own occupancy cannot be priced. B1.4 replaces B1.3's
// two-geometry tag with the pair (`site`, `geometry`): `site` names which of the six real matvec
// (class, size) cells P0a measured this dispatch against, and `geometry` is the occupancy it runs
// at -- B1.3's carried assignment on the baseline arms, P0a's measured knee on the N1 arms.
import {
  headMatrix,
  layerMatrices,
  quantMatrixBytes,
  type ModelShape,
  type WeightMatrix,
} from "./shape.ts";

export type DispatchKind =
  | "embed"
  | "norm_matvec"
  | "conv_core"
  | "matvec_residual"
  | "norm_matvec_swiglu"
  | "norm_head"
  | "sample_partial"
  | "sample_final"
  | "attn_proj"
  | "attn_core_qkv"
  | "attn_core_flash"
  | "attn_merge"
  | "mlp_fused"
  | "fold";

// The kinds that publish partial slices and flip both parities. B1.3's two attention blocks are
// gone: no arm of this unit fuses attention.
export const FUSED_KINDS: readonly DispatchKind[] = ["mlp_fused"];

// Which half of an attention layer a dispatch is. The spec requires core and projection time to be
// logged separately so the two are priced apart, and N2 splits the core itself into a partial pass
// and a merge, so the merge is its own part rather than being folded into the core's total.
export type AttnPart = "none" | "projection" | "core" | "merge" | "out_projection";

// The six real matvec (class, size) cells P0a measures. A dispatch's price is an exact lookup in
// that table -- the plan's shapes ARE the measured shapes, so nothing is interpolated.
export const MATVEC_SHAPES = [
  "attn_o_proj",
  "mlp_down",
  "attn_qkv",
  "conv_in_proj",
  "mlp_gate_up",
  "head",
] as const;
export type MatvecShape = (typeof MATVEC_SHAPES)[number];

// The five int4 matvec kernel classes. A class is the fused pre/post work around the same inner
// product -- a plain residual accumulate, a pre-norm, a pre-norm feeding a projection, a pre-norm
// feeding swiglu, a pre-norm feeding the head's zero-pointed rows -- and the class is what P0a
// measures a curve for. Each of the six shapes is one class at one size, so `BW_kernel(class,
// bytes, ...)` and `BW_kernel(shape, ...)` index the same table.
export const MATVEC_CLASSES = [
  "matvec_residual",
  "norm_matvec",
  "norm_projection",
  "norm_swiglu",
  "norm_head",
] as const;
export type MatvecClass = (typeof MATVEC_CLASSES)[number];

export const CLASS_OF_SHAPE: Readonly<Record<MatvecShape, MatvecClass>> = {
  attn_o_proj: "matvec_residual",
  mlp_down: "matvec_residual",
  attn_qkv: "norm_projection",
  conv_in_proj: "norm_matvec",
  mlp_gate_up: "norm_swiglu",
  head: "norm_head",
};

// The seven matvec dispatch sites in the token. `conv_out_proj` is not a seventh shape: the conv
// out-projection is 1024x1024 `matvec_residual`, byte for byte and class for class the attention
// out-projection's cell, so the two share one measured curve. They do NOT share an occupancy --
// B1.3 ran the conv projections at base geometry and the attention out-projection at wide -- which
// is exactly why the site and the shape are separate fields.
export const MATVEC_SITES = [
  "conv_in_proj",
  "conv_out_proj",
  "attn_qkv",
  "attn_o_proj",
  "mlp_gate_up",
  "mlp_down",
  "head",
] as const;
export type MatvecSite = (typeof MATVEC_SITES)[number];

export const SHAPE_OF_SITE: Readonly<Record<MatvecSite, MatvecShape>> = {
  conv_in_proj: "conv_in_proj",
  conv_out_proj: "attn_o_proj",
  attn_qkv: "attn_qkv",
  attn_o_proj: "attn_o_proj",
  mlp_gate_up: "mlp_gate_up",
  mlp_down: "mlp_down",
  head: "head",
};

// The sites N1 re-occupies. The lever is registered as "MLP as two wide dispatches ... same for
// attention out-projection and conv in/out projections", so the attention q/k/v projection and the
// head keep B1.3's geometry on every arm and are not part of the lever. Sites outside this set run
// at the carried geometry whatever the arm.
export const SPLIT_N_SITES: readonly MatvecSite[] = [
  "conv_in_proj",
  "conv_out_proj",
  "attn_o_proj",
  "mlp_gate_up",
  "mlp_down",
];

export interface FusedWeight {
  // The offline row concatenation this dispatch reads, in source order. A single-element list is
  // an unmodified ONNX matrix; a multi-element list is a row concatenation whose packed bytes are
  // the concatenation of the sources' packed bytes (same block size, same column count).
  readonly sources: readonly WeightMatrix[];
  readonly rows: number;
  readonly cols: number;
  readonly zeroPoints: boolean;
  readonly bytes: number;
}

// One quantised matrix as a dispatch reads it. `section` is the blob section base name, which the
// layout arm's plan defines for every matrix in the model; `kSlices` selects the K-sliced repack of
// that matrix when the reader is a split-K dispatch, and 0 means the unpermuted section. `bytes` is
// what this dispatch actually moves, which exceeds the matrix's own size wherever a fusion reads a
// row twice.
export interface WeightRead {
  readonly section: string;
  readonly rows: number;
  readonly cols: number;
  readonly zeroPoints: boolean;
  readonly bytes: number;
  readonly kSlices: number;
}

export interface FusionMeta {
  // Partial slices pending on entry, which this dispatch reduces while staging.
  readonly foldSlices: number;
  // Partial slices this dispatch writes, one per workgroup.
  readonly slices: number;
  readonly residualIn: number;
  readonly residualOut: number;
  readonly partialsIn: number;
  readonly partialsOut: number;
}

// One matvec geometry. B1.4 adds `subgroupsPerRow`: the number of subgroups that cooperate on one
// output row, which is what makes fewer rows per workgroup than subgroups per workgroup a real
// occupancy point rather than idle lanes. At subgroupsPerRow = 1 the generalised kernel is bitwise
// B1.3's.
export interface MatvecGeometry {
  readonly workgroupSize: number;
  readonly subgroupSize: number;
  readonly rowsPerSubgroup: number;
  readonly subgroupsPerRow: number;
}

export interface FoldGeometry {
  readonly workgroupSize: number;
  readonly threadsPerRow: number;
}

// Everything the plan needs to state its own occupancy. Nothing here is defaulted downstream: the
// runtime encodes the counts this module computes and asserts they match.
export interface PlanGeometry {
  // B1.3's registered assignment, per site. The baseline arm runs every site here.
  readonly carried: Readonly<Record<MatvecSite, MatvecGeometry>>;
  // P0a's measured knee, per shape cell. N1 runs SPLIT_N_SITES here.
  readonly knee: Readonly<Record<MatvecShape, MatvecGeometry>>;
  readonly blockWorkgroupSize: number;
  readonly attnCoreWorkgroupSize: number;
  readonly flashCoreWorkgroupSize: number;
  readonly mergeWorkgroupSize: number;
  readonly convWorkgroupSize: number;
  readonly embedWorkgroupSize: number;
  readonly sampleWorkgroupSize: number;
  readonly foldGeometry: FoldGeometry;
}

export interface PlannedDispatch {
  readonly index: number;
  readonly name: string;
  readonly kind: DispatchKind;
  readonly layer: number;
  // The matrix a matvec kernel reads. Null on the fused MLP and on every non-matmul dispatch;
  // `reads` is the byte-accurate list in all cases.
  readonly weight: FusedWeight | null;
  readonly reads: readonly WeightRead[];
  // Non-matrix constants this dispatch reads (norm weights, conv taps, rope rows).
  readonly constantBytes: number;
  // Activation, cache and logit traffic, at the position the plan is priced for.
  readonly readBytes: number;
  readonly writeBytes: number;
  readonly fusion: FusionMeta;
  // The occupancy axis: what this dispatch is encoded with, not what it might be.
  readonly workgroups: number;
  readonly workgroupSize: number;
  // Which of the seven matvec sites this dispatch is, and the geometry it runs at. Both null on
  // everything that is not a matvec.
  readonly site: MatvecSite | null;
  readonly geometry: MatvecGeometry | null;
  readonly attnPart: AttnPart;
}

export interface PlanPrecision {
  readonly residualBytes: number;
  readonly hiddenBytes: number;
  readonly kvCacheBytes: number;
  readonly convCacheBytes: number;
  readonly logitsBytes: number;
}

// The two levers, as the independent booleans they are. `splitN` unfuses the MLP and moves every
// restructured matvec to P0a's knee; `flash` replaces the streaming attention core with the
// split-position core and its merge. Four arms, and each lever is measured alone and in
// combination.
export interface PlanLevers {
  readonly splitN: boolean;
  readonly flash: boolean;
}

export interface PlanSlices {
  // Partial slices the fused MLP publishes: one per quantisation block of the K it splits.
  readonly mlp: number;
}

// N2's shape. `blocks` is S, the number of position blocks the core is split across, fixed for a
// whole decode run and chosen per context length from P0c's curve. `partialWords` is the f32 stride
// of one (block, q head) partial: the accumulator lanes plus the running maximum and denominator,
// padded so consecutive partials do not share a cache line boundary.
export interface PlanFlash {
  readonly blocks: number;
  readonly maxBlocks: number;
  readonly partialWords: number;
}

// The number of (score, index) pairs the sampling scan leaves for the final reduction.
export interface PlanSamplePartials {
  readonly plain: number;
}

export interface PlanOptions {
  readonly queries: number;
  readonly position: number;
  readonly precision: PlanPrecision;
  readonly levers: PlanLevers;
  readonly slices: PlanSlices;
  readonly flash: PlanFlash;
  readonly geometry: PlanGeometry;
  readonly samplePartials: PlanSamplePartials;
}

// Split-N alone: the arm that reads every matrix in the model through its own dispatch and fuses
// nothing, so its dispatch names define the blob's section set. It is not the no-lever arm -- that
// is the baseline, whose MLP is fused -- and the name says which of the two it is.
export const LAYOUT_LEVERS: PlanLevers = { splitN: true, flash: false };

// One sampling partial is a (bitcast score, vocabulary index) pair of u32.
export const SAMPLE_PAIR_BYTES = 8;

export function rowsPerWorkgroup(geometry: MatvecGeometry): number {
  const { workgroupSize, subgroupSize, rowsPerSubgroup, subgroupsPerRow } = geometry;
  if (workgroupSize % subgroupSize !== 0) {
    throw new Error(`rowsPerWorkgroup: workgroup ${workgroupSize} is not a multiple of subgroup ${subgroupSize}`);
  }
  const subgroups = workgroupSize / subgroupSize;
  if (!Number.isInteger(subgroupsPerRow) || subgroupsPerRow < 1) {
    throw new Error(`rowsPerWorkgroup: subgroupsPerRow must be a positive integer, got ${subgroupsPerRow}`);
  }
  if (subgroups % subgroupsPerRow !== 0) {
    throw new Error(`rowsPerWorkgroup: ${subgroups} subgroups do not partition into groups of ${subgroupsPerRow}`);
  }
  if (!Number.isInteger(rowsPerSubgroup) || rowsPerSubgroup < 1) {
    throw new Error(`rowsPerWorkgroup: rowsPerSubgroup must be a positive integer, got ${rowsPerSubgroup}`);
  }
  return (subgroups / subgroupsPerRow) * rowsPerSubgroup;
}

// The inverse of `rowsPerWorkgroup`: the one geometry the P0a sweep runs at a given rows-per-
// workgroup. The map has to be a function for the curve to be a table, so the sweep fixes it --
// below one row per subgroup, subgroups cooperate on a row and each owns exactly one row's share;
// at or above it, every subgroup owns whole rows and none cooperate. Both of B1.3's carried
// geometries are fixed points of this rule, so the baseline arm is priced from measured cells.
export function geometryForRows(workgroupSize: number, subgroupSize: number, rows: number): MatvecGeometry {
  if (workgroupSize % subgroupSize !== 0) {
    throw new Error(`geometryForRows: workgroup ${workgroupSize} is not a multiple of subgroup ${subgroupSize}`);
  }
  if (!Number.isInteger(rows) || rows < 1) {
    throw new Error(`geometryForRows: rows must be a positive integer, got ${rows}`);
  }
  const subgroups = workgroupSize / subgroupSize;
  if (rows >= subgroups) {
    if (rows % subgroups !== 0) {
      throw new Error(`geometryForRows: ${rows} rows do not divide across ${subgroups} subgroups`);
    }
    return { workgroupSize, subgroupSize, rowsPerSubgroup: rows / subgroups, subgroupsPerRow: 1 };
  }
  if (subgroups % rows !== 0) {
    throw new Error(`geometryForRows: ${subgroups} subgroups do not partition into ${rows} rows`);
  }
  return { workgroupSize, subgroupSize, rowsPerSubgroup: 1, subgroupsPerRow: subgroups / rows };
}

// Whether a cell is a real occupancy point: every cooperating subgroup must get at least one
// quantisation block of the row, or the cell leaves whole subgroups idle and measures nothing.
export function cellReachable(geometry: MatvecGeometry, cols: number, quantBlock: number): boolean {
  if (cols % quantBlock !== 0) {
    throw new Error(`cellReachable: ${cols} columns are not a multiple of the ${quantBlock}-wide block`);
  }
  return geometry.subgroupsPerRow * geometry.subgroupSize <= cols / quantBlock;
}

// The occupancy a site runs at on an arm: the carried assignment everywhere, except that N1 moves
// the sites it restructures onto their shape's measured knee.
export function geometryFor(site: MatvecSite, levers: PlanLevers, geometry: PlanGeometry): MatvecGeometry {
  if (levers.splitN && SPLIT_N_SITES.includes(site)) {
    return geometry.knee[SHAPE_OF_SITE[site]];
  }
  return geometry.carried[site];
}

// S from the registered rule, for a depth P0c did not measure: one 64-position block per tile,
// capped at maxBlocks.
export function blocksForPositions(totalPositions: number, positionsPerBlock: number, maxBlocks: number): number {
  if (!Number.isInteger(totalPositions) || totalPositions < 1) {
    throw new Error(`blocksForPositions: ${totalPositions} is not a positive position count`);
  }
  if (!Number.isInteger(positionsPerBlock) || positionsPerBlock < 1) {
    throw new Error(`blocksForPositions: ${positionsPerBlock} is not a positive block length`);
  }
  return Math.min(maxBlocks, Math.max(1, Math.ceil(totalPositions / positionsPerBlock)));
}

function fuse(sources: readonly WeightMatrix[], quantBlock: number, bits: number): FusedWeight {
  const first = sources[0];
  if (first === undefined) {
    throw new Error("fuse: empty source list");
  }
  for (const source of sources) {
    if (source.cols !== first.cols) {
      throw new Error(`fuse: column mismatch ${source.name} ${source.cols} != ${first.cols}`);
    }
    if (source.zeroPoints !== first.zeroPoints) {
      throw new Error(`fuse: zero-point mismatch in ${source.name}`);
    }
  }
  const rows = sources.reduce((sum, source) => sum + source.rows, 0);
  const bytes = quantMatrixBytes(rows, first.cols, first.zeroPoints, bits, quantBlock);
  const summed = sources.reduce((sum, source) => sum + source.bytes.total, 0);
  if (bytes.total !== summed) {
    throw new Error(`fuse: concatenated bytes ${bytes.total} != sum of sources ${summed}`);
  }
  return { sources, rows, cols: first.cols, zeroPoints: first.zeroPoints, bytes: bytes.total };
}

function pick(matrices: readonly WeightMatrix[], role: string): WeightMatrix {
  const found = matrices.find((matrix) => matrix.role === role);
  if (found === undefined) {
    throw new Error(`plan: no matrix with role ${role}`);
  }
  return found;
}

// The K-sliced repack is a pure permutation of the packed nibbles and the scales, so a slice's
// bytes are the matrix's bytes divided by the slice count and the total is unchanged.
function readOf(section: string, rows: number, cols: number, zeroPoints: boolean, shape: ModelShape, kSlices: number): WeightRead {
  const bytes = quantMatrixBytes(rows, cols, zeroPoints, shape.quantBits, shape.quantBlock).total;
  return { section, rows, cols, zeroPoints, bytes, kSlices };
}

interface Cursor {
  residual: number;
  partials: number;
  pending: number;
}

// How many fused blocks one token encodes on an arm. The residual and partial buffers ping-pong on
// each of them and nowhere else, so the plan is replayable only when this is even. LFM2.5-230M has
// fourteen layers, so the arms that keep the fused MLP flip fourteen times and the N1 arms not at
// all -- both even.
export function fusedBlocksPerToken(shape: ModelShape, levers: PlanLevers): number {
  return levers.splitN ? 0 : shape.layerTypes.length;
}

export function planClosesOnStartingParity(shape: ModelShape, levers: PlanLevers): boolean {
  return fusedBlocksPerToken(shape, levers) % 2 === 0;
}

export function planStep(shape: ModelShape, options: PlanOptions): readonly PlannedDispatch[] {
  const { queries: q, position, precision, levers, slices, flash, geometry } = options;
  if (!Number.isInteger(q) || q < 1) {
    throw new Error(`planStep: queries must be a positive integer, got ${q}`);
  }
  const anyFusion = !levers.splitN;
  if (!planClosesOnStartingParity(shape, levers)) {
    throw new Error(
      `planStep: ${fusedBlocksPerToken(shape, levers)} fused blocks at ${shape.layerTypes.length} layers is odd, so the ` +
        "residual ping-pong does not close and the token cannot be encoded once and replayed",
    );
  }
  if (anyFusion && q !== 1) {
    throw new Error(`planStep: the fused MLP is a single-query decode specialisation, got q = ${q}`);
  }
  if (!Number.isInteger(flash.blocks) || flash.blocks < 1 || flash.blocks > flash.maxBlocks) {
    throw new Error(`planStep: ${flash.blocks} position blocks is outside 1..${flash.maxBlocks}`);
  }
  const { hidden, ffn, vocab, heads, headDim, kvHeads, convCache, quantBlock, quantBits } = shape;
  if (heads % kvHeads !== 0) {
    throw new Error(`planStep: ${heads} q heads do not divide into ${kvHeads} kv heads`);
  }
  const group = heads / kvHeads;
  const kv = kvHeads * headDim;
  const residual = hidden * precision.residualBytes * q;
  const hiddenVec = hidden * precision.hiddenBytes * q;
  const partialSlice = hidden * 4;
  // The split arm's projection scratch: q, k and v rows as f32, so the core's rope and norm see the
  // projection at accumulate precision and round once, into the f16 cache.
  const projectionRows = heads * headDim + 2 * kvHeads * headDim;
  const projection = projectionRows * 4 * q;
  const out: PlannedDispatch[] = [];
  const cursor: Cursor = { residual: 0, partials: 0, pending: 0 };

  // A split-N matvec gives each workgroup `rowsPerWorkgroup` output rows and one workgroup column
  // per query. `emits` is the number of rows the kernel writes, which is the matrix's row count
  // everywhere except the SwiGLU pair, where gate and up rows collapse to one output row each.
  const matvecWorkgroups = (emits: number, matvecGeometry: MatvecGeometry): number =>
    Math.ceil(emits / rowsPerWorkgroup(matvecGeometry)) * q;

  const push = (
    name: string,
    kind: DispatchKind,
    layer: number,
    weight: FusedWeight | null,
    reads: readonly WeightRead[],
    constantBytes: number,
    readBytes: number,
    writeBytes: number,
    fusion: FusionMeta,
    workgroups: number,
    workgroupSize: number,
    site: MatvecSite | null,
    dispatchGeometry: MatvecGeometry | null,
    attnPart: AttnPart,
  ): void => {
    if (!Number.isInteger(workgroups) || workgroups < 1) {
      throw new Error(`planStep: ${name} has a non-positive workgroup count ${workgroups}`);
    }
    if ((site === null) !== (dispatchGeometry === null)) {
      throw new Error(`planStep: ${name} carries a site without a geometry, or the reverse`);
    }
    out.push({
      index: out.length,
      name,
      kind,
      layer,
      weight,
      reads,
      constantBytes,
      readBytes,
      writeBytes,
      fusion,
      workgroups,
      workgroupSize,
      site,
      geometry: dispatchGeometry,
      attnPart,
    });
  };

  // One matvec dispatch, at whatever occupancy this arm gives its site.
  const matvec = (
    name: string,
    kind: DispatchKind,
    layer: number,
    weight: FusedWeight,
    reads: readonly WeightRead[],
    constantBytes: number,
    readBytes: number,
    writeBytes: number,
    emits: number,
    site: MatvecSite,
    attnPart: AttnPart,
  ): void => {
    const siteGeometry = geometryFor(site, levers, geometry);
    push(
      name,
      kind,
      layer,
      weight,
      reads,
      constantBytes,
      readBytes,
      writeBytes,
      plain(),
      matvecWorkgroups(emits, siteGeometry),
      siteGeometry.workgroupSize,
      site,
      siteGeometry,
      attnPart,
    );
  };

  const plain = (): FusionMeta => ({
    foldSlices: 0,
    slices: 0,
    residualIn: cursor.residual,
    residualOut: cursor.residual,
    partialsIn: cursor.partials,
    partialsOut: cursor.partials,
  });

  // Emitted before any consumer that must not fold: the head, whose 4096 tiles would each re-read
  // every slice, and every norm dispatch of an unfused block, whose per-thread staging cost would
  // otherwise exceed its own arithmetic. On an N1 arm nothing is ever pending and this emits
  // nothing, which is how the fold leaves the plan.
  const emitFold = (label: string): void => {
    if (cursor.pending === 0) {
      return;
    }
    const foldSlices = cursor.pending;
    push(
      label,
      "fold",
      -1,
      null,
      [],
      0,
      residual + foldSlices * partialSlice,
      residual,
      {
        foldSlices,
        slices: 0,
        residualIn: cursor.residual,
        residualOut: cursor.residual,
        partialsIn: cursor.partials,
        partialsOut: cursor.partials,
      },
      Math.ceil(hidden / (geometry.foldGeometry.workgroupSize / geometry.foldGeometry.threadsPerRow)),
      geometry.foldGeometry.workgroupSize,
      null,
      null,
      "none",
    );
    cursor.pending = 0;
  };

  // A fused block consumes whatever is pending and publishes its own slices, flipping both
  // parities. Nothing else in the plan flips them.
  const blockFusion = (sliceCount: number): FusionMeta => {
    const meta: FusionMeta = {
      foldSlices: cursor.pending,
      slices: sliceCount,
      residualIn: cursor.residual,
      residualOut: 1 - cursor.residual,
      partialsIn: cursor.partials,
      partialsOut: 1 - cursor.partials,
    };
    cursor.residual = 1 - cursor.residual;
    cursor.partials = 1 - cursor.partials;
    cursor.pending = sliceCount;
    return meta;
  };

  const blocks = hidden / quantBlock;
  const embeddingRow = hidden / 2 + blocks * 4 + Math.ceil(blocks / 2);
  // One embedding row per query token: dequantise into the fp32 residual stream.
  push("embed", "embed", -1, null, [], embeddingRow * q, 4 * q, residual, plain(), q, geometry.embedWorkgroupSize, null, null, "none");

  for (let layer = 0; layer < shape.layerTypes.length; layer += 1) {
    const kind = shape.layerTypes[layer];
    const matrices = layerMatrices(shape, layer);
    const normWeight = hidden * 4;
    const ropeRow = 2 * (headDim / 2) * 4 * q;
    if (kind === "conv") {
      const inProj = fuse([pick(matrices, "conv.in_proj")], quantBlock, quantBits);
      const outProj = fuse([pick(matrices, "conv.out_proj")], quantBlock, quantBits);
      const convState = hidden * convCache * precision.convCacheBytes;
      // The conv block is never fused in this unit: B1.2's F3 measured that fusing it cost
      // bandwidth. N1's only change here is the occupancy of the two projections.
      emitFold(`L${layer}.conv.fold`);
      matvec(
        `L${layer}.conv.in_proj`,
        "norm_matvec",
        layer,
        inProj,
        [readOf(`L${layer}.conv.in_proj`, inProj.rows, inProj.cols, false, shape, 0)],
        normWeight,
        residual,
        3 * hiddenVec,
        inProj.rows,
        "conv_in_proj",
        "none",
      );
      push(
        `L${layer}.conv.core`,
        "conv_core",
        layer,
        null,
        [],
        hidden * convCache * 4,
        3 * hiddenVec + convState,
        hiddenVec + convState,
        plain(),
        Math.ceil(hidden / geometry.convWorkgroupSize) * q,
        geometry.convWorkgroupSize,
        null,
        null,
        "none",
      );
      matvec(
        `L${layer}.conv.out_proj`,
        "matvec_residual",
        layer,
        outProj,
        [readOf(`L${layer}.conv.out_proj`, outProj.rows, outProj.cols, false, shape, 0)],
        0,
        hiddenVec + residual,
        residual,
        outProj.rows,
        "conv_out_proj",
        "none",
      );
    } else {
      // q, k and v are one row concatenation everywhere in this unit: the blob places one section
      // per dispatch weight, so reading them apart would need a section set no arm lays out. The
      // concatenation is a pure repack -- `fuse` refuses a source list whose bytes do not add up --
      // so it changes which matrix a dispatch binds and never how many bytes it reads.
      const qkv = fuse(
        [pick(matrices, "attn.q_proj"), pick(matrices, "attn.k_proj"), pick(matrices, "attn.v_proj")],
        quantBlock,
        quantBits,
      );
      const oProj = fuse([pick(matrices, "attn.o_proj")], quantBlock, quantBits);
      const totalPositions = position + q;
      const kvAppend = 2 * kv * precision.kvCacheBytes * q;
      emitFold(`L${layer}.attn.fold`);
      // The projection: RMSNorm and the whole q/k/v matrix as one split-N matvec. Rope is
      // deliberately NOT here: a non-interleaved rotation pairs row i with row i + headDim/2,
      // which at four rows per workgroup lands in a different workgroup, so doing it here would
      // either double the projection work or need a fourth dispatch. The core owns a whole KV
      // head's rows and does it for free.
      matvec(
        `L${layer}.attn.qkv`,
        "attn_proj",
        layer,
        qkv,
        [readOf(`L${layer}.attn.qkv`, qkv.rows, qkv.cols, false, shape, 0)],
        normWeight,
        residual,
        projection,
        qkv.rows,
        "attn_qkv",
        "projection",
      );
      if (levers.flash) {
        // N2. One workgroup per (KV head, position block), workgroupSize = headDim. Each block
        // reads only its own KV head's q rows and only the cached positions inside its own block,
        // K and V staged one 64-position tile at a time, and writes one online-softmax partial per
        // q head: 64 accumulator lanes plus the running maximum and denominator. The LAST block
        // owns the current position: it reads k and v from the projection, applies the head norms
        // and the rotation, writes them to the cache and uses its own copy, so no workgroup reads
        // a value another workgroup wrote in the same dispatch.
        const blockQRead = group * headDim * 4 * q;
        const flashPartials = flash.blocks * heads * flash.partialWords * 4 * q;
        push(
          `L${layer}.attn.flash`,
          "attn_core_flash",
          layer,
          null,
          [],
          2 * headDim * 4 + ropeRow,
          kvHeads * flash.blocks * blockQRead +
            kvHeads * 2 * headDim * 4 * q +
            2 * kvHeads * position * headDim * precision.kvCacheBytes,
          flashPartials + kvAppend,
          plain(),
          kvHeads * flash.blocks * q,
          geometry.flashCoreWorkgroupSize,
          null,
          null,
          "core",
        );
        // One workgroup per q head, one thread per dimension, blocks summed in ascending index
        // order so the log-sum-exp merge is bitwise reproducible. Always emitted: at S = 1 it is
        // acc / l, which is correct, so the dispatch list does not change with context length.
        push(
          `L${layer}.attn.merge`,
          "attn_merge",
          layer,
          null,
          [],
          0,
          flashPartials,
          hiddenVec,
          plain(),
          heads * q,
          geometry.mergeWorkgroupSize,
          null,
          null,
          "merge",
        );
      } else {
        // B1.3's core, carried: one workgroup per KV head carrying both of its q heads, streaming
        // K and V from global memory -- staging both for one KV head at 192 positions would be
        // 49 152 B against a 32 768 B workgroup budget, and 524 288 B at 2048. It writes the new k
        // and v first and then reads the whole cache including them.
        push(
          `L${layer}.attn.core`,
          "attn_core_qkv",
          layer,
          null,
          [],
          2 * headDim * 4 + ropeRow,
          projection + 2 * kvHeads * totalPositions * headDim * precision.kvCacheBytes,
          hiddenVec + kvAppend,
          plain(),
          kvHeads * q,
          geometry.attnCoreWorkgroupSize,
          null,
          null,
          "core",
        );
      }
      // The out-projection as a split-N matvec, in place on the residual. Split-N needs no partials
      // at all, so this arm carries no deferred sum here and no fold after it.
      matvec(
        `L${layer}.attn.o_proj`,
        "matvec_residual",
        layer,
        oProj,
        [readOf(`L${layer}.attn.o_proj`, oProj.rows, oProj.cols, false, shape, 0)],
        0,
        hiddenVec + residual,
        residual,
        oProj.rows,
        "attn_o_proj",
        "out_projection",
      );
    }

    // The SwiGLU pair is the same row concatenation: gate and up share a column count and a block
    // size, so the two matrices are one 5120-row read and the kernel emits one row per pair.
    const gateUp = fuse([pick(matrices, "mlp.gate_proj"), pick(matrices, "mlp.up_proj")], quantBlock, quantBits);
    const down = fuse([pick(matrices, "mlp.down_proj")], quantBlock, quantBits);
    if (levers.splitN) {
      // N1: two wide dispatches that own output rows. The fused block's 80-workgroup cap -- set by
      // the down projection's K-slice count, ffn / quantBlock = 80, the largest split-K the
      // reduction admits -- is gone, and with it the partial slices and the fold.
      emitFold(`L${layer}.mlp.fold`);
      matvec(
        `L${layer}.mlp.gate_up`,
        "norm_matvec_swiglu",
        layer,
        gateUp,
        [readOf(`L${layer}.mlp.gate_up`, gateUp.rows, gateUp.cols, false, shape, 0)],
        normWeight,
        residual,
        ffn * precision.hiddenBytes * q,
        // Gate and up rows collapse pairwise through the SwiGLU, so the kernel emits ffn rows from
        // 2 * ffn matrix rows.
        ffn,
        "mlp_gate_up",
        "none",
      );
      matvec(
        `L${layer}.mlp.down`,
        "matvec_residual",
        layer,
        down,
        [readOf(`L${layer}.mlp.down`, down.rows, down.cols, false, shape, 0)],
        0,
        ffn * precision.hiddenBytes * q + residual,
        residual,
        down.rows,
        "mlp_down",
        "none",
      );
    } else {
      const meta = blockFusion(slices.mlp);
      push(
        `L${layer}.mlp.fused`,
        "mlp_fused",
        layer,
        null,
        [
          readOf(`L${layer}.mlp.gate_up`, gateUp.rows, gateUp.cols, false, shape, 0),
          readOf(`L${layer}.mlp.down`, down.rows, down.cols, false, shape, slices.mlp),
        ],
        normWeight,
        residual + meta.foldSlices * partialSlice,
        residual + slices.mlp * partialSlice,
        meta,
        slices.mlp * q,
        geometry.blockWorkgroupSize,
        null,
        null,
        "none",
      );
    }
  }

  // Fused final RMSNorm + tied LM head. Only the last query's logits are sampled, but every
  // query's logits are produced so the verification path can score a whole draft. The head is not
  // an N1 site: at 4096 workgroups B1.3 measured it reading 376 GB/s, the top of its surface.
  emitFold("head.fold");
  const head = fuse([headMatrix(shape)], quantBlock, quantBits);
  const logits = vocab * precision.logitsBytes * q;
  const samplePairs = options.samplePartials.plain;
  matvec(
    "head",
    "norm_head",
    -1,
    head,
    [readOf("head", head.rows, head.cols, true, shape, 0)],
    hidden * 4,
    residual,
    logits,
    head.rows,
    "head",
    "none",
  );
  push(
    "sample.partial",
    "sample_partial",
    -1,
    null,
    [],
    0,
    logits,
    SAMPLE_PAIR_BYTES * samplePairs,
    plain(),
    samplePairs,
    geometry.sampleWorkgroupSize,
    null,
    null,
    "none",
  );
  push(
    "sample.final",
    "sample_final",
    -1,
    null,
    [],
    0,
    SAMPLE_PAIR_BYTES * samplePairs,
    SAMPLE_PAIR_BYTES,
    plain(),
    1,
    geometry.sampleWorkgroupSize,
    null,
    null,
    "none",
  );

  if (cursor.residual !== 0 || cursor.pending !== 0) {
    throw new Error(`planStep: token does not close on its starting parity (residual ${cursor.residual}, pending ${cursor.pending})`);
  }
  return out;
}

export interface PlanTotals {
  readonly dispatches: number;
  readonly matmulDispatches: number;
  readonly weightBytes: number;
  readonly constantBytes: number;
  readonly readBytes: number;
  readonly writeBytes: number;
  readonly totalBytes: number;
}

export function planTotals(plan: readonly PlannedDispatch[]): PlanTotals {
  let weightBytes = 0;
  let constantBytes = 0;
  let readBytes = 0;
  let writeBytes = 0;
  let matmulDispatches = 0;
  for (const dispatch of plan) {
    if (dispatch.reads.length > 0) {
      matmulDispatches += 1;
      for (const read of dispatch.reads) {
        weightBytes += read.bytes;
      }
    }
    constantBytes += dispatch.constantBytes;
    readBytes += dispatch.readBytes;
    writeBytes += dispatch.writeBytes;
  }
  return {
    dispatches: plan.length,
    matmulDispatches,
    weightBytes,
    constantBytes,
    readBytes,
    writeBytes,
    totalBytes: weightBytes + constantBytes + readBytes + writeBytes,
  };
}

export function dispatchBudget(shape: ModelShape, options: PlanOptions): number {
  return planStep(shape, options).length;
}

// Bytes each dispatch moves, which is what the cost model divides by BW_kernel(class, bytes, wg, size).
export function dispatchBytes(dispatch: PlannedDispatch): number {
  let weightBytes = 0;
  for (const read of dispatch.reads) {
    weightBytes += read.bytes;
  }
  return weightBytes + dispatch.constantBytes + dispatch.readBytes + dispatch.writeBytes;
}

// The dispatch after which one block of one layer has landed its whole contribution in the
// residual, which is the only point at which two arms with different structures inside that block
// hold the same quantity.
//
// A block that defers nothing has landed when its last dispatch ends; a block that publishes
// partial slices has landed when the fold that consumes them ends, and that fold is emitted at the
// head of the next block. Reading the landing point off the fusion metadata rather than off a
// dispatch name means a lever that moves the fold moves this with it.
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
