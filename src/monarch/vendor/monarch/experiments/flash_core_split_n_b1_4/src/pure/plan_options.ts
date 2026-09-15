// The single construction of PlanOptions from the registered constants, shared by every stage that
// re-derives the dispatch plan so no stage can price a different plan than P0 registered.
import {
  ARMS,
  FUSION,
  KERNELS,
  LAYOUT_ARM,
  LEVERS,
  PRECISION,
  PROMPT,
  RUNTIME,
  type ArmKey,
} from "../config.ts";
import {
  blocksForPositions,
  MATVEC_SHAPES,
  type MatvecGeometry,
  type MatvecShape,
  type MatvecSite,
  type PlanFlash,
  type PlanGeometry,
  type PlanLevers,
  type PlanOptions,
  type PlanSamplePartials,
  type PlanSlices,
} from "./plan.ts";
import { repackSpecs, weightLayout, type Layout, type RepackSpec } from "./sections.ts";
import { LFM2_5_230M, type ModelShape } from "./shape.ts";

export function armSpec(arm: ArmKey): (typeof ARMS)[number] {
  const found = ARMS.find((entry) => entry.key === arm);
  if (found === undefined) {
    throw new Error(`armSpec: unknown arm ${arm}`);
  }
  return found;
}

// The M stage publishes its winning arm as a string. This is the one place a string becomes an
// ArmKey, so an arm the plan does not know fails here rather than deeper in.
export function armKeyOf(value: string): ArmKey {
  const spec = ARMS.find((entry) => entry.key === value);
  if (spec === undefined) {
    throw new Error(`armKeyOf: unknown arm ${value}`);
  }
  return spec.key;
}

// An arm IS its two lever settings; ARMS is the registered table of the four combinations this unit
// measures, so the mapping is a lookup rather than a set of flags to be assembled.
export function leversFor(arm: ArmKey): PlanLevers {
  const spec = armSpec(arm);
  return { splitN: spec.splitN, flash: spec.flash };
}

// The arm carrying one lever and nothing else. Derived from the registry rather than named, so a
// change to the lever lattice moves every reader with it: the correctness battery's attributable
// pairing -- the single-lever arm and the baseline differ by exactly one lever, so a mismatch at the
// landing point names the lever that caused it -- and the L3 rung's droppable set are this same
// derivation asked for two different levers.
export function singleLeverArm(lever: keyof PlanLevers): ArmKey {
  const found = ARMS.filter((entry) => {
    const levers = leversFor(entry.key);
    return levers[lever] && Object.entries(levers).every(([name, on]) => name === lever || !on);
  });
  const arm = found[0];
  if (arm === undefined || found.length !== 1) {
    throw new Error(`SINGLE_LEVER_ARM: ${found.length} registered arms carry ${lever} alone`);
  }
  return arm.key;
}

// L3 drops the single-lever N2 arm from the timed set, which is that arm and only that one. N2 is
// still priced after the drop, by difference between the two remaining arms that differ only in it.
export function n2AloneArms(): readonly ArmKey[] {
  return [singleLeverArm("flash")];
}

// The slice count is a rule, not a number: the fused MLP publishes one slice per quantisation block
// of the K it splits. LEVERS registers what the rule evaluates to at LFM2.5-230M's shape and that
// evaluation is asserted below, so a shape change cannot silently re-slice the fused block.
export function planSlicesFor(shape: ModelShape): PlanSlices {
  const mlp = shape.ffn / shape.quantBlock;
  if (!Number.isInteger(mlp)) {
    throw new Error(`planSlicesFor: ffn ${shape.ffn} must be a multiple of the quantisation block ${shape.quantBlock}`);
  }
  if (shape.hidden % shape.heads !== 0) {
    throw new Error(`planSlicesFor: hidden ${shape.hidden} does not divide into ${shape.heads} heads`);
  }
  if (shape.heads % shape.kvHeads !== 0) {
    throw new Error(`planSlicesFor: ${shape.heads} q heads do not divide into ${shape.kvHeads} kv heads`);
  }
  return { mlp };
}

// The largest number of slices any single dispatch publishes, i.e. the partial buffer's row count.
export function maxSlicesOf(shape: ModelShape): number {
  return planSlicesFor(shape).mlp;
}

// The registered model's evaluation of the rule, held to the number LEVERS registers.
export const PLAN_SLICES: PlanSlices = planSlicesFor(LFM2_5_230M);
export const MAX_SLICES = maxSlicesOf(LFM2_5_230M);

if (LEVERS.mlpFused.slices !== PLAN_SLICES.mlp) {
  throw new Error(`plan_options: mlpFused registers ${LEVERS.mlpFused.slices} slices, the shape rule gives ${PLAN_SLICES.mlp}`);
}

// N2's core runs one thread per head dimension in the score pass and one per dimension in the value
// pass, so its workgroup size IS the head dimension. The merge is thread-per-dimension too.
if (LEVERS.flash.coreWorkgroupSize !== LFM2_5_230M.headDim) {
  throw new Error(
    `plan_options: the flash core registers workgroup ${LEVERS.flash.coreWorkgroupSize} against head dimension ${LFM2_5_230M.headDim}`,
  );
}
if (LEVERS.flash.mergeWorkgroupSize !== LFM2_5_230M.headDim) {
  throw new Error(
    `plan_options: the merge registers workgroup ${LEVERS.flash.mergeWorkgroupSize} against head dimension ${LFM2_5_230M.headDim}`,
  );
}
// One partial is headDim accumulator lanes plus the running maximum and denominator, padded up.
if (LEVERS.flash.partialWords < LFM2_5_230M.headDim + 2) {
  throw new Error(
    `plan_options: ${LEVERS.flash.partialWords} partial words cannot hold ${LFM2_5_230M.headDim} lanes plus a maximum and a denominator`,
  );
}

// FUSION records that this unit's blob concatenates q/k/v and gate/up; planStep implements it
// unconditionally, because the layout places one section per dispatch weight and an unconcatenated
// pair or triple would need sections no arm defines. This holds the record to the implementation so
// a future unit that wants the matrices apart has to change both.
if (!FUSION.concatQkv || !FUSION.concatGateUp) {
  throw new Error("plan_options: planStep has no unconcatenated form; FUSION registers both concatenations as true");
}

export const PLAN_SAMPLE_PARTIALS: PlanSamplePartials = { plain: KERNELS.sample.partials };

const BASE_GEOMETRY: MatvecGeometry = {
  workgroupSize: KERNELS.matvec.workgroupSize,
  subgroupSize: KERNELS.matvec.subgroupSize,
  rowsPerSubgroup: KERNELS.matvec.rowsPerSubgroup,
  subgroupsPerRow: KERNELS.matvec.subgroupsPerRow,
};

const WIDE_GEOMETRY: MatvecGeometry = {
  workgroupSize: KERNELS.matvecWide.workgroupSize,
  subgroupSize: KERNELS.matvecWide.subgroupSize,
  rowsPerSubgroup: KERNELS.matvecWide.rowsPerSubgroup,
  subgroupsPerRow: KERNELS.matvecWide.subgroupsPerRow,
};

// B1.3's occupancy assignment, per site, carried unchanged. This is what the baseline arm runs and
// what every site outside N1's reach runs on every arm; it is the reason A2 here is the same
// encoded command buffer B1.3 published.
export const CARRIED_GEOMETRY: Readonly<Record<MatvecSite, MatvecGeometry>> = {
  conv_in_proj: BASE_GEOMETRY,
  conv_out_proj: BASE_GEOMETRY,
  attn_qkv: WIDE_GEOMETRY,
  attn_o_proj: WIDE_GEOMETRY,
  mlp_gate_up: WIDE_GEOMETRY,
  mlp_down: WIDE_GEOMETRY,
  head: BASE_GEOMETRY,
};

export type KneeTable = Readonly<Record<MatvecShape, MatvecGeometry>>;

// The stand-in knee, used only where the plan's occupancy cannot matter: the blob layout, the
// repack set and the dispatch counts are all geometry-independent, and every timed plan is built
// after P0a has published the measured table. It is the geometry B1.3 ran each shape at, so a plan
// built with it is B1.3's plan.
export const CARRIED_KNEE: KneeTable = {
  attn_o_proj: WIDE_GEOMETRY,
  mlp_down: WIDE_GEOMETRY,
  attn_qkv: WIDE_GEOMETRY,
  conv_in_proj: BASE_GEOMETRY,
  mlp_gate_up: WIDE_GEOMETRY,
  head: BASE_GEOMETRY,
};

// The knee table the node stage measured, as the plan reads it. The argument is the flat form a
// stage file and a runtime config carry the table in, taken structurally so the validated record
// type stays in the schemas and this module stays free of them. A shape missing from the entries is
// a P0a table that did not cover the plan, which is a build failure and not a fallback.
export function kneeTableOf(
  entries: readonly { readonly shape: MatvecShape; readonly geometry: MatvecGeometry }[],
): KneeTable {
  const found = new Map<MatvecShape, MatvecGeometry>();
  for (const entry of entries) {
    if (found.has(entry.shape)) {
      throw new Error(`knee table: ${entry.shape} appears twice`);
    }
    found.set(entry.shape, entry.geometry);
  }
  const cell = (shape: MatvecShape): MatvecGeometry => {
    const geometry = found.get(shape);
    if (geometry === undefined) {
      throw new Error(`knee table: no measured cell for ${shape}`);
    }
    return geometry;
  };
  return {
    attn_o_proj: cell("attn_o_proj"),
    mlp_down: cell("mlp_down"),
    attn_qkv: cell("attn_qkv"),
    conv_in_proj: cell("conv_in_proj"),
    mlp_gate_up: cell("mlp_gate_up"),
    head: cell("head"),
  };
}

// One measured optimum from P0c: the block count S that minimised core time at that many total
// positions. Depths P0c did not measure fall back to the registered rule.
export interface FlashOptimum {
  readonly positions: number;
  readonly blocks: number;
}

export interface PlanTuning {
  readonly knee: KneeTable;
  readonly flashOptima: readonly FlashOptimum[];
}

// The tuning a stage uses before P0a and P0c have published theirs.
export const CARRIED_TUNING: PlanTuning = { knee: CARRIED_KNEE, flashOptima: [] };

export function flashBlocksFor(tuning: PlanTuning, totalPositions: number): number {
  const measured = tuning.flashOptima.find((entry) => entry.positions === totalPositions);
  if (measured !== undefined) {
    if (!Number.isInteger(measured.blocks) || measured.blocks < 1 || measured.blocks > LEVERS.flash.maxBlocks) {
      throw new Error(`flashBlocksFor: P0c optimum ${measured.blocks} at ${totalPositions} positions is outside 1..${LEVERS.flash.maxBlocks}`);
    }
    return measured.blocks;
  }
  return blocksForPositions(totalPositions, LEVERS.flash.positionsPerBlockRule, LEVERS.flash.maxBlocks);
}

// The occupancy half of the plan: B1.3's carried assignment per site, and P0a's measured knee per
// shape cell. Every stage reads the same pair, and a stage that has no measured knee must say so by
// passing CARRIED_KNEE rather than by omitting it.
export function planGeometry(knee: KneeTable): PlanGeometry {
  for (const shape of MATVEC_SHAPES) {
    const cell = knee[shape];
    if (cell.workgroupSize % cell.subgroupSize !== 0) {
      throw new Error(`planGeometry: knee for ${shape} has workgroup ${cell.workgroupSize} against subgroup ${cell.subgroupSize}`);
    }
  }
  return {
    carried: CARRIED_GEOMETRY,
    knee,
    blockWorkgroupSize: LEVERS.mlpFused.workgroupSize,
    attnCoreWorkgroupSize: LEVERS.attnSplit.coreWorkgroupSize,
    flashCoreWorkgroupSize: LEVERS.flash.coreWorkgroupSize,
    mergeWorkgroupSize: LEVERS.flash.mergeWorkgroupSize,
    convWorkgroupSize: KERNELS.conv.workgroupSize,
    embedWorkgroupSize: KERNELS.embed.workgroupSize,
    sampleWorkgroupSize: KERNELS.sample.workgroupSize,
    foldGeometry: { workgroupSize: LEVERS.fold.workgroupSize, threadsPerRow: LEVERS.fold.threadsPerRow },
  };
}

export const PLAN_GEOMETRY: PlanGeometry = planGeometry(CARRIED_KNEE);

export function planFlashFor(tuning: PlanTuning, totalPositions: number): PlanFlash {
  return {
    blocks: flashBlocksFor(tuning, totalPositions),
    maxBlocks: LEVERS.flash.maxBlocks,
    partialWords: LEVERS.flash.partialWords,
  };
}

// `position` is the cache depth this plan prices bytes at. `flashPosition` is the depth N2's block
// count was chosen at, which is NOT the same number: S is fixed for a whole run -- one encoded
// command buffer serves every token in it -- while the bytes the attention core streams grow token
// by token inside that run. Passing the two separately is what lets the byte accounting be re-done
// at the depth a dispatch actually ran without silently re-choosing an occupancy the encoder never
// used.
export function planOptionsFor(
  shape: ModelShape,
  queries: number,
  position: number,
  flashPosition: number,
  arm: ArmKey,
  tuning: PlanTuning,
): PlanOptions {
  return {
    queries,
    position,
    precision: {
      residualBytes: PRECISION.residualBytes,
      hiddenBytes: PRECISION.hiddenBytes,
      kvCacheBytes: PRECISION.kvCacheBytes,
      convCacheBytes: PRECISION.convCacheBytes,
      logitsBytes: PRECISION.logitsBytes,
    },
    levers: leversFor(arm),
    slices: planSlicesFor(shape),
    // S is looked up at the REGISTERED depth, not at position + queries. P0c measures its optimum at
    // 192, 1024 and 2048 attended positions and the gates name the same three numbers, so asking the
    // table with 193 would miss every measured cell and fall through to the rule at every depth.
    flash: planFlashFor(tuning, flashPosition),
    geometry: planGeometry(tuning.knee),
    samplePartials: PLAN_SAMPLE_PARTIALS,
  };
}

// What a runtime needs to build one arm. It lives here, with the plan options, so no stage can hand
// the runtime a plan the gates were not registered against.
export interface ArmPlanSpec {
  readonly key: ArmKey;
  readonly options: PlanOptions;
}

// Every registered arm's plan options, in registered order. The runtime builds one step list per
// entry at init, so switching arms between blocks costs nothing and every arm decodes the plan the
// gates were registered against.
export function armPlanOptions(
  shape: ModelShape,
  position: number,
  flashPosition: number,
  tuning: PlanTuning,
): readonly ArmPlanSpec[] {
  return ARMS.map((arm) => ({ key: arm.key, options: planOptionsFor(shape, 1, position, flashPosition, arm.key, tuning) }));
}

// The K-sliced copies the fused arms need. Taken from the baseline arm, the only structure in this
// unit that splits K at all: the fused MLP's down projection at 80 slices. The N1 arms defer no sum
// and read every matrix through its unpermuted section, so they add nothing here. Geometry does not
// enter a repack, so the carried tuning is the right one to derive it with.
export function repacksFor(shape: ModelShape): readonly RepackSpec[] {
  return repackSpecs(shape, planOptionsFor(shape, 1, PROMPT.warmPosition, PROMPT.warmPosition, "A2", CARRIED_TUNING));
}

// The one blob layout every arm binds: the layout arm's sections at the layout arm's offsets, then
// the repacks, then the extended rotary tables. A2N1 reads every matrix through its own dispatch
// and fuses nothing, so its section set is B1.1's -- which is what makes B1.1's blob a byte prefix
// of this one.
export function blobLayout(shape: ModelShape): Layout {
  return weightLayout(
    shape,
    planOptionsFor(shape, 1, PROMPT.warmPosition, PROMPT.warmPosition, LAYOUT_ARM, CARRIED_TUNING),
    RUNTIME.ropeRows,
    RUNTIME.ropeRowsExtended,
    RUNTIME.sectionAlignment,
    repacksFor(shape),
  );
}
