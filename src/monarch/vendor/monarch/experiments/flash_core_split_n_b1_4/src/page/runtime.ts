// The arm-aware fused runtime. One set of device buffers, one pipeline cache, and one step list per
// registered arm, all built at init; `setArm` chooses which list a token encodes. Nothing in the
// step loop creates a buffer, writes a buffer, or maps memory, so a run of N tokens is N command
// buffers submitted back to back with no host gap between them.
//
// Two device facts force the shape of this file.
//
// The adapter allows ten storage buffers per shader stage, so a fused block cannot bind one buffer
// per matrix. It binds the whole weight blob once and indexes it with u32 word offsets baked into
// its WGSL, which makes every fused pipeline specific to its layer. The sections are 256 B aligned,
// so `offset / 4` is exact and f32 constants come back through `bitcast<f32>`.
//
// A fused block's deferred reduction ping-pongs both the residual and the partials, so in a mixed
// arm the *same* R1 dispatch reads residual parity 0 in one layer and parity 1 in another,
// depending on how many fused blocks precede it in that arm. Bind groups are therefore built per
// arm; pipelines are still shared through the cache, keyed by everything that changes their code.
import { KERNELS, LEVERS, PRECISION, RUNTIME, SHAPE } from "../config.ts";
import { planStep, type FusedWeight, type PlannedDispatch, type PlanOptions } from "../pure/plan.ts";
import { maxSlicesOf, planSlicesFor } from "../pure/plan_options.ts";
import type { ModelShape } from "../pure/shape.ts";
import {
  attnCoreFlashKernel,
  attnCoreFlashWorkgroups,
  attnCoreQkvKernel,
  attnCoreQkvWorkgroups,
  attnMergeKernel,
  attnMergeWorkgroups,
  flashRecordBytes,
  foldKernel,
  foldWorkgroups,
  fusedBlockWorkgroups,
  mlpFusedKernel,
  type AttnCoreOffsets,
  type FusedGeometry,
  type MlpBlockOffsets,
} from "./fused_wgsl.ts";
import {
  convKernel,
  convWorkgroups,
  embedKernel,
  embedWorkgroups,
  matvecKernel,
  matvecWorkgroups,
  samplePartialKernel,
  samplePartialWorkgroups,
  sampleFinalKernel,
  sampleFinalWorkgroups,
  withMaxCols,
  type BindingName,
  type BindingSlot,
  type DequantVariant,
  type KernelSource,
  type MatvecMode,
  type RuntimeShape,
  type Workgroups,
} from "./wgsl.ts";

export interface Section {
  readonly offset: number;
  readonly bytes: number;
}

export type SectionTable = ReadonlyMap<string, Section>;

export interface RuntimeState {
  readonly position: number;
  readonly step: number;
  readonly queries: number;
  // 0 = argmax, 1 = Gumbel-max at T = 1. Both are exact; greedy is the correctness arm.
  readonly sampleMode: number;
  readonly rng: number;
  // 1 = feed the sampled id to the next step (free running), 0 = teacher forced from `tokens`.
  readonly freerun: number;
  readonly advance: number;
}

export const STATE_WORDS = 8;

interface Step {
  readonly plan: PlannedDispatch;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly workgroups: (queries: number) => Workgroups;
}

// One arm: the plan it was built from, its bind groups, and whether it carries a fused block. A
// fused arm is decode-only — its blocks are written for one query token — so a multi-query encode
// on it is a programming error, not a slow path.
interface Arm {
  readonly key: string;
  readonly plan: readonly PlannedDispatch[];
  readonly steps: readonly Step[];
  readonly fused: boolean;
}

// The plan a runtime should carry for one arm. Every stage builds these from `planOptionsFor`, so
// no stage can encode a different plan than the one P0 priced.
export interface ArmPlan {
  readonly key: string;
  readonly options: PlanOptions;
}

export interface FusedRuntime {
  readonly shape: ModelShape;
  readonly armKeys: readonly string[];
  readonly arm: string;
  readonly steps: readonly Step[];
  readonly dispatchCount: number;
  readonly pipelineCount: number;
  readonly bindGroupCount: number;
  readonly weightBytes: number;
  readonly liveGpuBytes: number;
  setArm(key: string): void;
  dispatchCountOf(key: string): number;
  planOf(key: string): readonly PlannedDispatch[];
  writeState(state: RuntimeState): void;
  writeTokens(ids: Uint32Array, at: number): void;
  encodeToken(encoder: GPUCommandEncoder, queries: number, timestamps: GPUComputePassTimestampWrites | null): void;
  encodePerDispatchPasses(encoder: GPUCommandEncoder, queries: number, querySet: GPUQuerySet, firstQuery: number): number;
  // One dispatch on its own, by plan name. The dequant check runs a single weighted dispatch
  // against a written residual, which is the only place a partial token is encoded.
  stepIndex(name: string): number;
  encodeOne(encoder: GPUCommandEncoder, index: number, queries: number): void;
  // N2's tile trip count is baked into the kernel from the largest total position count this
  // runtime was built for, so a run that walked past it would silently drop the positions past the
  // last tile. Every caller that knows where its run ends says so here, before it starts.
  assertReach(totalPositions: number): void;
  clearCaches(): void;
  clearResidual(): void;
  readonly buffers: Readonly<
    Record<
      | "state"
      | "tokens"
      | "sampled"
      | "logits"
      | "residual"
      | "residualB"
      | "deferredA"
      | "deferredB"
      | "partials"
      | "proj"
      | "qkvProj"
      | "flash"
      | "mid",
      GPUBuffer
    >
  >;
  destroy(): void;
}

function bufferBinding(buffer: GPUBuffer, section: Section): GPUBufferBinding {
  if (section.offset % 256 !== 0) {
    throw new Error(`runtime: section offset ${section.offset} is not 256-aligned`);
  }
  return { buffer, offset: section.offset, size: section.bytes };
}

function layoutFor(device: GPUDevice, kernel: KernelSource, label: string): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label,
    entries: kernel.bindings.map((slot) => ({
      binding: slot.binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: slot.access === "read" ? ("read-only-storage" as const) : ("storage" as const) },
    })),
  });
}

function pipelineFor(device: GPUDevice, kernel: KernelSource, label: string): { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout } {
  const layout = layoutFor(device, kernel, `${label}.layout`);
  const module = device.createShaderModule({ code: kernel.code, label });
  const pipeline = device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: "main" },
  });
  return { pipeline, layout };
}

export interface BlockGeometry {
  readonly workgroupSize: number;
  // The fused block closes its row sums with subgroup adds, so the lane count is part of its
  // geometry rather than a device fact it can look up.
  readonly subgroupSize: number;
  readonly slices: number;
}

export interface FoldGeometry {
  readonly workgroupSize: number;
  readonly threadsPerRow: number;
}

export interface RuntimeOptions {
  readonly maxQueries: number;
  readonly maxPositions: number;
  // The occupancy axis is no longer here. Every matvec dispatch carries the geometry it runs at in
  // its own plan entry -- B1.3's carried assignment on the sites N1 does not touch, P0a's measured
  // knee on the sites it does -- so the encoder reads occupancy off the plan it is asserting
  // against instead of choosing between two registered constants.
  //
  // The one bound that is the kernel's and not the plan's: the widest activation row a matvec
  // stages in workgroup memory.
  readonly matvecMaxCols: number;
  // N3. P0b's adopted inner product, applied to every arm at once -- it is a kernel-internal
  // substitution, not a lever, so no arm may run a different one than another.
  readonly dequantVariant: DequantVariant;
  readonly samplePartials: number;
  readonly sampleWorkgroup: number;
  readonly embedWorkgroup: number;
  readonly convWorkgroup: number;
  // A2's core: one workgroup per KV head per query, sized so a whole head row fits its lanes.
  readonly attnCoreWorkgroup: number;
  // N2's core: one workgroup per (KV head, position block), and its merge: one per q head. Both
  // are one thread per head dimension.
  readonly flashCoreWorkgroup: number;
  readonly mergeWorkgroup: number;
  // The largest total position count -- cached positions plus the token being decoded -- any run
  // on this runtime reaches. N2's tile trip count is a compile-time constant of its kernel, so it
  // is baked from this; a shorter token runs the tiles it does not need as zero-length staging
  // loops. It is a property of the session, identical on every arm.
  readonly flashMaxTotalPositions: number;
  // Reproduces SkipSimplifiedLayerNormalization(x, x) at the head exactly.
  readonly headPreScale: number;
  readonly mlpFused: BlockGeometry;
  readonly fold: FoldGeometry;
  // Rows of the partial buffer: the largest slice count any single dispatch publishes.
  readonly maxSlices: number;
}

export function runtimeShapeOf(shape: ModelShape, options: RuntimeOptions): RuntimeShape {
  return {
    hidden: shape.hidden,
    ffn: shape.ffn,
    vocab: shape.vocab,
    heads: shape.heads,
    kvHeads: shape.kvHeads,
    headDim: shape.headDim,
    convTaps: shape.convCache,
    quantBlock: shape.quantBlock,
    epsilon: SHAPE.normEps,
    maxPositions: options.maxPositions,
    maxQueries: options.maxQueries,
  };
}

// Every distinct kernel source is compiled once and shared by every dispatch that generates the
// same code. R1's key is (kind, rows, cols, zeroPoints); a fused block's is (kind, layer,
// foldSlices, embed), because its weight offsets are baked into the shader.
class PipelineCache {
  private readonly entries = new Map<string, { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout; kernel: KernelSource }>();

  constructor(private readonly device: GPUDevice) {}

  get(key: string, build: () => KernelSource): { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout; kernel: KernelSource } {
    const found = this.entries.get(key);
    if (found !== undefined) {
      return found;
    }
    const kernel = build();
    const { pipeline, layout } = pipelineFor(this.device, kernel, key);
    const entry = { pipeline, layout, kernel };
    this.entries.set(key, entry);
    return entry;
  }

  get size(): number {
    return this.entries.size;
  }
}

export function createRuntime(
  device: GPUDevice,
  shape: ModelShape,
  weights: GPUBuffer,
  weightBytes: number,
  sections: SectionTable,
  armPlans: readonly ArmPlan[],
  options: RuntimeOptions,
): FusedRuntime {
  if (armPlans.length === 0) {
    throw new Error("runtime: no arms to build");
  }
  const runtimeShape = runtimeShapeOf(shape, options);
  const { maxQueries, maxPositions } = options;
  const attnLayers = shape.layerTypes.filter((kind) => kind === "attention").length;
  const convLayers = shape.layerTypes.length - attnLayers;

  const section = (name: string): Section => {
    const found = sections.get(name);
    if (found === undefined) {
      throw new Error(`runtime: weight section ${name} missing`);
    }
    return found;
  };

  // A fused block indexes the blob as `array<u32>`, so every offset it bakes must be a whole word.
  const word = (name: string): number => {
    const found = section(name);
    if (found.offset % 4 !== 0) {
      throw new Error(`runtime: section ${name} at ${found.offset} is not word aligned`);
    }
    return found.offset / 4;
  };

  // The attention dispatch reads one contiguous range holding [operator_norm, q_norm, k_norm];
  // the layout places them adjacent and 256-aligned, and this asserts it rather than assuming it.
  const normTriple = (base: string): Section => {
    const norm = section(`${base}.norm`);
    const qNorm = section(`${base}.q_norm`);
    const kNorm = section(`${base}.k_norm`);
    if (qNorm.offset !== norm.offset + norm.bytes || kNorm.offset !== qNorm.offset + qNorm.bytes) {
      throw new Error(`runtime: ${base} norm/q_norm/k_norm are not contiguous`);
    }
    return { offset: norm.offset, bytes: norm.bytes + qNorm.bytes + kNorm.bytes };
  };

  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const make = (bytes: number, label: string): GPUBuffer => device.createBuffer({ size: bytes, usage: storage, label });

  const residualBytes = maxQueries * shape.hidden * PRECISION.residualBytes;
  const deferredBytes = options.maxSlices * shape.hidden * PRECISION.residualBytes;
  const projBytes = maxQueries * 3 * shape.hidden * PRECISION.hiddenBytes;
  const midBytes = maxQueries * Math.max(shape.hidden, shape.ffn) * PRECISION.hiddenBytes;
  // A2's q/k/v scratch, at accumulate precision: the projection writes f32 rows and the core reads
  // them, applies the two head norms and the rotation, and rounds once on the way into the f16
  // cache. Sized by the projection's row count, which is q rows plus one k and one v row per KV
  // head -- 2048 rows here, not 3 * hidden, because k and v are not broadcast to q's width.
  const qkvProjBytes = maxQueries * (shape.hidden + 2 * shape.kvHeads * shape.headDim) * 4;
  const kvLayerBytes = shape.kvHeads * maxPositions * shape.headDim * PRECISION.kvCacheBytes;
  const convLayerBytes = shape.hidden * shape.convCache * PRECISION.convCacheBytes;
  const logitBytes = maxQueries * shape.vocab * PRECISION.logitsBytes;
  // N2's partial store, sized by the largest block count any arm on this runtime plans. It is
  // allocated on every runtime, not only the ones carrying a flash arm: at S = 1 it is 4 352 B,
  // and making it conditional would make an arm's live-GPU-bytes depend on which OTHER arms the
  // session happened to build.
  const flashBlocks = armPlans.reduce((most, armPlan) => Math.max(most, armPlan.options.flash.blocks), 1);
  const flashBytes = flashRecordBytes(runtimeShape, flashBlocks, maxQueries);
  const samplePairs = options.samplePartials;
  if (kvLayerBytes % 256 !== 0 || convLayerBytes % 256 !== 0) {
    throw new Error("runtime: per-layer cache stride is not 256-aligned");
  }

  const residual = make(residualBytes, "residual_a");
  const residualB = make(residualBytes, "residual_b");
  const deferredA = make(deferredBytes, "deferred_a");
  const deferredB = make(deferredBytes, "deferred_b");
  const proj = make(projBytes, "proj");
  const mid = make(midBytes, "mid");
  const qkvProj = make(qkvProjBytes, "qkv_proj");
  const keyCache = make(attnLayers * kvLayerBytes, "key_cache");
  const valueCache = make(attnLayers * kvLayerBytes, "value_cache");
  const convCache = make(convLayers * convLayerBytes, "conv_cache");
  const logits = make(logitBytes, "logits");
  const state = make(STATE_WORDS * 4, "state");
  const tokens = make(maxPositions * 4, "tokens");
  const sampled = make(maxPositions * 4, "sampled");
  const partials = make(samplePairs * 2 * 4, "partials");
  const flash = make(flashBytes, "flash");

  const whole = (buffer: GPUBuffer): GPUBufferBinding => ({ buffer, offset: 0, size: buffer.size });
  const cacheSlice = (buffer: GPUBuffer, index: number, stride: number): GPUBufferBinding => ({
    buffer,
    offset: index * stride,
    size: stride,
  });
  const parity = (which: number, a: GPUBuffer, b: GPUBuffer, label: string): GPUBuffer => {
    if (which === 0) {
      return a;
    }
    if (which === 1) {
      return b;
    }
    throw new Error(`runtime: ${label} parity ${which} is not 0 or 1`);
  };
  const residualOf = (which: number): GPUBufferBinding => whole(parity(which, residual, residualB, "residual"));
  const deferredOf = (which: number): GPUBufferBinding => whole(parity(which, deferredA, deferredB, "partials"));

  const cache = new PipelineCache(device);

  const bind = (
    layout: GPUBindGroupLayout,
    slots: readonly BindingSlot[],
    resources: Partial<Record<BindingName, GPUBufferBinding>>,
    label: string,
  ): GPUBindGroup => {
    const entries: GPUBindGroupEntry[] = slots.map((slot) => {
      const resource = resources[slot.name];
      if (resource === undefined) {
        throw new Error(`runtime: ${label} has no resource for ${slot.name}`);
      }
      return { binding: slot.binding, resource };
    });
    return device.createBindGroup({ layout, entries, label });
  };

  const mlpOffsets = (layer: number): MlpBlockOffsets => ({
    norm: word(`L${layer}.mlp.gate_up.norm`),
    gateUpQuant: word(`L${layer}.mlp.gate_up.quant`),
    gateUpScales: word(`L${layer}.mlp.gate_up.scales`),
    downKQuant: word(`L${layer}.mlp.down.kquant`),
    downKScales: word(`L${layer}.mlp.down.kscales`),
  });

  // Both cores read the two head norms and the rotary tables out of the blob and nothing else:
  // their q, k and v arrive through the projection scratch, and their output leaves through `mid`
  // -- through the flash partials and then the merge, on the arms that carry N2.
  //
  // The extended tables, not B1.1's 1024-row prefix: the sweep decodes at 2048 and a shorter table
  // would return zeros for every position past its end instead of failing. Their first 1024 rows
  // are byte identical to the prefix -- the weights stage asserts it -- so nothing below 1024 moves.
  const attnCoreOffsets = (layer: number): AttnCoreOffsets => ({
    qNorm: word(`L${layer}.attn.qkv.q_norm`),
    kNorm: word(`L${layer}.attn.qkv.k_norm`),
    cos: word("rope.cos_ext"),
    sin: word("rope.sin_ext"),
  });

  const blockGeometry = (block: BlockGeometry, dispatch: PlannedDispatch): FusedGeometry => {
    if (dispatch.fusion.slices !== block.slices) {
      throw new Error(
        `runtime: ${dispatch.name} publishes ${dispatch.fusion.slices} slices, the kernel is built for ${block.slices}`,
      );
    }
    if (block.slices > options.maxSlices) {
      throw new Error(`runtime: ${dispatch.name} publishes ${block.slices} slices into a ${options.maxSlices}-row partial buffer`);
    }
    return {
      workgroupSize: block.workgroupSize,
      subgroupSize: block.subgroupSize,
      slices: block.slices,
      foldSlices: dispatch.fusion.foldSlices,
    };
  };

  // Every matmul kind carries its matrix; the plan types it nullable because the fused blocks and
  // the non-matmul dispatches do not.
  const weightOf = (dispatch: PlannedDispatch): FusedWeight => {
    if (dispatch.weight === null) {
      throw new Error(`runtime: ${dispatch.name} is a matmul with no weight`);
    }
    return dispatch.weight;
  };

  // ---- one arm ----

  const buildArm = (armPlan: ArmPlan): Arm => {
    const plan = planStep(shape, armPlan.options);
    const armQueries = armPlan.options.queries;
    // S, and the reach its tile count is baked for. S belongs to the arm -- it is chosen per
    // context length from P0c's curve -- and the reach belongs to the session.
    const splits = armPlan.options.flash.blocks;
    const flashReach = options.flashMaxTotalPositions;
    const steps: Step[] = [];
    let attnIndex = 0;
    let convIndex = 0;
    let fused = false;

    // Every dispatch goes through here, so the occupancy the plan priced and the occupancy the
    // runtime encodes are asserted equal once, for all fourteen kinds. This unit's levers move
    // exactly that number, and a lever that moved it in the plan but not in the encoder would be
    // priced against a dispatch that never ran.
    const record = (
      dispatch: PlannedDispatch,
      entry: { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout; kernel: KernelSource },
      resources: Partial<Record<BindingName, GPUBufferBinding>>,
      workgroups: (queries: number) => Workgroups,
    ): void => {
      if (entry.kernel.workgroupSize !== dispatch.workgroupSize) {
        throw new Error(
          `runtime: ${dispatch.name} is planned at workgroup size ${dispatch.workgroupSize}, its kernel is ${entry.kernel.workgroupSize}`,
        );
      }
      const [x, y, z] = workgroups(armQueries);
      if (x * y * z !== dispatch.workgroups) {
        throw new Error(`runtime: ${dispatch.name} is planned at ${dispatch.workgroups} workgroups, the runtime encodes ${x * y * z}`);
      }
      steps.push({
        plan: dispatch,
        pipeline: entry.pipeline,
        bindGroup: bind(entry.layout, entry.kernel.bindings, resources, `${armPlan.key}.${dispatch.name}`),
        workgroups,
      });
    };

    const matvecStep = (
      dispatch: PlannedDispatch,
      mode: MatvecMode,
      rows: number,
      cols: number,
      zeroPoints: boolean,
      preScale: number,
      quantName: string,
      resources: Partial<Record<BindingName, GPUBufferBinding>>,
    ): void => {
      // The plan states the occupancy; the runtime never chooses it. N1 IS a different geometry on
      // the same bytes, so a lever that moved the plan's number without moving the encoded dispatch
      // would be priced against a dispatch that never ran -- which is what `record` then asserts.
      if (dispatch.site === null || dispatch.geometry === null) {
        throw new Error(`runtime: ${dispatch.name} is a matvec, so the plan must name the site and the geometry it runs at`);
      }
      const matvecGeometry = withMaxCols(dispatch.geometry, options.matvecMaxCols);
      const { workgroupSize, subgroupSize, rowsPerSubgroup, subgroupsPerRow } = matvecGeometry;
      const occupancy = `w${workgroupSize}s${subgroupSize}r${rowsPerSubgroup}c${subgroupsPerRow}`;
      const key = `${mode}_${rows}x${cols}${zeroPoints ? "_zp" : ""}_${occupancy}_${options.dequantVariant}`;
      const entry = cache.get(key, () =>
        matvecKernel(mode, rows, cols, zeroPoints, matvecGeometry, runtimeShape, preScale, options.dequantVariant),
      );
      const full: Partial<Record<BindingName, GPUBufferBinding>> = {
        weights: bufferBinding(weights, section(`${quantName}.quant`)),
        scales: bufferBinding(weights, section(`${quantName}.scales`)),
        state: whole(state),
        partials: whole(partials),
        ...resources,
      };
      if (zeroPoints) {
        full.zero_points = bufferBinding(weights, section(`${quantName}.zero_points`));
      }
      record(dispatch, entry, full, (queries) => matvecWorkgroups(mode, rows, runtimeShape, matvecGeometry, queries));
    };

    const blockStep = (
      dispatch: PlannedDispatch,
      key: string,
      block: BlockGeometry,
      build: (geometry: FusedGeometry) => KernelSource,
      resources: Partial<Record<BindingName, GPUBufferBinding>>,
    ): void => {
      const geometry = blockGeometry(block, dispatch);
      const entry = cache.get(key, () => build(geometry));
      const full: Partial<Record<BindingName, GPUBufferBinding>> = {
        blob: whole(weights),
        residual_in: residualOf(dispatch.fusion.residualIn),
        residual_out: residualOf(dispatch.fusion.residualOut),
        partials_in: deferredOf(dispatch.fusion.partialsIn),
        partials_out: deferredOf(dispatch.fusion.partialsOut),
        state: whole(state),
        ...resources,
      };
      record(dispatch, entry, full, () => fusedBlockWorkgroups(block.slices));
      fused = true;
    };

    for (const dispatch of plan) {
      const f = dispatch.fusion;
      switch (dispatch.kind) {
        case "embed": {
          const entry = cache.get("embed", () => embedKernel(runtimeShape, options.embedWorkgroup));
          record(
            dispatch,
            entry,
            {
              weights: bufferBinding(weights, section("embed.quant")),
              scales: bufferBinding(weights, section("embed.scales")),
              zero_points: bufferBinding(weights, section("embed.zero_points")),
              residual: residualOf(f.residualOut),
              state: whole(state),
              tokens: whole(tokens),
            },
            (queries) => embedWorkgroups(queries),
          );
          break;
        }
        case "norm_matvec": {
          const weight = weightOf(dispatch);
          matvecStep(dispatch, "norm_matvec", weight.rows, weight.cols, weight.zeroPoints, 1, dispatch.name, {
            gamma: bufferBinding(weights, section(`${dispatch.name}.norm`)),
            residual: residualOf(f.residualIn),
            destination: whole(proj),
          });
          break;
        }
        case "conv_core": {
          const entry = cache.get("conv_core", () => convKernel(runtimeShape, options.convWorkgroup));
          record(
            dispatch,
            entry,
            {
              gamma: bufferBinding(weights, section(`${dispatch.name}.taps`)),
              source: whole(proj),
              destination: whole(mid),
              state: whole(state),
              cache: cacheSlice(convCache, convIndex, convLayerBytes),
            },
            () => convWorkgroups(runtimeShape, options.convWorkgroup),
          );
          convIndex += 1;
          break;
        }
        case "matvec_residual": {
          const weight = weightOf(dispatch);
          matvecStep(dispatch, "matvec_residual", weight.rows, weight.cols, weight.zeroPoints, 1, dispatch.name, {
            residual: residualOf(f.residualOut),
            source: whole(mid),
          });
          break;
        }
        case "norm_matvec_swiglu": {
          const weight = weightOf(dispatch);
          matvecStep(dispatch, "norm_swiglu", weight.rows, weight.cols, weight.zeroPoints, 1, dispatch.name, {
            gamma: bufferBinding(weights, section(`${dispatch.name}.norm`)),
            residual: residualOf(f.residualIn),
            destination: whole(mid),
          });
          break;
        }
        // A2's projection: the same split-N matvec every other projection compiles to, at the wide
        // geometry, writing f32 q/k/v rows instead of an f16 activation. It carries the operator
        // norm and nothing else -- the rotation and the two head norms belong to the core.
        case "attn_proj": {
          const weight = weightOf(dispatch);
          matvecStep(dispatch, "norm_projection", weight.rows, weight.cols, weight.zeroPoints, 1, dispatch.name, {
            gamma: bufferBinding(weights, section(`${dispatch.name}.norm`)),
            residual: residualOf(f.residualIn),
            projection: whole(qkvProj),
          });
          break;
        }
        case "attn_core_qkv": {
          const layer = dispatch.layer;
          const entry = cache.get(`attn_core_qkv_L${layer}`, () =>
            attnCoreQkvKernel(runtimeShape, attnCoreOffsets(layer), options.attnCoreWorkgroup),
          );
          record(
            dispatch,
            entry,
            {
              blob: whole(weights),
              state: whole(state),
              key_cache: cacheSlice(keyCache, attnIndex, kvLayerBytes),
              value_cache: cacheSlice(valueCache, attnIndex, kvLayerBytes),
              projection: whole(qkvProj),
              destination: whole(mid),
            },
            (queries) => attnCoreQkvWorkgroups(runtimeShape, queries),
          );
          attnIndex += 1;
          break;
        }
        // N2's core. It binds the same six buffers A2's core does with `flash` in place of
        // `destination`, and it is the only dispatch in the unit whose kernel depends on how far
        // the run walks: its tile trip count is baked, so the pipeline key carries both S and the
        // reach it was baked for.
        case "attn_core_flash": {
          const layer = dispatch.layer;
          const entry = cache.get(`attn_core_flash_L${layer}_s${splits}_t${flashReach}`, () =>
            attnCoreFlashKernel(runtimeShape, attnCoreOffsets(layer), options.flashCoreWorkgroup, splits, flashReach),
          );
          record(
            dispatch,
            entry,
            {
              blob: whole(weights),
              state: whole(state),
              key_cache: cacheSlice(keyCache, attnIndex, kvLayerBytes),
              value_cache: cacheSlice(valueCache, attnIndex, kvLayerBytes),
              projection: whole(qkvProj),
              flash: whole(flash),
            },
            (queries) => attnCoreFlashWorkgroups(runtimeShape, splits, queries),
          );
          attnIndex += 1;
          break;
        }
        // The log-sum-exp fold that closes the split. One workgroup per q head, blocks summed in
        // ascending index order, so the merge is bitwise reproducible run to run.
        case "attn_merge": {
          const entry = cache.get(`attn_merge_s${splits}`, () =>
            attnMergeKernel(runtimeShape, options.mergeWorkgroup, splits),
          );
          record(
            dispatch,
            entry,
            { flash: whole(flash), destination: whole(mid) },
            (queries) => attnMergeWorkgroups(runtimeShape, queries),
          );
          break;
        }
        case "mlp_fused": {
          const layer = dispatch.layer;
          blockStep(
            dispatch,
            `mlp_fused_L${layer}_f${f.foldSlices}`,
            options.mlpFused,
            (geometry) => mlpFusedKernel(runtimeShape, mlpOffsets(layer), geometry),
            {},
          );
          break;
        }
        case "fold": {
          if (f.residualIn !== f.residualOut || f.partialsIn !== f.partialsOut) {
            throw new Error(`runtime: ${dispatch.name} is in place, so its parities must match`);
          }
          const entry = cache.get(`fold_${f.foldSlices}`, () =>
            foldKernel(runtimeShape, options.fold.workgroupSize, options.fold.threadsPerRow, f.foldSlices),
          );
          record(
            dispatch,
            entry,
            { residual_out: residualOf(f.residualOut), partials_in: deferredOf(f.partialsIn) },
            () => foldWorkgroups(runtimeShape, options.fold.workgroupSize, options.fold.threadsPerRow),
          );
          break;
        }
        case "norm_head": {
          const weight = weightOf(dispatch);
          matvecStep(dispatch, "norm_head", weight.rows, weight.cols, weight.zeroPoints, options.headPreScale, "head", {
            gamma: bufferBinding(weights, section("head.norm")),
            residual: residualOf(f.residualIn),
            logits: whole(logits),
          });
          break;
        }
        case "sample_partial": {
          const entry = cache.get(`sample_partial_${samplePairs}`, () =>
            samplePartialKernel(runtimeShape, options.sampleWorkgroup, samplePairs),
          );
          record(
            dispatch,
            entry,
            { state: whole(state), partials: whole(partials), logits: whole(logits) },
            () => samplePartialWorkgroups(samplePairs),
          );
          break;
        }
        case "sample_final": {
          const entry = cache.get(`sample_final_${samplePairs}`, () =>
            sampleFinalKernel(runtimeShape, options.sampleWorkgroup, samplePairs),
          );
          record(
            dispatch,
            entry,
            { state: whole(state), tokens: whole(tokens), partials: whole(partials), sampled: whole(sampled) },
            () => sampleFinalWorkgroups(),
          );
          break;
        }
      }
    }
    if (steps.length !== plan.length) {
      throw new Error(`runtime: arm ${armPlan.key} encoded ${steps.length} of ${plan.length} planned dispatches`);
    }
    if (fused && armQueries !== 1) {
      throw new Error(`runtime: arm ${armPlan.key} is fused, which is a single-query decode specialisation, and plans ${armQueries} queries`);
    }
    return { key: armPlan.key, plan, steps, fused };
  };

  const arms = new Map<string, Arm>();
  for (const armPlan of armPlans) {
    if (arms.has(armPlan.key)) {
      throw new Error(`runtime: arm ${armPlan.key} built twice`);
    }
    arms.set(armPlan.key, buildArm(armPlan));
  }
  const first = arms.get(armPlans[0]?.key ?? "");
  if (first === undefined) {
    throw new Error("runtime: the first arm did not build");
  }
  let active: Arm = first;

  const owned = [
    residual,
    residualB,
    deferredA,
    deferredB,
    proj,
    mid,
    qkvProj,
    keyCache,
    valueCache,
    convCache,
    logits,
    state,
    tokens,
    sampled,
    partials,
    flash,
  ];
  const liveGpuBytes = owned.reduce((sum, buffer) => sum + buffer.size, weightBytes);
  let bindGroupCount = 0;
  for (const arm of arms.values()) {
    bindGroupCount += arm.steps.length;
  }
  const stateWords = new Uint32Array(STATE_WORDS);

  // A fused block is written for one query token: its workgroup count is a slice count, not a
  // query count. Multi-query encodes belong to R1, and asking a fused arm for one is a bug.
  const stepsFor = (queries: number): readonly Step[] => {
    if (queries !== 1 && active.fused) {
      throw new Error(`runtime: arm ${active.key} is fused and encodes one query, not ${queries}`);
    }
    return active.steps;
  };

  return {
    shape,
    armKeys: [...arms.keys()],
    get arm(): string {
      return active.key;
    },
    get steps(): readonly Step[] {
      return active.steps;
    },
    get dispatchCount(): number {
      return active.steps.length;
    },
    pipelineCount: cache.size,
    bindGroupCount,
    weightBytes,
    liveGpuBytes,
    buffers: { state, tokens, sampled, logits, residual, residualB, deferredA, deferredB, partials, proj, qkvProj, flash, mid },
    setArm(key: string): void {
      const found = arms.get(key);
      if (found === undefined) {
        throw new Error(`runtime: no arm ${key}`);
      }
      active = found;
    },
    dispatchCountOf(key: string): number {
      const found = arms.get(key);
      if (found === undefined) {
        throw new Error(`runtime: no arm ${key}`);
      }
      return found.steps.length;
    },
    planOf(key: string): readonly PlannedDispatch[] {
      const found = arms.get(key);
      if (found === undefined) {
        throw new Error(`runtime: no arm ${key}`);
      }
      return found.plan;
    },
    writeState(next: RuntimeState): void {
      stateWords[0] = next.position;
      stateWords[1] = next.step;
      stateWords[2] = next.queries;
      stateWords[3] = next.sampleMode;
      stateWords[4] = next.rng;
      stateWords[5] = next.freerun;
      stateWords[6] = next.advance;
      stateWords[7] = 0;
      device.queue.writeBuffer(state, 0, stateWords);
    },
    writeTokens(ids: Uint32Array, at: number): void {
      device.queue.writeBuffer(tokens, at * 4, ids);
    },
    encodeToken(encoder: GPUCommandEncoder, queries: number, timestamps: GPUComputePassTimestampWrites | null): void {
      const pass = timestamps === null
        ? encoder.beginComputePass()
        : encoder.beginComputePass({ timestampWrites: timestamps });
      for (const step of stepsFor(queries)) {
        pass.setPipeline(step.pipeline);
        pass.setBindGroup(0, step.bindGroup);
        const [x, y, z] = step.workgroups(queries);
        pass.dispatchWorkgroups(x, y, z);
      }
      pass.end();
    },
    assertReach(totalPositions: number): void {
      if (!Number.isInteger(totalPositions) || totalPositions < 1) {
        throw new Error(`runtime: ${totalPositions} is not a positive total position count`);
      }
      if (totalPositions > options.flashMaxTotalPositions) {
        throw new Error(
          `runtime: a run reaching ${totalPositions} total positions was built for ${options.flashMaxTotalPositions}, ` +
            "so the split-position core would drop every position past its last tile",
        );
      }
      if (totalPositions > maxPositions) {
        throw new Error(`runtime: a run reaching ${totalPositions} total positions is past the ${maxPositions}-position cache`);
      }
    },
    stepIndex(name: string): number {
      const index = active.steps.findIndex((step) => step.plan.name === name);
      if (index < 0) {
        throw new Error(`runtime: arm ${active.key} has no planned dispatch named ${name}`);
      }
      return index;
    },
    encodeOne(encoder: GPUCommandEncoder, index: number, queries: number): void {
      const step = stepsFor(queries)[index];
      if (step === undefined) {
        throw new Error(`runtime: dispatch index ${index} out of range`);
      }
      const pass = encoder.beginComputePass();
      pass.setPipeline(step.pipeline);
      pass.setBindGroup(0, step.bindGroup);
      const [x, y, z] = step.workgroups(queries);
      pass.dispatchWorkgroups(x, y, z);
      pass.end();
    },
    // M2's per-dispatch instrument: one pass per dispatch so each gets its own timestamp pair.
    // This changes the pass structure, so its wall time is never reported as t_AR.
    encodePerDispatchPasses(encoder: GPUCommandEncoder, queries: number, querySet: GPUQuerySet, firstQuery: number): number {
      let query = firstQuery;
      for (const step of stepsFor(queries)) {
        const pass = encoder.beginComputePass({
          timestampWrites: { querySet, beginningOfPassWriteIndex: query, endOfPassWriteIndex: query + 1 },
        });
        pass.setPipeline(step.pipeline);
        pass.setBindGroup(0, step.bindGroup);
        const [x, y, z] = step.workgroups(queries);
        pass.dispatchWorkgroups(x, y, z);
        pass.end();
        query += 2;
      }
      return query - firstQuery;
    },
    clearCaches(): void {
      const encoder = device.createCommandEncoder();
      encoder.clearBuffer(keyCache);
      encoder.clearBuffer(valueCache);
      encoder.clearBuffer(convCache);
      encoder.clearBuffer(sampled);
      device.queue.submit([encoder.finish()]);
    },
    clearResidual(): void {
      const encoder = device.createCommandEncoder();
      encoder.clearBuffer(residual);
      encoder.clearBuffer(residualB);
      encoder.clearBuffer(deferredA);
      encoder.clearBuffer(deferredB);
      device.queue.submit([encoder.finish()]);
    },
    destroy(): void {
      for (const buffer of owned) {
        buffer.destroy();
      }
    },
  };
}

// The workgroup sizes are registered constants; the slice counts are the shape's, so the kernel a
// pipeline is built for and the plan that drives it are slicing the same way by construction.
//
// Three fields are measurements and cannot be registered here: P0b's adopted inner product, and
// the reach the split-position core's tiles are baked for, which is a property of the run rather
// than of the model. The caller supplies both; there is no stand-in for either.
export function registeredRuntimeOptions(
  shape: ModelShape,
  dequantVariant: DequantVariant,
  flashMaxTotalPositions: number,
): RuntimeOptions {
  const slices = planSlicesFor(shape);
  if (KERNELS.matvec.maxCols !== KERNELS.matvecWide.maxCols) {
    throw new Error(
      `registeredRuntimeOptions: the two registered geometries stage ${KERNELS.matvec.maxCols} and ` +
        `${KERNELS.matvecWide.maxCols} columns, but the stage width is a kernel bound and must be one number`,
    );
  }
  return {
    maxQueries: RUNTIME.maxQueries,
    maxPositions: KERNELS.maxPositions,
    matvecMaxCols: KERNELS.matvec.maxCols,
    dequantVariant,
    samplePartials: KERNELS.sample.partials,
    sampleWorkgroup: KERNELS.sample.workgroupSize,
    embedWorkgroup: KERNELS.embed.workgroupSize,
    convWorkgroup: KERNELS.conv.workgroupSize,
    attnCoreWorkgroup: LEVERS.attnSplit.coreWorkgroupSize,
    flashCoreWorkgroup: LEVERS.flash.coreWorkgroupSize,
    mergeWorkgroup: LEVERS.flash.mergeWorkgroupSize,
    flashMaxTotalPositions,
    headPreScale: RUNTIME.headPreScale,
    mlpFused: {
      workgroupSize: LEVERS.mlpFused.workgroupSize,
      subgroupSize: KERNELS.matvec.subgroupSize,
      slices: slices.mlp,
    },
    fold: LEVERS.fold,
    maxSlices: maxSlicesOf(shape),
  };
}
