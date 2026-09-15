// Driving an arm: prime the caches, then run tokens through the pre-recorded command buffers.
// One session holds every registered arm; `session.runtime.setArm` picks which step list a token
// encodes, so an ABAB interleave switches arms between blocks without rebuilding anything.
//
// The pipelined loop is the registered t_AR instrument (one command buffer per token, submitted
// back to back with no await between them, per-token times from the pass timestamps).
// `decodeSync` submits and awaits each token, which is B1.0's instrument shape and is reported
// only for the side-by-side table.
import { ARMS, DEVICE_MINIMUMS, FUSION, PRECISION, SHAPE } from "../config.ts";
import { attnPartCode } from "../../../flash_core_split_n_b1_4/src/pure/codes.ts";
import { planStep, type MatvecGeometry, type MatvecShape, type PlannedDispatch } from "./efficiency_plan.ts";
import {efficiencyPlan} from "./efficiency_plan.ts";
import {candidateOf} from "./plans.ts";
import { kneeTableOf, type PlanTuning } from "../../../flash_core_split_n_b1_4/src/pure/plan_options.ts";
import {armPlanOptions,candidateMaxSlices} from "./plans.ts";
import type { ModelShape } from "../../../flash_core_split_n_b1_4/src/pure/shape.ts";
import type { HookSnapshot, PerDispatch, RuntimeConfig, RuntimeReady } from "../schemas.ts";
import {
  deviceForRuntime,
  readBuffer,
  resolveTimestamps,
  waitIdle,
  withValidation,
  pageEnv,
  type DeviceNeeds,
  type Gpu,
} from "../../../flash_core_split_n_b1_4/src/page/device.ts";
import { createRuntime, registeredRuntimeOptions, type FusedRuntime, type SectionTable } from "./runtime.ts";
import { fetchWeights, sectionTableOf, type WeightBlob } from "../../../flash_core_split_n_b1_4/src/page/weights.ts";

export interface FusedSession {
  readonly gpu: Gpu;
  readonly runtime: FusedRuntime;
  readonly blob: WeightBlob;
  readonly shape: ModelShape;
  readonly sections: SectionTable;
  // The measured tuning the arm plans were built from, and the depth N2's block count was chosen
  // at. Both travel with the session so a later accounting re-plan cannot re-choose an occupancy
  // the encoder never ran.
  readonly tuning: PlanTuning;
  readonly planPosition: number;
  readonly errors: string[];
  destroy(): void;
}

export function tuningOf(config: RuntimeConfig): PlanTuning {
  return { knee: kneeTableOf(config.knee), flashOptima: config.flashOptima.map((entry) => ({ ...entry })) };
}

export function shapeOf(config: RuntimeConfig): ModelShape {
  return {
    layerTypes: config.shape.layerTypes,
    hidden: config.shape.hidden,
    ffn: config.shape.ffn,
    vocab: config.shape.vocab,
    heads: config.shape.heads,
    kvHeads: config.shape.kvHeads,
    headDim: config.shape.headDim,
    convCache: config.shape.convCache,
    quantBits: config.shape.quantBits,
    quantBlock: config.shape.quantBlock,
  };
}

// Every arm the session can run comes from the registry. There is no runtime-built arm in this
// unit: P3 sweeps context on an arm that was already measured, so the step list it times is
// byte-for-byte the one the M stage timed.
export function buildSession(
  gpu: Gpu,
  config: RuntimeConfig,
  blob: WeightBlob,
  errors: string[],
): FusedSession {
  const shape = shapeOf(config);
  const sections = sectionTableOf(config.sections);
  const tuning = tuningOf(config);
  const options = {...registeredRuntimeOptions(shape, config.dequantVariant, config.maxTotalPositions),maxSlices:candidateMaxSlices()};
  const runtime = createRuntime(
    gpu.device,
    shape,
    blob.buffer,
    blob.bytes,
    sections,
    // `planPosition` is the attended total; the plan's own position is one query below it, because
    // the core reads `position + queries` rows. Passing the attended count to both would build the
    // plan one row deep and look P0c's table up at a depth it never measured.
    armPlanOptions(shape, config.planPosition - config.maxQueries, config.planPosition, tuning),
    {
      ...options,
      maxQueries: config.maxQueries,
      maxPositions: config.maxPositions,
      samplePartials: config.samplePartials,
    },
  );
  return {
    gpu,
    runtime,
    blob,
    shape,
    sections,
    tuning,
    planPosition: config.planPosition,
    errors,
    destroy(): void {
      runtime.destroy();
      blob.buffer.destroy();
    },
  };
}

// The minima the runtime's own kernels impose on the device it lands on. The kernel geometry
// fixes all but two of them; those two depend on the weights, since the blob is one buffer and
// its largest section — or the logits buffer, in a narrow enough model — is the largest range
// ever bound. The depth sweep passes its own two, which are its widest point's, not this model's.
export function deviceNeeds(features: readonly string[], bufferBytes: number, bindingBytes: number): DeviceNeeds {
  return {
    features,
    limits: { ...DEVICE_MINIMUMS, maxBufferSize: bufferBytes, maxStorageBufferBindingSize: bindingBytes },
  };
}

// The largest range any bind group ever covers. A fused block binds the whole blob as one
// `array<u32>` and indexes it with baked offsets, so on any arm that carries a lever the blob
// itself is that range; R1's largest is a single weight section, and a narrow enough model makes
// the logits buffer larger than either.
export function maxBoundBytes(
  shapeVocab: number,
  maxQueries: number,
  sections: readonly { readonly bytes: number }[],
  blobBytes: number,
): number {
  let largest = Math.max(shapeVocab * maxQueries * PRECISION.logitsBytes, blobBytes);
  for (const section of sections) {
    largest = Math.max(largest, section.bytes);
  }
  return largest;
}

export function deviceNeedsFor(config: RuntimeConfig): DeviceNeeds {
  return deviceNeeds(
    config.extraFeatures,
    config.weightBytes,
    maxBoundBytes(config.shape.vocab, config.maxQueries, config.sections, config.weightBytes),
  );
}

export async function initFused(config: RuntimeConfig): Promise<{ session: FusedSession; ready: RuntimeReady }> {
  const errors: string[] = [];
  const gpu = await deviceForRuntime(deviceNeedsFor(config), config.powerPreference, config.expectedSubgroupSize);
  const blob = await fetchWeights(gpu.device, config.weightsUrl, config.weightBytes);
  const buildStart = performance.now();
  let session: FusedSession | null = null;
  await withValidation(gpu.device, errors, "runtime_build", async () => {
    session = buildSession(gpu, config, blob, errors);
    await waitIdle(gpu.device);
  });
  if (session === null) {
    throw new Error(`runtime build failed: ${errors.join("; ")}`);
  }
  const built: FusedSession = session;
  const buildMs = performance.now() - buildStart;
  const ready: RuntimeReady = {
    env: pageEnv(),
    device: gpu.descriptor,
    dispatches_per_token: built.runtime.dispatchCount,
    arm_dispatches: built.runtime.armKeys.map((key) => built.runtime.dispatchCountOf(key)),
    pipelines: built.runtime.pipelineCount,
    bind_groups: built.runtime.bindGroupCount,
    weight_bytes: blob.bytes,
    live_gpu_bytes: built.runtime.liveGpuBytes,
    build_ms: buildMs,
    weights_fetch_ms: blob.fetchMs,
    weights_sha256: blob.sha256,
    subgroup_min: gpu.descriptor.adapter.subgroupMinSize,
    subgroup_max: gpu.descriptor.adapter.subgroupMaxSize,
    validation_errors: [...errors],
  };
  return { session: built, ready };
}

// ---- readback ring ----

// One staging slot per `readbackEvery` tokens of the registered ring. Recycling a slot awaits its
// outstanding map, which is the only backpressure in the pipelined loop: with the registered ring
// the GPU is never more than one ring behind, so the wait is free at every 32 and becomes the
// measured tax at every 1.
class ReadbackRing {
  private readonly slots: GPUBuffer[] = [];
  private readonly pending: (Promise<undefined> | null)[] = [];
  private readonly chunkAt: number[] = [];
  private readonly ids: number[] = [];

  constructor(
    private readonly device: GPUDevice,
    private readonly slotCount: number,
    private readonly tokensPerSlot: number,
  ) {
    for (let i = 0; i < slotCount; i += 1) {
      this.slots.push(
        device.createBuffer({ size: tokensPerSlot * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: `readback${i}` }),
      );
      this.pending.push(null);
      this.chunkAt.push(-1);
    }
  }

  private slot(index: number): GPUBuffer {
    const buffer = this.slots[index % this.slotCount];
    if (buffer === undefined) {
      throw new Error("readback ring index");
    }
    return buffer;
  }

  async recycle(chunk: number): Promise<GPUBuffer> {
    const index = chunk % this.slotCount;
    const outstanding = this.pending[index];
    if (outstanding !== undefined && outstanding !== null) {
      await outstanding;
      this.drainSlot(index);
    }
    return this.slot(chunk);
  }

  private drainSlot(index: number): void {
    const buffer = this.slots[index];
    const at = this.chunkAt[index];
    if (buffer === undefined || at === undefined || at < 0) {
      return;
    }
    const view = new Uint32Array(buffer.getMappedRange().slice(0));
    buffer.unmap();
    for (let i = 0; i < view.length; i += 1) {
      const value = view[i];
      if (value === undefined) {
        throw new Error("readback view index");
      }
      this.ids[at * this.tokensPerSlot + i] = value;
    }
    this.pending[index] = null;
    this.chunkAt[index] = -1;
  }

  start(chunk: number): void {
    const index = chunk % this.slotCount;
    this.chunkAt[index] = chunk;
    this.pending[index] = this.slot(chunk).mapAsync(GPUMapMode.READ);
  }

  async drain(): Promise<readonly number[]> {
    for (let index = 0; index < this.slotCount; index += 1) {
      const outstanding = this.pending[index];
      if (outstanding !== undefined && outstanding !== null) {
        await outstanding;
        this.drainSlot(index);
      }
    }
    return this.ids;
  }

  destroy(): void {
    for (const buffer of this.slots) {
      buffer.destroy();
    }
  }
}

// ---- run primitives ----

export interface PrimeOptions {
  readonly tokenIds: Uint32Array;
  readonly promptLength: number;
  readonly seed: number;
  // 1 = the prefill's own sampled id is written to tokens[promptLength], which is what a
  // free-running continuation reads next. Teacher forcing leaves the written stream intact.
  readonly freerun: boolean;
}

// Walks the prompt through the arm that is already selected, one token at a time, leaving
// position = step = promptLength on the device.
//
// There is no multi-query prefill program in this unit. Every arm is decode-only -- the fused
// blocks are written for a single query token and the split arm's core carries one query per
// workgroup -- so maxQueries is 1 and the warm cache is built by stepping the arm under test.
// That is the stronger construction anyway: the KV rows, the conv state and the residual parity a
// measured run reads were written by the same kernels that then read them, so an arm is never
// timed against a cache some other arm's kernels filled.
//
// The whole prompt goes into one command buffer because planStep proves each token closes on the
// parity it opened on, so the tokens chain without a resubmit between them.
export async function prime(session: FusedSession, options: PrimeOptions): Promise<void> {
  const { runtime, gpu } = session;
  // The prompt walk ends with `promptLength` positions cached and the last of them attended, so the
  // reach it needs is exactly that count. Asserting before the walk means a runtime whose flash
  // tiles stop short fails here rather than silently dropping the positions past the last tile.
  runtime.assertReach(options.promptLength);
  runtime.clearCaches();
  runtime.clearResidual();
  runtime.writeTokens(options.tokenIds, 0);
  runtime.writeState({
    position: 0,
    step: 0,
    queries: 1,
    sampleMode: 0,
    rng: options.seed,
    freerun: 0,
    advance: 1,
  });
  const encoder = gpu.device.createCommandEncoder();
  for (let token = 0; token < options.promptLength; token += 1) {
    runtime.encodeToken(encoder, 1, null);
  }
  // Teacher-force the complete prompt. Only its final prediction seeds free generation.
  if(options.freerun){
    if(options.promptLength<1)throw new Error('Free generation needs a nonempty prompt');
    encoder.copyBufferToBuffer(runtime.buffers.sampled,(options.promptLength-1)*4,runtime.buffers.tokens,options.promptLength*4,4);
  }
  gpu.device.queue.submit([encoder.finish()]);
  await waitIdle(gpu.device);
}

export interface DecodeOptions {
  readonly tokens: number;
  readonly startPosition: number;
  readonly startStep: number;
  readonly readbackEvery: number;
  readonly ringTokens: number;
  readonly timestamps: boolean;
  readonly freerun: boolean;
  readonly seed: number;
}

export interface DecodeResult {
  readonly wallMs: number;
  readonly perTokenMs: readonly number[];
  readonly gpuBusyMs: number;
  readonly gpuSpanMs: number;
  readonly sampled: readonly number[];
  readonly snapshot: HookSnapshot;
  readonly gpuChunks?:readonly {tokens:number;ms:number}[];
}

function setDecodeState(session: FusedSession, options: DecodeOptions): void {
  session.runtime.writeState({
    position: options.startPosition,
    step: options.startStep,
    queries: 1,
    sampleMode: 0,
    rng: options.seed,
    freerun: options.freerun ? 1 : 0,
    advance: 1,
  });
}

function ringFor(device: GPUDevice, options: DecodeOptions): ReadbackRing {
  if (options.ringTokens % options.readbackEvery !== 0) {
    throw new Error(`ring of ${options.ringTokens} tokens does not divide by readback every ${options.readbackEvery}`);
  }
  return new ReadbackRing(device, options.ringTokens / options.readbackEvery, options.readbackEvery);
}

async function timing(
  device: GPUDevice,
  querySet: GPUQuerySet | null,
  tokens: number,
  wallMs: number,
): Promise<{ perTokenMs: number[]; gpuBusyMs: number; gpuSpanMs: number }> {
  if (querySet === null) {
    return { perTokenMs: [], gpuBusyMs: -1, gpuSpanMs: -1 };
  }
  const stamps = await resolveTimestamps(device, querySet, tokens);
  const perTokenMs: number[] = [];
  let busy = 0;
  for (let i = 0; i < tokens; i += 1) {
    const begin = stamps[i * 2];
    const end = stamps[i * 2 + 1];
    const nextBegin = i + 1 < tokens ? stamps[(i + 1) * 2] : undefined;
    if (begin === undefined || end === undefined) {
      throw new Error("timestamp index");
    }
    busy += end - begin;
    // Per-token cost is begin-to-begin so the inter-token gap is charged to the token that
    // preceded it; the last token has no successor and is charged its own span.
    perTokenMs.push(((nextBegin === undefined ? end : nextBegin) - begin) / 1e6);
  }
  const first = stamps[0];
  const last = stamps[tokens * 2 - 1];
  if (first === undefined || last === undefined) {
    throw new Error("timestamp span index");
  }
  return { perTokenMs, gpuBusyMs: busy / 1e6, gpuSpanMs: (last - first) / 1e6 };
}

export async function decodePipelined(session: FusedSession, options: DecodeOptions): Promise<DecodeResult> {
  if(session.runtime.tokensPerPass>1)return decodeChunked(session,options);
  const { runtime, gpu } = session;
  runtime.assertReach(options.startPosition + options.tokens);
  const device = gpu.device;
  const hooks = globalThis.__probeHooks;
  const querySet = options.timestamps
    ? device.createQuerySet({ type: "timestamp", count: options.tokens * 2, label: "token_pass" })
    : null;
  const ring = ringFor(device, options);
  setDecodeState(session, options);
  await waitIdle(device);
  hooks.beginStep();
  const start = performance.now();
  for (let i = 0; i < options.tokens; i += 1) {
    const encoder = device.createCommandEncoder();
    runtime.encodeToken(
      encoder,
      1,
      querySet === null ? null : { querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 },
    );
    const done = (i + 1) % options.readbackEvery === 0;
    let chunk = -1;
    if (done) {
      chunk = (i + 1) / options.readbackEvery - 1;
      const slot = await ring.recycle(chunk);
      encoder.copyBufferToBuffer(runtime.buffers.sampled, (options.startStep + i + 1 - options.readbackEvery) * 4, slot, 0, options.readbackEvery * 4);
    }
    device.queue.submit([encoder.finish()]);
    if (done) {
      ring.start(chunk);
    }
  }
  await waitIdle(device);
  const wallMs = performance.now() - start;
  const snapshot = hooks.endStep();
  const sampled = await ring.drain();
  ring.destroy();
  const times = await timing(device, querySet, options.tokens, wallMs);
  if (querySet !== null) {
    querySet.destroy();
  }
  return { wallMs, sampled: [...sampled], snapshot, ...times };
}

async function decodeChunked(session:FusedSession,options:DecodeOptions):Promise<DecodeResult>{
 const {runtime,gpu}=session,device=gpu.device,hooks=globalThis.__probeHooks;
 runtime.assertReach(options.startPosition+options.tokens);
 const size=Math.min(runtime.tokensPerPass,options.readbackEvery);
 if(options.readbackEvery%size)throw new Error('Chunk does not divide readback cadence');
 const count=Math.ceil(options.tokens/size),qs=options.timestamps?device.createQuerySet({type:'timestamp',count:count*2,label:'decode_chunk'}):null;
 const ring=ringFor(device,options);setDecodeState(session,options);await waitIdle(device);hooks.beginStep();const start=performance.now();
 const lengths:number[]=[];
 for(let first=0,index=0;first<options.tokens;first+=size,index++){
  const n=Math.min(size,options.tokens-first),done=(first+n)%options.readbackEvery===0;
  const encoder=device.createCommandEncoder();runtime.encodeChunk(encoder,n,qs?{querySet:qs,beginningOfPassWriteIndex:index*2,endOfPassWriteIndex:index*2+1}:null);lengths.push(n);
  let chunk=-1;
  if(done){chunk=(first+n)/options.readbackEvery-1;const slot=await ring.recycle(chunk);encoder.copyBufferToBuffer(runtime.buffers.sampled,(options.startStep+first+n-options.readbackEvery)*4,slot,0,options.readbackEvery*4);}
  device.queue.submit([encoder.finish()]);if(done)ring.start(chunk);
 }
 await waitIdle(device);const wallMs=performance.now()-start,snapshot=hooks.endStep(),sampled=await ring.drain();ring.destroy();
 const gpuChunks:{tokens:number;ms:number}[]=[];let gpuBusyMs=-1,gpuSpanMs=-1;
 if(qs){const stamps=await resolveTimestamps(device,qs,count);gpuBusyMs=0;
  for(let i=0;i<count;i++){const ms=(stamps[i*2+1]!-stamps[i*2]!)/1e6;gpuChunks.push({tokens:lengths[i]!,ms});gpuBusyMs+=ms;}
  gpuSpanMs=(stamps[count*2-1]!-stamps[0]!)/1e6;qs.destroy();
 }
 return {wallMs,snapshot,sampled:[...sampled],perTokenMs:[],gpuBusyMs,gpuSpanMs,gpuChunks};
}

export async function decodeSync(session: FusedSession, options: DecodeOptions): Promise<DecodeResult> {
  const { runtime, gpu } = session;
  runtime.assertReach(options.startPosition + options.tokens);
  const device = gpu.device;
  const hooks = globalThis.__probeHooks;
  const querySet = options.timestamps
    ? device.createQuerySet({ type: "timestamp", count: options.tokens * 2, label: "token_pass_sync" })
    : null;
  const ring = ringFor(device, options);
  setDecodeState(session, options);
  await waitIdle(device);
  const perTokenWall: number[] = [];
  hooks.beginStep();
  const start = performance.now();
  for (let i = 0; i < options.tokens; i += 1) {
    const tokenStart = performance.now();
    const encoder = device.createCommandEncoder();
    runtime.encodeToken(
      encoder,
      1,
      querySet === null ? null : { querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 },
    );
    const done = (i + 1) % options.readbackEvery === 0;
    let chunk = -1;
    if (done) {
      chunk = (i + 1) / options.readbackEvery - 1;
      const slot = await ring.recycle(chunk);
      encoder.copyBufferToBuffer(runtime.buffers.sampled, (options.startStep + i + 1 - options.readbackEvery) * 4, slot, 0, options.readbackEvery * 4);
    }
    device.queue.submit([encoder.finish()]);
    if (done) {
      ring.start(chunk);
    }
    await waitIdle(device);
    perTokenWall.push(performance.now() - tokenStart);
  }
  const wallMs = performance.now() - start;
  const snapshot = hooks.endStep();
  const sampled = await ring.drain();
  ring.destroy();
  const times = await timing(device, querySet, options.tokens, wallMs);
  if (querySet !== null) {
    querySet.destroy();
  }
  // The synchronous arm's per-token number is its own wall time, not a timestamp difference.
  return { wallMs, perTokenMs: perTokenWall, gpuBusyMs: times.gpuBusyMs, gpuSpanMs: times.gpuSpanMs, sampled: [...sampled], snapshot };
}

// M3: one compute pass per dispatch so each gets its own timestamp pair. The pass structure is
// different from a real token, so this run's wall time is never reported as t_AR.
export async function perDispatchTiming(
  session: FusedSession,
  queries: number,
  position: number,
  step: number,
  reps: number,
  seed: number,
): Promise<number[]> {
  const { runtime, gpu } = session;
  runtime.assertReach(position + queries);
  const device = gpu.device;
  const count = runtime.dispatchCount;
  const querySet = device.createQuerySet({ type: "timestamp", count: count * 2, label: "per_dispatch" });
  runtime.writeState({ position, step, queries, sampleMode: 0, rng: seed, freerun: 0, advance: 0 });
  await waitIdle(device);
  const totals = new Float64Array(count);
  for (let rep = 0; rep < reps; rep += 1) {
    const encoder = device.createCommandEncoder();
    const written = runtime.encodePerDispatchPasses(encoder, queries, querySet, 0);
    if (written !== count * 2) {
      throw new Error(`per-dispatch instrument wrote ${written} queries for ${count} dispatches`);
    }
    device.queue.submit([encoder.finish()]);
    await waitIdle(device);
    const stamps = await resolveTimestamps(device, querySet, count);
    for (let i = 0; i < count; i += 1) {
      const begin = stamps[i * 2];
      const end = stamps[i * 2 + 1];
      if (begin === undefined || end === undefined) {
        throw new Error("per-dispatch timestamp index");
      }
      const previous = totals[i];
      if (previous === undefined) {
        throw new Error("per-dispatch total index");
      }
      totals[i] = previous + (end - begin) / 1e6;
    }
  }
  querySet.destroy();
  return Array.from(totals, (value) => value / reps);
}

// The per-dispatch table, one row per dispatch of the arm that is currently selected. Both the M
// stage and P3 publish this table -- M to price the levers, P3 to separate the attention core's
// growth in position from the projections' flatness -- so the two read it through one construction.
// What each dispatch moves at the cache depth it actually ran at. The encoder is depth-independent
// -- the kernels read the position from a uniform, so one set of pipelines and bind groups serves
// every depth and the session builds its step lists once, at the warm position -- but the bytes are
// not: the attention core streams 2 * kvHeads * (position + q) * headDim * kvCacheBytes of cache,
// which is 393 216 B at 192 positions and 4 194 304 B at 2048. Reporting the build position's figure
// would publish a 192-position read for a 2048-position dispatch, which is the one quantity the
// context sweep exists to measure. So the accounting is re-planned at the run's depth and asserted
// dispatch-for-dispatch against the plan the encoder holds; only the byte fields may differ.
function accountingPlan(session: FusedSession, position: number): readonly PlannedDispatch[] {
  const arm = session.runtime.arm;
  // Bytes at `position`, occupancy at the session's own plan position: S is fixed for a run, so
  // re-choosing it here would build a plan the encoder never ran and the assertion below -- which is
  // what makes the two plans interchangeable on every non-byte field -- would fire.
  const spec = armPlanOptions(session.shape, position, session.planPosition, session.tuning).find(
    (entry) => entry.key === arm,
  );
  if (spec === undefined) {
    throw new Error(`per-dispatch record: ${arm} is not a registered arm`);
  }
  const plan = efficiencyPlan(planStep(session.shape, spec.options),candidateOf(arm),session.shape,spec.options.position,spec.options.queries);
  if (plan.length !== session.runtime.steps.length) {
    throw new Error(
      `per-dispatch record: ${arm} encodes ${session.runtime.steps.length} dispatches, plans ${plan.length} at position ${position}`,
    );
  }
  return plan;
}

export function perDispatchRecords(
  session: FusedSession,
  gpuMs: readonly number[],
  position: number,
): PerDispatch[] {
  const planned = accountingPlan(session, position);
  const records: PerDispatch[] = [];
  for (let index = 0; index < session.runtime.steps.length; index += 1) {
    const step = session.runtime.steps[index];
    const measured = gpuMs[index];
    const at = planned[index];
    if (step === undefined || measured === undefined || at === undefined) {
      throw new Error(`per-dispatch record ${index} missing`);
    }
    const { plan } = step;
    if (at.name !== plan.name || at.kind !== plan.kind) {
      throw new Error(
        `per-dispatch record ${index}: ${plan.name}/${plan.kind} encoded, ${at.name}/${at.kind} planned at position ${position}`,
      );
    }
    if (at.workgroups !== plan.workgroups || at.workgroupSize !== plan.workgroupSize) {
      throw new Error(
        `per-dispatch record ${plan.name}: ${plan.workgroups}x${plan.workgroupSize} encoded, ` +
          `${at.workgroups}x${at.workgroupSize} planned at position ${position} -- occupancy is not a function of depth`,
      );
    }
    // Every matrix this dispatch moves, not just the one a matvec kernel binds: `plan.weight` is
    // null on every fused block, so reading it here would report a block that streams three
    // matrices as a dispatch that reads no weights at all, and the registered partition would
    // charge it to the dispatch term.
    const weightBytes = at.reads.reduce((total, read) => total + read.bytes, 0);
    records.push({
      index: plan.index,
      name: plan.name,
      kind: plan.kind,
      gpu_ms: measured,
      weight_bytes: weightBytes,
      total_bytes: weightBytes + at.constantBytes + at.readBytes + at.writeBytes,
      // Read off the plan the encoder ran, not re-derived: the per-kind table is compared against
      // the P0a surface cell this dispatch was priced in, and a re-derivation could disagree. The
      // assertion above is what makes the two plans interchangeable on this field.
      workgroups: plan.workgroups,
      workgroup_size: plan.workgroupSize,
      site: plan.site,
      rows_per_subgroup: plan.geometry === null ? null : plan.geometry.rowsPerSubgroup,
      subgroups_per_row: plan.geometry === null ? null : plan.geometry.subgroupsPerRow,
      attn_part: attnPartCode(plan.attnPart),
    });
  }
  return records;
}

export async function readSampled(session: FusedSession, from: number, count: number): Promise<Uint32Array> {
  const raw = await readBuffer(session.gpu.device, session.runtime.buffers.sampled, from * 4, count * 4);
  return new Uint32Array(raw);
}

export async function readLogitsRow(session: FusedSession, row: number, vocab: number): Promise<Float32Array> {
  const raw = await readBuffer(session.gpu.device, session.runtime.buffers.logits, row * vocab * 4, vocab * 4);
  return new Float32Array(raw);
}

export const NORM_EPS = SHAPE.normEps;

// Same per-dispatch timestamps, with 32 repeats between readbacks instead of one.
export async function perDispatchTimingBatched(session:FusedSession,position:number,reps:number,wholePass=false,onAudit?:((value:unknown)=>void)):Promise<number[]>{
 const {runtime,gpu}=session,device=gpu.device,count=wholePass?1:runtime.dispatchCount;
 runtime.writeState({position,step:position,queries:1,sampleMode:0,rng:111004,freerun:0,advance:0});await waitIdle(device);
 const totals=new Float64Array(count);let completed=0;const chunks:any[]=[];
 while(completed<reps){const chunk=Math.min(Math.floor(4096/(count*2)),reps-completed);const qs=device.createQuerySet({type:'timestamp',count:count*2*chunk});const enc=device.createCommandEncoder();
  for(let rep=0;rep<chunk;rep++){if(wholePass)runtime.encodeToken(enc,1,{querySet:qs,beginningOfPassWriteIndex:rep*2,endOfPassWriteIndex:rep*2+1});else{const written=runtime.encodePerDispatchPasses(enc,1,qs,rep*count*2);if(written!==count*2)throw new Error(`Batched query write mismatch ${written} != ${count*2}`);}}
  device.queue.submit([enc.finish()]);await waitIdle(device);const stamps=await resolveTimestamps(device,qs,count*chunk);
  const diffs=Array.from({length:count*chunk},(_,i)=>stamps[i*2+1]!-stamps[i*2]!);
  const gcd=(a:number,b:number):number=>{while(b){const t=a%b;a=b;b=t;}return a;};
  chunks.push({count:count*chunk,zero:diffs.filter(x=>x===0).length,negative:diffs.filter(x=>x<0).length,positive:diffs.filter(x=>x>0).length,timestamp_grid_ns:Array.from(stamps).reduce((a,b)=>gcd(a,b),0),first_pairs_ns:Array.from(stamps.slice(0,32))});
  for(let rep=0;rep<chunk;rep++)for(let i=0;i<count;i++){const at=(rep*count+i)*2;totals[i]=totals[i]!+(stamps[at+1]!-stamps[at]!)/1e6;}

  qs.destroy();completed+=chunk;
 }
 onAudit?.({arm:runtime.arm,position,reps,count,chunks});
 if([...totals].some(x=>!(x>0))){
  const error=Object.assign(new Error('Nonpositive GPU timestamps in batched instrument'),{details:{arm:runtime.arm,position,reps,count,totals_ms:Array.from(totals),invalid_indices:Array.from(totals).flatMap((v,i)=>v>0?[]:[i]),chunks}});throw error;
 }
 return Array.from(totals,x=>x/reps);
}
