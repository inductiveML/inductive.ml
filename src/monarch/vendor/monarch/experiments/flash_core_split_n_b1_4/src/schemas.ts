// Zod schemas for every boundary: stage files, page <-> node messages, results.json.
import { z } from "zod";
import { MATVEC_CLASSES, MATVEC_SHAPES, MATVEC_SITES } from "./pure/plan.ts";

// The dequant variants, as names rather than as the page's kernel union: schemas.ts is the node
// side of the boundary and must not pull a WGSL module in. `wgsl.ts` asserts the two lists agree.
export const DEQUANT_VARIANT_NAMES = ["carried", "v1_vec4_f16", "v2_vec4_accumulator", "v3_both"] as const;
export type DequantVariantName = (typeof DEQUANT_VARIANT_NAMES)[number];

export const finite = z.number().finite();
export const nonNegative = finite.min(0);
export const count = z.number().int().min(0);
export const ci = z.tuple([finite, finite]);

// ---- config.json of the exported model ----
export const ModelConfigSchema = z.object({
  architectures: z.array(z.string()),
  model_type: z.string(),
  num_hidden_layers: z.number().int().min(1),
  layer_types: z.array(z.enum(["conv", "full_attention"])),
  hidden_size: z.number().int().min(1),
  intermediate_size: z.number().int().min(1),
  vocab_size: z.number().int().min(1),
  num_attention_heads: z.number().int().min(1),
  num_key_value_heads: z.number().int().min(1),
  conv_L_cache: z.number().int().min(1),
  conv_bias: z.boolean(),
  norm_eps: finite.positive(),
  rope_parameters: z.object({ rope_theta: finite.positive(), rope_type: z.string() }),
  tie_embedding: z.boolean(),
  max_position_embeddings: z.number().int().min(1),
  bos_token_id: z.number().int().min(0),
  eos_token_id: z.number().int().min(0),
});
export type ModelConfig = z.infer<typeof ModelConfigSchema>;

// ---- P0 ----
export const TensorSourceSchema = z.object({
  name: z.string(),
  dataType: z.number().int(),
  dims: z.array(z.number().int()),
  bytes: count,
  external: z.boolean(),
  offset: count,
});

export const P0MatrixSchema = z.object({
  node: z.string(),
  role: z.string(),
  layer: z.number().int(),
  rows: z.number().int().min(1),
  cols: z.number().int().min(1),
  bits: z.number().int(),
  block_size: z.number().int(),
  accuracy_level: z.number().int(),
  zero_points: z.boolean(),
  bytes: count,
});

export const P0DispatchSchema = z.object({
  index: count,
  name: z.string(),
  kind: z.string(),
  layer: z.number().int(),
  rows: count,
  cols: count,
  weight_bytes: count,
  constant_bytes: count,
  read_bytes: count,
  write_bytes: count,
  total_bytes: count,
  byte_ms: nonNegative,
  latency_bound: z.boolean(),
});

export const P0PredictionSchema = z.object({
  bytes_ms: nonNegative,
  submit_ms: nonNegative,
  dispatch_cost_ms: nonNegative,
  dispatches: count,
  dispatch_ms: nonNegative,
  idle_ms: nonNegative,
  total_ms: nonNegative,
});

export const P0ResultSchema = z.object({
  unit: z.string(),
  spec: z.string(),
  model: z.string(),
  shape: z.object({
    layers: count,
    conv_layers: count,
    attention_layers: count,
    hidden: count,
    ffn: count,
    vocab: count,
    heads: count,
    kv_heads: count,
    head_dim: count,
    conv_cache: count,
    quant_bits: count,
    quant_block: count,
  }),
  config_json_verified: z.boolean(),
  model_files_verified: z.boolean(),
  graph: z.object({
    matmul_nodes: count,
    op_counts: z.record(z.string(), z.number().int()),
    external_bytes: count,
    matrices: z.array(P0MatrixSchema),
  }),
  plan: z.object({
    d_budget: count,
    matmul_dispatches: count,
    dispatches_per_conv_layer: count,
    dispatches_per_attention_layer: count,
    concat_qkv: z.boolean(),
    concat_gate_up: z.boolean(),
    weight_bytes: count,
    constant_bytes: count,
    read_bytes: count,
    write_bytes: count,
    total_bytes: count,
    position: count,
    queries: count,
    dispatches: z.array(P0DispatchSchema),
  }),
  plan_matches_graph: z.boolean(),
  byte_model: z.object({
    matmul: count,
    head: count,
    layer_norms: count,
    conv_weights: count,
    head_norms: count,
    final_norm: count,
    layer_other_at_least_1kib: count,
    embedding_row: count,
    rope_rows: count,
    total: count,
  }),
  byte_model_matches_b1_0_graph: z.boolean(),
  b1_0: z.object({
    constants_verified: z.boolean(),
    a_submit_ms: nonNegative,
    b_dispatch_ms: nonNegative,
    matvec_slope_ms_per_byte: nonNegative,
    matvec_GBps: nonNegative,
    chain_effective_GBps: nonNegative,
    chain_latency_per_dispatch_ms: finite,
    t_AR_ms: nonNegative,
    dispatches_per_step: count,
    idle_share: nonNegative,
    bytes_floor_ms: nonNegative,
  }),
  prediction: P0PredictionSchema,
  prediction_measured_constants: P0PredictionSchema,
  prediction_full_traffic: P0PredictionSchema,
  prediction_latency_floor: z.object({
    latency_ms_per_dispatch: finite,
    gpu_ms: nonNegative,
    total_ms: nonNegative,
    latency_bound_dispatches: count,
  }),
  pricing_miss_band_ms: ci,
  d_budget_in_spec_range: z.boolean(),
  d_budget_impl_gap_ceiling: count,
  named_risk_term: z.string(),
  context_configs: z.array(
    z.object({
      layers: count,
      hidden: count,
      ffn: count,
      ffn_ratio: finite,
      heads: count,
      kv_heads: count,
      attention_layers: count,
      matmul_bytes: count,
      relative_error: finite,
      d_budget: count,
    }),
  ),
  notes: z.array(z.string()),
});
export type P0Result = z.infer<typeof P0ResultSchema>;

// ---- extract ----
export const WeightSectionSchema = z.object({
  name: z.string(),
  role: z.string(),
  layer: z.number().int(),
  // `kquant` / `kscales` are the K-sliced repacks the fused blocks read: a pure permutation of the
  // matching `quant` / `scales` section, appended after everything the R1 plan needs.
  kind: z.enum([
    "quant",
    "scales",
    "zero_points",
    "norm",
    "conv",
    "rope",
    "embed_quant",
    "embed_scales",
    "embed_zero_points",
    "kquant",
    "kscales",
  ]),
  data_type: z.number().int(),
  rows: count,
  cols: count,
  offset: count,
  bytes: count,
  sources: z.array(z.string()),
});
export type WeightSection = z.infer<typeof WeightSectionSchema>;

export const WeightManifestSchema = z.object({
  unit: z.string(),
  model: z.string(),
  file: z.string(),
  file_bytes: count,
  file_sha256: z.string(),
  alignment: z.number().int().min(1),
  rope_rows: count,
  quant_bits: z.number().int(),
  quant_block: z.number().int(),
  sections: z.array(WeightSectionSchema),
  totals: z.object({
    sections: count,
    quant_bytes: count,
    scales_bytes: count,
    zero_point_bytes: count,
    constant_bytes: count,
    matmul_bytes: count,
  }),
});
export type WeightManifest = z.infer<typeof WeightManifestSchema>;

// ---- page boundary (ported from B1.0 so the reference arm's records are field-identical) ----
export const AdapterRecordSchema = z.object({
  vendor: z.string(),
  architecture: z.string(),
  device: z.string(),
  description: z.string(),
  isFallbackAdapter: z.boolean(),
  subgroupMinSize: count,
  subgroupMaxSize: count,
  features: z.array(z.string()),
  limits: z.record(z.string(), finite),
});
export type AdapterRecord = z.infer<typeof AdapterRecordSchema>;

export const DeviceDescriptorRecordSchema = z.object({
  requestedFeatures: z.array(z.string()),
  requiredFeatures: z.array(z.string()),
  addedFeatures: z.array(z.string()),
  requiredLimits: z.record(z.string(), finite),
  label: z.string(),
  adapter: AdapterRecordSchema,
});
export type DeviceDescriptorRecord = z.infer<typeof DeviceDescriptorRecordSchema>;

export const HookSnapshotSchema = z.object({
  dispatches: count,
  dispatches_indirect: count,
  submits: count,
  command_buffers: count,
  passes: count,
  bind_groups: count,
  write_buffers: count,
  write_buffer_bytes: count,
  create_buffers: count,
  create_buffer_bytes: count,
  destroy_buffers: count,
  pipelines: count,
  pipelines_async: count,
  pipeline_compile_ms: nonNegative,
  shader_modules: count,
  map_asyncs: count,
  copy_buffer_to_buffer: count,
  first_submit_ms: finite,
  last_submit_ms: finite,
  submit_offsets_ms: z.array(finite),
  injected: count,
  live_gpu_bytes: count,
  step_ms: nonNegative,
});
export type HookSnapshot = z.infer<typeof HookSnapshotSchema>;

export const TimestampPassSchema = z.object({
  index: count,
  dispatches: count,
  begin_ns: finite,
  end_ns: finite,
});
export type TimestampPass = z.infer<typeof TimestampPassSchema>;

export const PageEnvSchema = z.object({
  user_agent: z.string(),
  cross_origin_isolated: z.boolean(),
  timer_resolution_ms: finite,
  hardware_concurrency: count,
  webgpu: z.boolean(),
});
export type PageEnv = z.infer<typeof PageEnvSchema>;

export const EmitRecordSchema = z.object({ t: count, kind: z.string(), payload: z.json() });
export type EmitRecord = z.infer<typeof EmitRecordSchema>;

// ---- e1 ----
export const E1ResultSchema = z.object({
  webgpu_feature_status: z.string(),
  gpu_page_text_has_hardware_webgpu: z.boolean(),
  gpu_page_webgpu_line: z.string(),
  machine_model: z.string(),
  gpu_cores: count,
  on_ac_power: z.boolean(),
  adapter: AdapterRecordSchema,
  env: PageEnvSchema,
  hardware_accelerated: z.boolean(),
});
export type E1Result = z.infer<typeof E1ResultSchema>;

// ---- runtime configuration handed to the page ----
export const ModelShapeSchema = z.object({
  layerTypes: z.array(z.enum(["conv", "attention"])),
  hidden: z.number().int().min(1),
  ffn: z.number().int().min(1),
  vocab: z.number().int().min(1),
  heads: z.number().int().min(1),
  kvHeads: z.number().int().min(1),
  headDim: z.number().int().min(1),
  convCache: z.number().int().min(1),
  quantBits: z.literal(4),
  quantBlock: z.number().int().min(1),
});
export type ModelShapeRecord = z.infer<typeof ModelShapeSchema>;

// One matvec occupancy point. Every field is encoded, never inferred: the workgroup count follows
// from `rowsPerSubgroup` and `subgroupsPerRow` alone, so a plan and the dispatch it asserts against
// cannot drift apart.
export const MatvecGeometrySchema = z.object({
  workgroupSize: z.number().int().min(1),
  subgroupSize: z.number().int().min(1),
  rowsPerSubgroup: z.number().int().min(1),
  subgroupsPerRow: z.number().int().min(1),
});
export type MatvecGeometryRecord = z.infer<typeof MatvecGeometrySchema>;

// P0a's knee, one geometry per measured shape cell, and P0c's optimum, one block count per depth.
// Both are measurements, so neither can be a page constant: the page is handed the table the node
// stage measured this session and the plan is built from it.
export const KneeEntrySchema = z.object({ shape: z.enum(MATVEC_SHAPES), geometry: MatvecGeometrySchema });
export type KneeEntry = z.infer<typeof KneeEntrySchema>;

export const FlashOptimumSchema = z.object({
  positions: z.number().int().min(1),
  blocks: z.number().int().min(1),
});
export type FlashOptimumRecord = z.infer<typeof FlashOptimumSchema>;

export const RuntimeConfigSchema = z.object({
  shape: ModelShapeSchema,
  weightsUrl: z.string(),
  weightBytes: count,
  sections: z.array(z.object({ name: z.string(), offset: count, bytes: count })),
  maxQueries: z.number().int().min(1),
  maxPositions: z.number().int().min(1),
  samplePartials: z.number().int().min(1),
  // The position the arm plans are built at -- the depth the per-dispatch byte table and every
  // priced dispatch refer to. It is also the depth P0c's block-count optimum is looked up at, so a
  // context sweep point states its own depth here rather than inheriting M's.
  planPosition: z.number().int().min(1),
  // The largest total position count (cached positions plus the token being decoded) any run on
  // this runtime reaches. N2's tile trip count is baked from it, and every entry point asserts
  // against it before it walks.
  maxTotalPositions: z.number().int().min(1),
  // P0a's measured knee per shape. N1 runs the split-N sites here; every other site runs at the
  // carried geometry, which is a registered constant and therefore not sent.
  knee: z.array(KneeEntrySchema),
  // P0b's adoption, applied to every arm at once. "carried" means no variant was adopted.
  dequantVariant: z.enum(DEQUANT_VARIANT_NAMES),
  // P0c's measured optimum, one entry per depth it swept. A depth with no entry falls back to the
  // registered rule.
  flashOptima: z.array(FlashOptimumSchema),
  extraFeatures: z.array(z.string()),
  powerPreference: z.enum(["high-performance", "low-power"]),
  expectedSubgroupSize: z.number().int().min(1),
});
export type RuntimeConfig = z.infer<typeof RuntimeConfigSchema>;

export const RuntimeReadySchema = z.object({
  env: PageEnvSchema,
  device: DeviceDescriptorRecordSchema,
  dispatches_per_token: count,
  // One entry per registered arm, in registered order: the dispatch count the encoder actually
  // built, which is what the D budget is adjudicated against.
  arm_dispatches: z.array(count),
  pipelines: count,
  bind_groups: count,
  weight_bytes: count,
  live_gpu_bytes: count,
  build_ms: nonNegative,
  weights_fetch_ms: nonNegative,
  weights_sha256: z.string(),
  subgroup_min: count,
  subgroup_max: count,
  validation_errors: z.array(z.string()),
});
export type RuntimeReady = z.infer<typeof RuntimeReadySchema>;

// Preflight's smoke run: the prompt through the prime dispatch, then a short decode, with the
// hook counters kept. It is the first proof that the encoder actually issues D_budget dispatches
// per token — `ready.dispatches_per_token` is only the planner's self-report of the same number.
export const SmokeConfigSchema = z.object({
  arm: z.string(),
  promptIds: z.array(count),
  tokens: z.number().int().min(1),
  readbackEvery: z.number().int().min(1),
  ringTokens: z.number().int().min(1),
  seed: count,
});
export type SmokeConfig = z.infer<typeof SmokeConfigSchema>;

export const SmokeResultSchema = z.object({
  prime_ms: nonNegative,
  wall_ms: nonNegative,
  sampled: z.array(z.number().int()),
  snapshot: HookSnapshotSchema,
});
export type SmokeResult = z.infer<typeof SmokeResultSchema>;

// ---- reference arm (B1.0's transformers.js path) ----
export const ReferenceLoadConfigSchema = z.object({
  repo: z.string(),
  dtype: z.literal("q4"),
  device: z.literal("webgpu"),
  localModelPath: z.string(),
  wasmPaths: z.string(),
  numThreads: z.number().int().min(1),
  proxy: z.boolean(),
  logSeverityLevel: z.literal([0, 1, 2, 3, 4]),
  logVerbosityLevel: z.number().int().min(0),
  expectedTransformersVersion: z.string(),
  expectedOrtVersion: z.string(),
});
export type ReferenceLoadConfig = z.infer<typeof ReferenceLoadConfigSchema>;

export const ReferenceLoadResultSchema = z.object({
  transformers_version: z.string(),
  ort_version: z.string(),
  input_names: z.array(z.string()),
  output_names: z.array(z.string()),
  load_ms: nonNegative,
  cache_names: z.array(z.string()),
  devices_created: count,
});
export type ReferenceLoadResult = z.infer<typeof ReferenceLoadResultSchema>;

export const TokenizeResultSchema = z.object({
  prompts: z.array(z.object({ index: count, raw_tokens: count, ids: z.array(count) })),
  bos_id: z.number().int(),
  eos_id: z.number().int(),
});
export type TokenizeResult = z.infer<typeof TokenizeResultSchema>;

export const ContinuationSchema = z.object({
  prompts: z.array(z.object({ index: count, prompt_ids: z.array(count), continuation_ids: z.array(count) })),
  sha256: z.string(),
});
export type Continuation = z.infer<typeof ContinuationSchema>;

export const TokenizeConfigSchema = z.object({ texts: z.array(z.string()), targetTokens: z.number().int().min(1) });
export type TokenizeConfig = z.infer<typeof TokenizeConfigSchema>;

// The reference arm generates the teacher-forcing stream once, in preflight, and it is pinned.
// Every later stage replays the pin so the fused and reference arms decode identical ids.
export const GenerateConfigSchema = z.object({
  prompts: z.array(z.object({ index: count, ids: z.array(count) })),
  continuationTokens: z.number().int().min(1),
});
export type GenerateConfig = z.infer<typeof GenerateConfigSchema>;

export const GenerateResultSchema = z.object({
  prompts: z.array(z.object({ index: count, prompt_ids: z.array(count), continuation_ids: z.array(count) })),
  prefill_ms: z.array(nonNegative),
  step_ms_p50: z.array(nonNegative),
});
export type GenerateResult = z.infer<typeof GenerateResultSchema>;

// ---- correctness ----
//
// Every arm is checked on its own: a lever that fuses correctly on paper can still get a
// reduction order, a cache index or an f16 rounding point wrong, and a wrong arm's timing is not
// a measurement of anything. Three instruments run per arm — agreement with the transformers.js
// reference over the full greedy continuation, bitwise determinism across repeated runs, and the
// deferred sums against the unfused arm, which is the single-dispatch fp32 reference for them.
export const CorrectnessConfigSchema = z.object({
  runtime: RuntimeConfigSchema,
  reference: ReferenceLoadConfigSchema,
  arms: z.array(z.string()),
  baselineArm: z.string(),
  // Each prompt carries the pin's continuation beside its ids. The page regenerates the greedy
  // continuation from the reference itself -- that is what the arms are compared against -- and
  // asserts it reproduces the pin token for token, so "the pin was reused" is a checked claim
  // rather than a filename.
  prompts: z.array(z.object({ index: count, ids: z.array(count), continuationIds: z.array(count) })),
  // The registered depth this battery adjudicates. 192 reuses B1.2's pin; 1024 runs the identical
  // battery on the winner and the baseline against a pin generated at that depth.
  depth: z.number().int().min(1),
  // How many teacher-forced tokens the pin holds, and the window of them the logits are compared
  // over. The window is centred on `depth`, so the deeper battery is the same 256-position
  // comparison made at 1024 cached positions rather than a longer one made at 192.
  continuationTokens: z.number().int().min(1),
  compareFrom: count,
  compareTokens: z.number().int().min(1),
  logitStride: z.number().int().min(1),
  top1Min: finite.min(0).max(1),
  determinismRuns: z.number().int().min(2),
  determinismTokens: z.number().int().min(1),
  // The arm that carries no deferred sum anywhere in the token: every projection is a split-N
  // matvec writing its own output rows, so each output element is one dispatch's f32 accumulation.
  // It is the reference every other arm's reduction is compared against, and it is never dropped
  // by the trim ladder even when its own timing is.
  referenceArm: z.string(),
  // The flash core check. N2's core takes S maxima and rescales where A2's takes one over the whole
  // row, so the log-sum-exp merge is the new numerics and it is checked against A2's core directly
  // -- the stronger statement than either tracking the fp32 reference equally well. Registered at
  // both 192 and 2048 positions, because the merge's rescaling error is what grows with S.
  //
  // Each depth carries its own runtime because the block count S is baked into the kernel at build
  // time: checking the 2048 core inside a session planned at 192 would check a kernel no arm ever
  // runs at 2048. It also carries the exact token stream it primes with, whose length IS the depth:
  // priming `positions - 1` tokens and encoding one more leaves the core attending `positions`.
  //
  // Null is the deeper rerun, which re-adjudicates top-1 at 1024 and not the core.
  flashCore: z
    .object({
      layer: z.number().int().min(0),
      arm: z.string(),
      referenceArm: z.string(),
      depths: z.array(
        z.object({
          // The stream holds exactly `positions` ids: the check primes all but the last and encodes
          // one more, which leaves the core attending `positions`.
          positions: z.number().int().min(1),
          tokenIds: z.array(count),
          runtime: RuntimeConfigSchema,
        }),
      ),
      tolerance: finite.positive(),
    })
    .nullable(),
  // The split-N check: N1's MLP against A2's fused block at the battery's own depth, both arms run
  // from the identical cleared state through the identical prefix. A2 sums eighty partial slices in
  // index order and N1 sums whole rows inside one workgroup, so the two differ only by summation
  // order. It primes from the battery's own first prompt, so unlike the flash core -- which is
  // registered at a depth the battery never reaches -- it carries no stream of its own. Null is the
  // deeper rerun.
  splitN: z
    .object({
      layer: z.number().int().min(0),
      arm: z.string(),
      referenceArm: z.string(),
      positions: z.number().int().min(1),
      tolerance: finite.positive(),
    })
    .nullable(),
  dequant: z.object({
    section: z.string(),
    rows: z.number().int().min(1),
    cols: z.number().int().min(1),
    zeroPoints: z.boolean(),
    layer: z.number().int(),
    rowSamples: z.number().int().min(1),
    seed: count,
    tolerance: finite.positive(),
  }),
});
export type CorrectnessConfig = z.infer<typeof CorrectnessConfigSchema>;

export const CorrectnessPromptSchema = z.object({
  prompt: count,
  positions: count,
  top1_matches: count,
  top1_agreement: finite.min(0).max(1),
  compared_logit_positions: count,
  max_abs_logit_diff: nonNegative,
  max_rel_l2: nonNegative,
  freerun_prefix_match: count,
  first_mismatch_position: z.number().int(),
});
export type CorrectnessPrompt = z.infer<typeof CorrectnessPromptSchema>;

export const DequantCheckSchema = z.object({
  layer: z.number().int(),
  rows_checked: count,
  cols: count,
  max_abs_diff: nonNegative,
  max_rel_diff: nonNegative,
  reference_norm: nonNegative,
  pass: z.boolean(),
});
export type DequantCheck = z.infer<typeof DequantCheckSchema>;

// A deferred sum against the unfused arm's sequential accumulation of the same quantity. Both
// arms run the identical dispatch prefix from the identical cleared state, so the two residuals
// differ only by the order the same f32 terms were added in.
export const DeferredCheckSchema = z.object({
  arm: z.string(),
  layer: z.number().int(),
  // How many dispatches of each arm were encoded before the residual was read back.
  arm_dispatches: count,
  baseline_dispatches: count,
  values: count,
  max_abs_diff: nonNegative,
  max_rel_diff: nonNegative,
  reference_norm: nonNegative,
  pass: z.boolean(),
});
export type DeferredCheck = z.infer<typeof DeferredCheckSchema>;

// One arm's landing point at one layer against another arm's, both run from the identical cleared
// state through the identical dispatch prefix, so the two differ only by the thing the lever
// changed. `positions` is the cache depth the block was evaluated at: the flash core is checked at
// two of them because the log-sum-exp merge's rescaling is what grows with the block count, and the
// block count grows with depth.
export const CoreCheckSchema = z.object({
  arm: z.string(),
  reference_arm: z.string(),
  layer: z.number().int(),
  positions: count,
  // The block count the checked arm ran at this depth, and the one the reference ran. A2's core is
  // one block by construction, so the pair states what the merge actually had to fold.
  blocks: count,
  reference_blocks: count,
  arm_dispatches: count,
  reference_dispatches: count,
  values: count,
  max_abs_diff: nonNegative,
  max_rel_diff: nonNegative,
  reference_norm: nonNegative,
  pass: z.boolean(),
});
export type CoreCheck = z.infer<typeof CoreCheckSchema>;

export const CorrectnessArmSchema = z.object({
  arm: z.string(),
  depth: count,
  dispatches: count,
  prompts: z.array(CorrectnessPromptSchema),
  top1_agreement: finite.min(0).max(1),
  max_abs_logit_diff: nonNegative,
  max_rel_l2: nonNegative,
  // FNV-1a over the greedy token ids and over the final logits row, one pair per repeated run.
  determinism_token_hash: z.array(count),
  determinism_logit_hash: z.array(count),
  determinism_identical: z.boolean(),
  // The whole token's deferred reduction: this arm's pre-head residual against the unfused arm's.
  residual: DeferredCheckSchema,
  pass: z.boolean(),
});
export type CorrectnessArm = z.infer<typeof CorrectnessArmSchema>;

export const CorrectnessResultSchema = z.object({
  ready: RuntimeReadySchema,
  reference: ReferenceLoadResultSchema,
  depth: count,
  arms: z.array(CorrectnessArmSchema),
  // One entry per registered flash-core position. Empty when no arm in this battery carries N2,
  // which is what the deeper rerun looks like if the winner has no flash core.
  flash_core: z.array(CoreCheckSchema),
  // Present only when both the split-N arm and the fused baseline are in this battery.
  split_n: z.array(CoreCheckSchema),
  dequant: DequantCheckSchema,
  top1_agreement: finite.min(0).max(1),
  pass: z.boolean(),
  validation_errors: z.array(z.string()),
});
export type CorrectnessResult = z.infer<typeof CorrectnessResultSchema>;

// ---- M ----
//
// Every fusion arm is interleaved against the unfused baseline inside every block, so a lever's
// delta is a paired difference under one thermal and version envelope rather than two sessions.
export const MConfigSchema = z.object({
  runtime: RuntimeConfigSchema,
  promptIds: z.array(count),
  continuationIds: z.array(count),
  warmupSteps: z.number().int().min(1),
  tokens: z.number().int().min(1),
  blocks: z.number().int().min(1),
  // The token position the per-dispatch table is taken at, and the depth the model's prices for
  // that table were written at. Node computes it and the page obeys it, because the byte check that
  // makes `predicted_ms` comparable to `gpu_ms` re-plans at this number on the node side.
  warmPosition: count,
  arms: z.array(z.string()),
  baselineArm: z.string(),
  interleave: z.array(z.enum(["A_arm", "B_baseline"])),
  thermalBlocks: z.number().int().min(0),
  readbackEvery: z.number().int().min(1),
  readbackVariants: z.array(z.number().int().min(1)),
  readbackReps: z.number().int().min(1),
  perDispatchReps: z.number().int().min(1),
  syncRuns: z.number().int().min(1),
  timestampCapacityPairs: z.number().int().min(1).max(8192),
  samplingSeed: count,
});
export type MConfig = z.infer<typeof MConfigSchema>;

export const ArmRunSchema = z.object({
  arm: z.string(),
  role: z.enum(["A_arm", "B_baseline"]),
  instrument: z.enum(["pipelined", "sync"]),
  block: z.number().int(),
  tokens: count,
  start_position: count,
  wall_ms: nonNegative,
  per_token_ms: z.array(nonNegative),
  gpu_busy_ms: finite,
  gpu_span_ms: finite,
  readback_every: z.number().int().min(1),
  snapshot: HookSnapshotSchema,
  // How many decoded tokens `snapshot` spans.
  snapshot_tokens: z.number().int().min(1),
});
export type ArmRun = z.infer<typeof ArmRunSchema>;

export const PerDispatchSchema = z.object({
  index: count,
  name: z.string(),
  kind: z.string(),
  gpu_ms: finite,
  weight_bytes: count,
  total_bytes: count,
  // The occupancy the plan priced this dispatch at, carried through the measurement so the per-kind
  // GB/s can be read against the P0a curve cell it was predicted from rather than against a
  // geometry re-derived on the node side. `site` is null on everything that is not a matvec, and
  // the three geometry fields are null with it.
  workgroups: count,
  workgroup_size: count,
  site: z.enum(MATVEC_SITES).nullable(),
  rows_per_subgroup: count.nullable(),
  subgroups_per_row: count.nullable(),
  // Which half of an attention layer this dispatch is, so the core and the projections are priced
  // apart in every arm -- including the two that fuse them into one dispatch, where the split is
  // reported as WHOLE and the core-only time comes from the arm that separates them.
  attn_part: count,
});
export type PerDispatch = z.infer<typeof PerDispatchSchema>;

export const ArmDispatchTableSchema = z.object({
  arm: z.string(),
  reps: count,
  dispatches: z.array(PerDispatchSchema),
});
export type ArmDispatchTable = z.infer<typeof ArmDispatchTableSchema>;

export const MPageResultSchema = z.object({
  ready: RuntimeReadySchema,
  runs: z.array(ArmRunSchema),
  thermal_runs: z.array(ArmRunSchema),
  per_dispatch: z.array(ArmDispatchTableSchema),
  per_dispatch_reps: count,
  readback: z.array(ArmRunSchema),
  validation_errors: z.array(z.string()),
  device_lost: z.array(z.string()),
});
export type MPageResult = z.infer<typeof MPageResultSchema>;

// ---- P0a / P0b: the kernel bandwidth curves ----
//
// The chain arm is B1.2's, unchanged: N minimal dispatches in one command buffer, each reading the
// previous one's output, so the chain is serialised by data dependence and the slope of time
// against N is lambda_chain. It is re-measured because the standing correction quotes it, not
// because any gate reads it.
//
// The curve arm is what replaces B1.3's streaming surface. B1.3 priced every dispatch off a
// synthetic streaming kernel and missed every lever, because the real kernels read at 0.1 to 0.8 of
// that rate and the fraction moves with occupancy. Here the kernels ARE the runtime's -- the same
// five int4 matvec classes, at the six sizes the plan dispatches -- swept across occupancy, and the
// published table is BW_kernel(shape, geometry). P0b re-runs the identical sweep with the
// dequantisation variants substituted, so an adoption decision compares like with like.
export const KernelCurveShapeSchema = z.object({
  key: z.enum(MATVEC_SHAPES),
  mode: z.enum(MATVEC_CLASSES),
  rows: z.number().int().min(1),
  cols: z.number().int().min(1),
  // Output rows the kernel writes. It differs from `rows` only for swiglu, which reads a 5120-row
  // matrix and emits 2560.
  emits: z.number().int().min(1),
  zeroPoints: z.boolean(),
  // The residual pre-scale the staged norm applies. It is 1 everywhere except the head, whose
  // tied embedding was exported at half scale, and it is carried per shape rather than inferred so
  // a curve cell runs the byte-for-byte source the runtime's dispatch of that shape runs.
  preScale: finite.positive(),
  weightBytes: count,
  // Weights + scales + zero points + the gamma row + the input vector + the output written, which
  // is the quantity the cost model divides by BW_kernel.
  dispatchBytes: count,
  // The occupancy axis for this shape, as rows per workgroup: the free parameter, since without
  // split-K the workgroup count is exactly ceil(emits / rowsPerWorkgroup).
  rowsPerWorkgroup: z.array(z.number().int().min(1)).min(1),
});
export type KernelCurveShape = z.infer<typeof KernelCurveShapeSchema>;

export const KernelCurveConfigSchema = z.object({
  shape: ModelShapeSchema,
  shapes: z.array(KernelCurveShapeSchema).min(1),
  workgroupSizes: z.array(z.number().int().min(1)).min(1),
  variants: z.array(z.enum(DEQUANT_VARIANT_NAMES)).min(1),
  // Two dispatch counts per cell: the rate is the slope between them, so the per-cell launch cost
  // cancels exactly instead of being fitted.
  dispatchCounts: z.array(z.number().int().min(1)).length(2),
  // The distinct footprint the per-dispatch windows sweep, and the chunk the host fills it in. A
  // cell whose windows fit in cache would measure the cache; the footprint is what says it did not.
  footprintBytes: z.number().int().min(1),
  fillChunkBytes: z.number().int().min(1),
  rounds: z.number().int().min(1),
  // Reps per round are chosen per cell so every cell moves about the same number of bytes per
  // round: a 43 MB cell at 16 dispatches would otherwise cost 65x what a 0.66 MB cell costs.
  targetBytesPerRound: z.number().int().min(1),
  minReps: z.number().int().min(1),
  maxReps: z.number().int().min(1),
  warmupReps: z.number().int().min(1),
  quantBlock: z.number().int().min(1),
  // The three runtime constants a matvec module needs that the model shape does not carry: the
  // norm's epsilon, the staging width the kernel is compiled against and the query count the
  // dispatch runs at. They are passed rather than assumed so a curve cell compiles the same module
  // text the runtime's dispatch of that shape compiles.
  epsilon: finite.positive(),
  maxPositions: z.number().int().min(1),
  maxQueries: z.number().int().min(1),
  seed: count,
  extraFeatures: z.array(z.string()),
  powerPreference: z.enum(["high-performance", "low-power"]),
  expectedSubgroupSize: z.number().int().min(1),
});
export type KernelCurveConfig = z.infer<typeof KernelCurveConfigSchema>;

export const P0aConfigSchema = z.object({
  chainCounts: z.array(z.number().int().min(1)),
  rounds: z.number().int().min(1),
  repsPerRound: z.number().int().min(1),
  warmupReps: z.number().int().min(1),
  workgroupSize: z.number().int().min(1),
  curve: KernelCurveConfigSchema,
  seed: count,
  extraFeatures: z.array(z.string()),
  powerPreference: z.enum(["high-performance", "low-power"]),
  expectedSubgroupSize: z.number().int().min(1),
});
export type P0aConfig = z.infer<typeof P0aConfigSchema>;

export const ChainArmSchema = z.object({
  dispatches: z.number().int().min(1),
  // One median per round; the fit runs on these so a slow round cannot leverage the slope.
  per_round_ms: z.array(nonNegative),
  p50_ms: nonNegative,
  checksum: finite,
});
export type ChainArm = z.infer<typeof ChainArmSchema>;

// One cell of a kernel curve at one dispatch count. Two of these -- same shape, same geometry, same
// variant, different dispatch counts -- make one published BW_kernel rate.
export const KernelCellSchema = z.object({
  shape: z.enum(MATVEC_SHAPES),
  mode: z.enum(MATVEC_CLASSES),
  variant: z.enum(DEQUANT_VARIANT_NAMES),
  bytes: count,
  rows_per_workgroup: z.number().int().min(1),
  workgroups: z.number().int().min(1),
  workgroup_size: z.number().int().min(1),
  subgroup_size: z.number().int().min(1),
  rows_per_subgroup: z.number().int().min(1),
  subgroups_per_row: z.number().int().min(1),
  dispatches: z.number().int().min(1),
  reps: z.number().int().min(1),
  windows: z.number().int().min(1),
  footprint_bytes: count,
  per_round_ms: z.array(nonNegative),
  p50_ms: nonNegative,
  checksum: finite,
});
export type KernelCell = z.infer<typeof KernelCellSchema>;

// A cell the reachability rule refuses: subgroupsPerRow * subgroupSize > cols / quantBlock leaves
// whole subgroups with no quantisation block to reduce, which is idle lanes rather than an
// occupancy point. Skipped cells are published, never measured at zero.
export const SkippedCellSchema = z.object({
  shape: z.enum(MATVEC_SHAPES),
  variant: z.enum(DEQUANT_VARIANT_NAMES),
  rows_per_workgroup: z.number().int().min(1),
  workgroup_size: z.number().int().min(1),
  subgroups_per_row: z.number().int().min(1),
  blocks_per_row: z.number().int().min(1),
  reason: z.enum(["unreachable_partition", "workgroup_limit", "stage_bytes"]),
});
export type SkippedCell = z.infer<typeof SkippedCellSchema>;

export const KernelCurvePageResultSchema = z.object({
  env: PageEnvSchema,
  device: DeviceDescriptorRecordSchema,
  cells: z.array(KernelCellSchema),
  skipped: z.array(SkippedCellSchema),
  validation_errors: z.array(z.string()),
});
export type KernelCurvePageResult = z.infer<typeof KernelCurvePageResultSchema>;

export const P0aPageResultSchema = z.object({
  env: PageEnvSchema,
  device: DeviceDescriptorRecordSchema,
  chain: z.array(ChainArmSchema),
  curve: KernelCurvePageResultSchema,
  validation_errors: z.array(z.string()),
});
export type P0aPageResult = z.infer<typeof P0aPageResultSchema>;

// One matrix a variant is checked on: a real section of the real blob, named rather than
// synthesised, because what a variant trades is the precision of an eight-term dot and the size of
// the error that trade produces depends on the magnitudes the real weights and scales carry.
export const VariantCheckShapeSchema = z.object({
  key: z.enum(MATVEC_SHAPES),
  mode: z.enum(MATVEC_CLASSES),
  rows: z.number().int().min(1),
  cols: z.number().int().min(1),
  emits: z.number().int().min(1),
  zeroPoints: z.boolean(),
  preScale: finite.positive(),
  quantSection: z.string(),
  scalesSection: z.string(),
  zeroPointsSection: z.string().nullable(),
  rowsPerWorkgroup: z.number().int().min(1),
  workgroupSize: z.number().int().min(1),
});
export type VariantCheckShape = z.infer<typeof VariantCheckShapeSchema>;

export const P0bConfigSchema = z.object({
  curve: KernelCurveConfigSchema,
  weightsUrl: z.string(),
  weightBytes: count,
  sections: z.array(z.object({ name: z.string(), offset: count, bytes: count })),
  checks: z.array(VariantCheckShapeSchema).min(1),
  tolerance: finite.positive(),
  // The staged input every checked matrix is run against. It is drawn once from this seed and
  // shared by the carried kernel and every variant, so the only difference between their outputs is
  // the inner product.
  seed: count,
});
export type P0bConfig = z.infer<typeof P0bConfigSchema>;

// P0b's correctness half: an adopted variant's rows against the carried kernel's, on the real
// weights, before any arm runs. A variant that is fast and wrong is not an adoption.
export const VariantCheckSchema = z.object({
  variant: z.enum(DEQUANT_VARIANT_NAMES),
  shape: z.enum(MATVEC_SHAPES),
  rows_checked: count,
  max_abs_diff: nonNegative,
  max_rel_diff: nonNegative,
  reference_norm: nonNegative,
  pass: z.boolean(),
});
export type VariantCheck = z.infer<typeof VariantCheckSchema>;

export const P0bPageResultSchema = z.object({
  curve: KernelCurvePageResultSchema,
  checks: z.array(VariantCheckSchema),
  weights_sha256: z.string(),
  validation_errors: z.array(z.string()),
});
export type P0bPageResult = z.infer<typeof P0bPageResultSchema>;

// ---- P0c: the flash core microbench ----
//
// The standalone split-position core at three depths and seven block counts, on synthetic KV caches
// cycled six slices deep -- a real token reads six different caches and then reads them all again
// on the next token, so one re-read slice would measure a warmer cache than the token has.
export const P0cConfigSchema = z.object({
  shape: ModelShapeSchema,
  positions: z.array(z.number().int().min(1)).min(1),
  blockCounts: z.array(z.number().int().min(1)).min(1),
  cacheSlices: z.number().int().min(1),
  // N2's core and A2's, both at the workgroup size the runtime launches them at: the split core is
  // one thread per head dim, the core it replaces is four position groups of them. Both are carried
  // rather than derived, so the microbench launches the byte-for-byte modules the arms launch.
  flashCoreWorkgroupSize: z.number().int().min(1),
  carriedCoreWorkgroupSize: z.number().int().min(1),
  mergeWorkgroupSize: z.number().int().min(1),
  tilePositions: z.number().int().min(1),
  // The three runtime constants the cores need that the model shape does not carry, plus the rotary
  // base the synthetic tables are built from. P0c reads no blob, so its rotary table is the standard
  // one at the registered base rather than the exported slice; both cores read the identical table,
  // so the substitution moves neither the timing nor the agreement it is compared under.
  epsilon: finite.positive(),
  maxPositions: z.number().int().min(1),
  maxQueries: z.number().int().min(1),
  ropeTheta: finite.positive(),
  dispatchCounts: z.array(z.number().int().min(1)).length(2),
  rounds: z.number().int().min(1),
  repsPerRound: z.number().int().min(1),
  warmupReps: z.number().int().min(1),
  tolerance: finite.positive(),
  seed: count,
  extraFeatures: z.array(z.string()),
  powerPreference: z.enum(["high-performance", "low-power"]),
  expectedSubgroupSize: z.number().int().min(1),
});
export type P0cConfig = z.infer<typeof P0cConfigSchema>;

// One (depth, S) cell at one dispatch count. `merge_ms` is measured in the same command buffer as
// the split pass, because the fold is part of what S costs and pricing the split alone would price
// a core that does not produce an answer.
export const FlashCoreCellSchema = z.object({
  positions: z.number().int().min(1),
  blocks: z.number().int().min(1),
  // The tile trip count baked into the kernel for this cell, and the K/V bytes one layer's core
  // reads at this depth.
  tile_iters: z.number().int().min(1),
  bytes: count,
  dispatches: z.number().int().min(1),
  reps: z.number().int().min(1),
  per_round_ms: z.array(nonNegative),
  p50_ms: nonNegative,
  // The same cell with the merge dispatch omitted, so the fold's share of the cell is measured
  // rather than inferred.
  split_only_p50_ms: nonNegative,
  checksum: finite,
});
export type FlashCoreCell = z.infer<typeof FlashCoreCellSchema>;

// A2's core measured on the identical synthetic caches at the identical depths, so P0c's curve is
// read against the thing N2 has to beat rather than against B1.3's number from another session.
export const CarriedCoreCellSchema = z.object({
  positions: z.number().int().min(1),
  bytes: count,
  dispatches: z.number().int().min(1),
  reps: z.number().int().min(1),
  per_round_ms: z.array(nonNegative),
  p50_ms: nonNegative,
  checksum: finite,
});
export type CarriedCoreCell = z.infer<typeof CarriedCoreCellSchema>;

export const P0cPageResultSchema = z.object({
  env: PageEnvSchema,
  device: DeviceDescriptorRecordSchema,
  cells: z.array(FlashCoreCellSchema),
  carried: z.array(CarriedCoreCellSchema),
  // The flash core's output against the streaming core's on the same synthetic cache, per cell.
  // P0c is a microbench, but a microbench of a wrong kernel prices nothing.
  agreement: z.array(
    z.object({
      positions: z.number().int().min(1),
      blocks: z.number().int().min(1),
      values: count,
      max_abs_diff: nonNegative,
      max_rel_diff: nonNegative,
      reference_norm: nonNegative,
      pass: z.boolean(),
    }),
  ),
  validation_errors: z.array(z.string()),
});
export type P0cPageResult = z.infer<typeof P0cPageResultSchema>;

// ---- P3: the context sweep ----
//
// HIT-CONTEXT is adjudicated here, so this stage is no longer a record: the winner AND the carried
// baseline are swept at the same depths in the same session, interleaved point by point, and the
// core slope is re-fitted for both. The cache is primed by free-running the runtime from an empty
// cache to the target position, never by writing a position into the state, so the keys and values
// it holds are the ones the model would actually have produced.
//
// Each depth is its own runtime: the plan's occupancy, N2's block count and the baked tile trip
// count are all functions of the depth, so one runtime built at 192 could not honestly price 2048.
export const ContextConfigSchema = z.object({
  runtimes: z.array(z.object({ positions: z.number().int().min(1), runtime: RuntimeConfigSchema })).min(1),
  promptIds: z.array(count),
  positions: z.array(z.number().int().min(1)),
  blocks: z.number().int().min(1),
  tokensPerBlock: z.number().int().min(1),
  warmupSteps: z.number().int().min(1),
  // The winner and the baseline, in the order they are run at every depth.
  arms: z.array(z.string()).min(1),
  baselineArm: z.string(),
  perDispatchReps: z.number().int().min(1),
  timestampCapacityPairs: z.number().int().min(1).max(8192),
  seed: count,
});
export type ContextConfig = z.infer<typeof ContextConfigSchema>;

export const ContextBlockSchema = z.object({
  arm: z.string(),
  block: z.number().int().min(0),
  start_position: count,
  tokens: count,
  wall_ms: nonNegative,
  per_token_ms: z.array(nonNegative),
  gpu_busy_ms: finite,
  gpu_span_ms: finite,
});
export type ContextBlock = z.infer<typeof ContextBlockSchema>;

export const ContextPointSchema = z.object({
  position: count,
  // The block count each arm's flash core ran at this depth, keyed by arm in `arms` order. An arm
  // without a flash core records 1.
  flash_blocks: z.array(z.object({ arm: z.string(), blocks: z.number().int().min(1) })),
  blocks: z.array(ContextBlockSchema),
  // The per-dispatch table at this depth, one per arm, which is what separates the core's growth
  // from the projections' flatness instead of inferring both from the total.
  per_dispatch: z.array(ArmDispatchTableSchema),
  per_dispatch_reps: count,
  snapshot: z.array(z.object({ arm: z.string(), snapshot: HookSnapshotSchema, tokens: z.number().int().min(1) })),
});
export type ContextPoint = z.infer<typeof ContextPointSchema>;

export const ContextPageResultSchema = z.object({
  ready: RuntimeReadySchema,
  positions: z.array(ContextPointSchema),
  validation_errors: z.array(z.string()),
  device_lost: z.array(z.string()),
});
export type ContextPageResult = z.infer<typeof ContextPageResultSchema>;
