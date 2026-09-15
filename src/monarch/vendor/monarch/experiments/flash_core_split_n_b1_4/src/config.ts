// MONARCH B1.4 FLASH_CORE_SPLIT_N registered constants.
// Every number that steers a measurement, a statistic, a gate, or a trim decision lives here.
// Nothing downstream defaults anything. The B1_3 block carries the baseline arm this unit's levers
// are priced against; it is read back from B1.3's artifacts at P0 and asserted against these
// values, so a stale copy fails loudly instead of silently re-pricing the prediction.

export const UNIT = {
  name: "flash_core_split_n_b1_4",
  spec: "MONARCH B1.4 FLASH_CORE_SPLIT_N",
  // B1.0 wrote O6, B1.1 appended O6.1, B1.2 O6.2 and B1.3 O6.3 on the same cell. This unit is the
  // fifth line on that cell.
  priorClosureCell: "O6.3",
  closureCell: "O6.4",
  closureId: "WEBGPU_LAPTOP_FUSED_DECODE_LFM2_5",
  // The baseline arm A2, its published numbers, its measured per-kind table and the streaming
  // surface this unit retires all come from B1.3.
  referenceUnit: "attention_bandwidth_b1_3",
  // The weight blob is B1.1's, as in B1.3: this unit's repack set is smaller still (no arm here
  // splits K), so B1.1's blob is the prefix and every shared section is compared by content digest.
  weightsUnit: "fused_runtime_probe_b1_1",
  // The ONNX file the extended rotary tables are sliced out of, and B1.0's instrument constants.
  instrumentUnit: "webgpu_pricing_probe_b1_0",
  // The pinned teacher-forced reference battery, reused unchanged at 192 positions.
  correctnessUnit: "hit_fusion_b1_2",
} as const;

export const PINS = {
  bun: "1.3.11",
  playwright: "1.62.1",
  chromiumBuild: "1234",
  chromiumVersion: "151.0.7922.34",
  transformersJs: "4.2.0",
  onnxruntimeWeb: "1.26.0-dev.20260416-b7804b056c",
  zod: "4.4.3",
  typescript: "7.0.2",
} as const;

export const SERVER = {
  host: "127.0.0.1",
  port: 47315,
  transformersBundle: "dist/transformers.js",
  transformersRoute: "/vendor/transformers/transformers.js",
  ortRoute: "/vendor/ort/",
  ortWasmFiles: ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"],
  modelsRoute: "/models/",
  weightsRoute: "/weights/",
  pageRoute: "/page.js",
  hooksRoute: "/hooks.js",
} as const;

export const CHROMIUM = {
  channel: "chromium",
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--ignore-gpu-blocklist",
    "--enable-dawn-features=allow_unsafe_apis",
    "--disable-dawn-features=timestamp_quantization",
  ],
  requiredFeatureStatus: "enabled",
  requiredGpuText: "Hardware accelerated",
  adapterPowerPreference: "high-performance",
} as const;

export const ORT = {
  numThreads: 1,
  proxy: false,
  logSeverityPreflight: 0,
  logSeverityTiming: 2,
  logVerbosity: 0,
  device: "webgpu",
  dtype: "q4",
  flushEveryDispatches: 16,
  extraDeviceFeatures: ["timestamp-query", "shader-f16", "subgroups"],
} as const;

export interface ModelFile {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

// The single model of this unit. Weight files are the byte-identical copies B1.0 already
// downloaded and sha256-pinned; this unit re-verifies the digests and never re-downloads. Only
// the rotary tables are read out of them here -- every matrix comes from B1.1's blob.
export const MODEL = {
  key: "lfm2_5_230m",
  repo: "LiquidAI/LFM2.5-230M-ONNX",
  revision: "c6f46e4e3f885ebcad164d14059a49f90e27eb4d",
  files: [
    { path: "config.json", bytes: 1668, sha256: "c09361ba08a21a464011710ade1bab1dbe7a9c43eadb70cae04ebb4825ff8233" },
    { path: "generation_config.json", bytes: 131, sha256: "85fa3172f3838eefa602843e3d97fbf532aeb585e0d7fb869dcd17c268e77f45" },
    { path: "tokenizer.json", bytes: 4733389, sha256: "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd" },
    { path: "tokenizer_config.json", bytes: 5347, sha256: "c46e3f5715c73f7ae9beeeebad8f7187fd647d2de352c3cd01fe250c88d2f960" },
    { path: "onnx/model_q4.onnx", bytes: 154010, sha256: "82a442c44d3d143432edff57984a4e7e8da65179d3a960a170d294aed8c5dd8d" },
    { path: "onnx/model_q4.onnx_data", bytes: 211111936, sha256: "b51a4580a88a2cd0486032cfdaf8694b09359abca5a97df8deb0a314ea6e5b34" },
  ] as readonly ModelFile[],
  onnxFile: "onnx/model_q4.onnx",
  onnxDataFile: "onnx/model_q4.onnx_data",
  nominalParams: 230_000_000,
} as const;

// Architecture, read out of config.json at P0 and asserted against these registered values.
export const SHAPE = {
  layers: 14,
  layerTypes: [
    "conv", "conv", "attention", "conv", "attention", "conv", "attention",
    "conv", "attention", "conv", "attention", "conv", "attention", "conv",
  ],
  hidden: 1024,
  ffn: 2560,
  vocab: 65536,
  heads: 16,
  kvHeads: 8,
  headDim: 64,
  convCache: 3,
  convBias: false,
  normEps: 1e-5,
  ropeTheta: 1_000_000,
  ropeInterleaved: false,
  tieEmbedding: true,
  quantBits: 4,
  quantBlock: 32,
  maxPositionEmbeddings: 128_000,
  bosTokenId: 1,
  eosTokenId: 7,
} as const;

// Storage precision of the fused runtime, carried from B1.2 unchanged. The residual stream and the
// logits stay fp32 because the residual accumulates 28 adds and the logits are compared against an
// fp32 reference.
export const PRECISION = {
  weightBits: 4,
  residualBytes: 4,
  hiddenBytes: 2,
  kvCacheBytes: 2,
  convCacheBytes: 2,
  logitsBytes: 4,
  accumulate: "f32",
} as const;

// Encoding invariants shared by every arm. The per-layer dispatch counts are not registered here:
// planStep() in src/pure/plan.ts is the single source of truth for the ordered list and ARMS
// carries the count each arm's plan must produce.
export const FUSION = {
  commandBuffersPerToken: 1,
  passesPerToken: 1,
  readbackEveryTokens: 32,
  readbackRingTokens: 64,
  // B1.1's export concatenates q/k/v into one matrix and gate/up into one, and every arm of this
  // unit reads those concatenations, so the blob's section set is B1.1's and A2CM's plan names the
  // same sections at the same offsets. Registered rather than assumed because the plan branches on
  // them and a different export would change the layout, not just the kernels.
  concatQkv: true,
  concatGateUp: true,
} as const;

// ---- P1 levers ----
//
// Two independent levers, so four arms.
//
// N1 SPLIT-N. Every matvec in the token becomes a split-N dispatch that owns whole output rows and
// reads the whole input vector: no K-split, no partial slice, no fold. The fused MLP is replaced by
// gate/up+swiglu and down; the attention and conv projections are already split-N and the lever
// moves them to the occupancy P0a measures as their knee. Because nothing defers a sum, the fold
// dispatch disappears from the plan on its own -- there is no pending partial for it to reduce.
//
// N2 FLASH CORE. The attention core is split across position blocks: S workgroups per KV head,
// each running an online-softmax pass over its own slice of the KV cache with K and V staged a tile
// at a time, then one index-ordered log-sum-exp merge per attention layer. S = 1 of this kernel is
// still a tiled online-softmax pass and is NOT A2's core; A2's core is measured separately, which
// is what prices the restructuring against the kernel it replaces rather than against itself.
//
// The deferred-reduction contract is carried unchanged from B1.3 for the arms that still fuse:
//
//   1. a fused dispatch stages x[i] = residual_cur[i] + sum over the pending partial slices;
//   2. it writes residual_next[i] = residual_cur[i] + that same sum, for rows no other workgroup
//      of the dispatch writes;
//   3. it writes its own output as `slices` fresh slices, one per workgroup.
//
// Reads land on residual_cur and partials_cur, writes on residual_next and partials_next, and
// slices are summed in index order in every consumer, so every sum is bitwise reproducible.
export const LEVERS = {
  // B1.3's A2 fused MLP, carried unchanged as the baseline arm's kernel. 2560/80 = 32 rows = one
  // quantisation block of the down projection's K, the largest slice count the reduction admits.
  mlpFused: { workgroupSize: 256, slices: 80 },
  // B1.3's A2 attention core: one workgroup per KV head carrying both its q heads, streaming K and
  // V from global memory, scores held for every position at once. The kernel N2 replaces.
  attnSplit: { coreWorkgroupSize: 256 },
  // N2's flash core. One workgroup per (KV head, position block); workgroupSize = headDim so a
  // thread owns one position in the score pass and one dimension in the value pass. K and V are
  // staged one TILE of positions at a time -- 2 * 64 * 64 * 2 B = 16 384 B, inside the 32 768 B
  // budget at every S and every context length, which is what lets S = 1 run at 2048 positions.
  // A block's partial is 64 accumulator lanes plus its running maximum and denominator, padded to
  // 68 f32 so consecutive blocks start 272 B apart and no two share a 128 B line boundary.
  flash: {
    coreWorkgroupSize: 64,
    tilePositions: 64,
    maxBlocks: 32,
    positionsPerBlockRule: 64,
    mergeWorkgroupSize: 64,
    partialWords: 68,
  },
  // The reduction that clears pending partials before any wide consumer, carried unchanged. Only
  // the arms that keep the fused MLP emit it; the N1 arms have no pending partial at any point.
  fold: { workgroupSize: 256, threadsPerRow: 8 },
  // B1.3's two registered matvec geometries, carried so the baseline arm is byte- and
  // occupancy-identical to the arm B1.3 published. N1 does not choose between them: it reads its
  // geometry per dispatch class out of P0a's measured knee.
  geometry: {
    base: { workgroupSize: 128, subgroupSize: 32, rowsPerSubgroup: 4, subgroupsPerRow: 1 },
    wide: { workgroupSize: 128, subgroupSize: 32, rowsPerSubgroup: 1, subgroupsPerRow: 1 },
  },
} as const;

// The registered staging choice for the attention core, and the numbers that force it. B1.3's core
// staged nothing: K and V for one KV head at 192 positions are 49 152 B and at 2048 positions
// 524 288 B, both above the 32 768 B workgroup budget, so it streamed both from global memory and
// held one f32 score per position -- 20 480 B of workgroup memory at maxPositions, which is the
// allocation that caps its occupancy at eight workgroups. The flash core stages a fixed 64-position
// TILE of K and V instead, 16 384 B independent of context length, and holds 64 scores per q head.
export const CORE_STAGING = {
  choiceBaseline: "stream_global",
  choiceFlash: "tile_staged_split_positions",
  stagedBytesIfBoth192: 49_152,
  stagedBytesIfBoth2048: 524_288,
  flashTileStagedBytes: 16_384,
  workgroupBudgetBytes: 32_768,
} as const;

// The arms. A2 is B1.3's winner, carried unchanged and rerun interleaved ABAB in every block. N1
// and N2 are independent, so both singles run and each lever is priced alone AND in combination.
//
//   A2      74 = 1 embed + 8 conv x (fold + 3) - 1 + 6 attn x (fold + 3) + 14 mlp.fused
//                + head.fold + head + 2 sample
//   A2N1    74 = 1 embed + 14 layers x (3 + 2 mlp) + head + 2 sample, and no fold anywhere
//   A2N2    80 = A2 with one merge dispatch added per attention layer
//   A2N1N2  80 = A2N1 with one merge dispatch added per attention layer
export const ARMS = [
  { key: "A2", splitN: false, flash: false, dispatches: 74 },
  { key: "A2N1", splitN: true, flash: false, dispatches: 74 },
  { key: "A2N2", splitN: false, flash: true, dispatches: 80 },
  { key: "A2N1N2", splitN: true, flash: true, dispatches: 80 },
] as const;
export type ArmKey = (typeof ARMS)[number]["key"];

export const BASELINE_ARM: ArmKey = "A2";

// The arm whose dispatch names reproduce B1.1's section set exactly -- every matrix read by its own
// dispatch, none fused -- so the blob laid out from its plan places every B1.1 section at B1.1's
// offset. A2N1 fuses nothing: it is that arm and it is measured.
export const LAYOUT_ARM: ArmKey = "A2N1";

// One timed session. All four arms interleave ABAB against A2 inside every block, so every lever's
// delta is a paired difference under one thermal and version envelope.
export const M_SESSION = ["A2N1", "A2N2", "A2N1N2"] as const;

// Dispatch count is recorded, never gated. Standing correction: lambda_chain is 0.88 us, so D is
// not a cost and bytes per dispatch is. No gate reads D.
export const D_RECORD = { baseline: 74 } as const;

// B1.1's extracted blob. Its bytes are a prefix of this unit's blob: this unit appends the K-sliced
// repacks the carried fused MLP reads, plus the extended rotary tables, and moves nothing.
export const REFERENCE_WEIGHTS = {
  fileSha256: "d26cca605835d4e14584f6861880094b4e3b34f793bcaef850bc33bcd581ab35",
  fileBytes: 145_054_720,
  manifestSha256: "121772dd59889f5b4c4a613fb53a3020a3a0bbf76bf13eaca70feea4ac1fb3dd",
  sections: 169,
} as const;

// B1.3's blob, for the content-digest comparison of the sections the two units share. B1.3 is not
// a prefix of this blob -- it appended A2C's six attention out-projection repacks where this unit
// appends none -- so the shared sections sit at different offsets and are reconciled by content.
// This unit builds only the repacks its own arms need: no arm here splits K except the carried
// fused MLP, so the fourteen down-projection repacks are the whole set, a strict subset of B1.3's
// twenty. Every section present in both blobs must be identical byte for byte, which is what is
// checked -- the baseline arm is B1.3's A2 and has to read exactly what B1.3's A2 read.
export const REFERENCE_BLOB_B1_3 = {
  fileSha256: "82f8825f03f50fbcd96f89262c21fccff44a80bb1a6feb3e41db9dd6fc0bf80e",
  fileBytes: 172_579_840,
  manifestSha256: "3eeb4fbb0c750392a774c0ac837ebbdaee9730db9c754e8a88699cc2b17f50f0",
  sections: 211,
  repacks: 20,
} as const;

// B1.2's teacher-forced pin, carried. The eight prompts and the greedy continuation they generate
// are the same token streams B1.2 and B1.3 measured, so the shallow correctness battery is a rerun
// of B1.2's, not a new one that happens to use the same corpus. Preflight regenerates the
// continuation from the same reference and refuses to proceed unless the digest matches this.
export const REFERENCE_PIN = {
  sha256: "71f41fa81b2fc5d8955cc76a3adc2adee6ccabb60d8af91b33da4381ddf61298",
  prompts: 8,
  continuationTokens: 256,
} as const;

export const RUNTIME = {
  // B1.1 and B1.2 extracted a 1024-row prefix of the exported rotary tables, which caps every
  // measured position below 1024. P3 measures at 2048, so this unit appends a 2560-row extension
  // sliced out of the same ONNX tensors -- not recomputed: numpy's float32 powf and JS's
  // fround(Math.pow) differ by 1 ulp on 8.8% of the cos entries and 7.0% of the sin entries, so a
  // reproduction would not be bitwise. The first 1024 rows of the extension are asserted byte
  // identical to B1.1's tables, which is what proves the slice is the same tensor.
  ropeRows: 1024,
  ropeRowsExtended: 2560,
  // The exported final norm is SkipSimplifiedLayerNormalization(x, x) = RMSNorm(2x) * gamma.
  // RMSNorm is only scale-invariant up to eps, so the factor is applied, not dropped.
  headPreScale: 2,
  sectionAlignment: 256,
  weightsFile: "weights.bin",
  manifestFile: "weights_manifest.json",
  // Every arm of this unit is decode-only: the fused blocks are written for one query token, and
  // the split arm's core carries one query per workgroup. The warm cache is therefore built by
  // stepping the arm under test one token at a time over the pinned prompt rather than by a
  // separate multi-query prefill program, which is also what removes B1.0's prefill/decode split.
  maxQueries: 1,
} as const;

// Kernel geometry. subgroupSize is asserted against the adapter at preflight, never assumed.
export const KERNELS = {
  // maxCols bounds the staged activation row, not any measured matrix: the decode shapes stage
  // 1024 (hidden) and 2560 (ffn) columns, and 2560 is the widest K any arm of this unit reads.
  matvec: { workgroupSize: 128, subgroupSize: 32, rowsPerSubgroup: 4, subgroupsPerRow: 1, maxCols: 2560, quantBlock: 32 },
  // The same kernel at one row per subgroup: 4 rows per workgroup instead of 16, so 4x the
  // workgroups over the same matrix. B1.3's C1/M1 lever, carried as the baseline arm's geometry.
  matvecWide: { workgroupSize: 128, subgroupSize: 32, rowsPerSubgroup: 1, subgroupsPerRow: 1, maxCols: 2560, quantBlock: 32 },
  conv: { workgroupSize: 256 },
  embed: { workgroupSize: 256 },
  sample: { workgroupSize: 256, partials: 256 },
  // The scores array in every attention kernel is sized by this, so it bounds P3's context sweep.
  maxPositions: 2560,
} as const;

// The fused kernels bind the whole weight blob once and index it with offsets baked in at pipeline
// build time. This adapter caps maxStorageBuffersPerShaderStage at 10; folding weights, scales,
// gammas and the rotary tables into one `blob` binding takes every fused kernel to six or fewer.
// The residual and partials appear twice because the deferred reduction ping-pongs: a dispatch
// reads the parity it was planned on and writes the other one, and the bind group decides which
// buffer is which. B1.3's fused attention BLOCK is gone from every arm of this unit -- A2 lifted
// the projection out into an ordinary matvec and N2 replaces what was left -- so the widest
// binding set here is six.
//   attn.core       blob, state, key_cache, value_cache, projection, destination           = 6
//   attn.core_flash blob, state, key_cache, value_cache, projection, flash                 = 6
//   attn.merge      flash, destination                                                     = 2
//   mlp.fused       blob, residual in/out, partials in/out, state                          = 6
//   fold            residual out (in place), partials in, state                            = 3
//
// A2's projection dispatch is NOT on this list: it is the ordinary matvec kernel at wide geometry
// with a new output mode, so it binds weights, scales, gamma, residual, projection and state as
// six separate buffers like every other matvec. That is the point of the arm -- the projection is
// an ordinary wide split-N matmul, priced on the same curve as the MLP's.
export const FUSED_BINDINGS = {
  blob: 0,
  residualIn: 1,
  residualOut: 2,
  partialsIn: 3,
  partialsOut: 4,
  state: 5,
  keyCache: 6,
  valueCache: 7,
  // A2's core alone: the f32 q/k/v scratch the projection wrote, and the f16 attention output the
  // out-projection reads as its source.
  projection: 8,
  destination: 9,
  // N2's per-position-block partials. A binding NUMBER, not an eleventh buffer: the flash core
  // binds six and the merge binds two.
  //   attn.core_flash blob, state, key_cache, value_cache, projection, flash                = 6
  //   attn.merge      flash, destination                                                    = 2
  flash: 10,
} as const;

// What the kernels need from whatever GPUDevice they run on. The runtime adopts ORT's device
// rather than creating one, so these are asserted, never requested;
// tests/device_minimums.test.ts holds them to the real kernels.
export const DEVICE_MINIMUMS = {
  maxStorageBuffersPerShaderStage: 10,
  maxBindingsPerBindGroup: 18,
  maxBindGroups: 1,
  // A2's streaming core is the largest allocation: reduce[256] 1024 + two score rows at
  // maxPositions 20 480 + q for two heads and one shared k row 768 + vcur 256 + vacc[256] 1024
  // = 23 552 B, inside the adapter's measured 32 768 B. The flash core allocates 18 176 B at every
  // S and every context length -- a 64-position K tile and V tile at 8 192 B each, q for two heads
  // and one k row 768, vcur 256, reduce[64] 256, and one 64-wide score row per q head 512 -- which
  // is the whole point of staging a fixed tile.
  maxComputeWorkgroupStorageSize: 23_552,
  maxComputeWorkgroupStorageSizeFlash: 18_176,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  // The head matvec at one row per workgroup would be 65 536; P0a sweeps it from 16 rows per
  // workgroup up, so its widest cell is 4096. The widest non-head cell is the conv in-projection
  // at one row per workgroup, 3072.
  maxComputeWorkgroupsPerDimension: 4096,
} as const;

// Prompt corpus: eight fixed prompts, carried from B1.2 verbatim so the correctness reference and
// the pinned continuation are the same token streams.
export const PROMPTS: readonly string[] = [
  "The lighthouse keeper wrote in his logbook every evening, recording the weather, the tides, " +
    "and the names of the ships that passed the point before dark. On the third of November the " +
    "wind rose from the northwest and the glass fell steadily through the afternoon, and by six " +
    "o'clock the horizon had closed to a grey wall. He noted the barometer, trimmed the lamp, and " +
    "went down to the landing to watch for the fishing fleet that would be returning to the harbour, " +
    "their lamps swinging in the swell, and beyond them the long grey line of the open sea where the " +
    "weather was made.",
  "A compiler turns source text into machine code in stages. The lexer groups characters into " +
    "tokens, the parser builds a syntax tree from those tokens, the type checker walks the tree and " +
    "annotates it, and the code generator finally emits instructions for a particular processor. " +
    "Optimisation passes run between the middle stages, rewriting the intermediate representation " +
    "into a cheaper program that computes the same result.",
  "The recipe calls for four cups of flour, two teaspoons of salt, a packet of dried yeast, and " +
    "enough warm water to bring the dough together. Knead it on a floured board for ten minutes, " +
    "until the surface is smooth and springs back when pressed, then leave it covered in a warm " +
    "place until it has doubled in size.",
  "In the winter of 1846 the survey party reached the eastern bank of the river and found the ford " +
    "impassable. They camped for eleven days while the water fell, mending harness and drying stores, " +
    "and on the twelfth morning the guide led the wagons across in single file with the water running " +
    "level with the axles.",
  "To prove the statement we argue by contradiction. Suppose there were a smallest positive rational " +
    "number with the stated property. Divide it by two: the result is positive, rational, and smaller, " +
    "and a short calculation shows that it has the property as well, which contradicts the assumption " +
    "that the first number was smallest. No such smallest number can exist, and the same argument " +
    "applies unchanged to any set closed under halving.",
  "The patient presented with a three-day history of fever, dry cough, and pleuritic chest pain on " +
    "the left side. Examination revealed reduced breath sounds at the left base and dullness to " +
    "percussion. A chest radiograph confirmed consolidation of the left lower lobe, and oral " +
    "antibiotics were started the same afternoon. Temperature settled within forty-eight hours and " +
    "the patient was reviewed in clinic a fortnight later with a repeat film.",
  "Interest rates rose through the second quarter and the housing market cooled sharply, with " +
    "transaction volumes falling by nearly a fifth against the same period last year. Builders " +
    "responded by slowing starts rather than cutting prices, so the stock of unsold new homes stayed " +
    "close to its long-run average despite the drop in demand. Rents continued to climb over the " +
    "same months, and the gap between owning and renting widened in every region.",
  "She had not been back to the house in eleven years, and the road up from the village was narrower " +
    "than she remembered, the hedges grown in over the passing places. The gate was open. Somebody " +
    "had cut the grass in the front field recently, and the smell of it came through the car window " +
    "with the dust.",
] as const;

export const PROMPT = {
  count: 8,
  targetTokens: 64,
  continuationTokens: 256,
  // The deeper pin, generated in this unit's preflight by the same greedy reference and asserted to
  // extend the carried one token for token over its first 256. Its length is set by the deepest
  // thing that replays it, which is the flash-core check at 2048 attended positions: the check
  // primes 2047 tokens and encodes one more, so the stream must hold 2048 ids and the prompt is
  // exactly 64 of them.
  deepContinuationTokens: 1984,
  // The registered measurement depth, in ATTENDED positions -- the KV rows the attention core reads,
  // which is one more than the token position, since a step at position p attends p + 1 rows. It is
  // the depth HIT-1400 names, the depth P0d prices and the depth P0c measures a block count at, and
  // it sits inside the span the timed runs walk: prompt (64) + 127 continuation tokens decoded, so
  // the table is a picture of a step in the middle of the run.
  warmPosition: 192,
  seed: 111003,
} as const;

export const PROTOCOL = {
  warmupSteps: 20,
  m1Tokens: 256,
  // Every arm is interleaved ABAB against the A2 baseline inside every block, so a lever's delta is
  // a paired difference under one thermal and version envelope rather than two sessions.
  interleaveOrder: ["A_arm", "B_baseline", "A_arm", "B_baseline"],
  m1Blocks: 16,
  m1BlocksAfterL4: 8,
  m1PromptIndex: 0,
  thermalRepeatBlocks: 1,
  thermalDriftThreshold: 0.1,
  thermalCooldownSeconds: 120,
  thermalMaxReruns: 1,
  m3PerDispatchReps: 64,
  mSyncRuns: 4,
  m5RunsPerVariant: 4,
  readbackVariants: [1, 32],
  determinismRuns: 3,
  determinismTokens: 256,
  correctnessTokens: 256,
  correctnessPrompts: 8,
  correctnessTop1Min: 0.99,
  correctnessLogitStride: 16,
  dequantLayerIndex: 0,
  dequantMatrix: "conv.in_proj",
  dequantRowSamples: 64,
  dequantSeed: 111_006,
  dequantTolerance: 0.005,
  deferredTolerance: 0.005,
  // The flash core check: the flash core's attention output for one layer against A2's streaming
  // core, on the same staged projection and the same cache, at BOTH 192 and 2048 positions. The
  // log-sum-exp merge is the new numerics -- A2 takes one maximum over the whole row and this takes
  // S maxima and rescales -- so the two agree to f32 rounding of a different summation order, not
  // bit for bit. Both maximum absolute and maximum relative difference are published.
  flashCoreTolerance: 0.005,
  flashCoreLayerIndex: 2,
  flashCorePositions: [192, 2048],
  // The split-N check: A2N1's MLP output for one layer against A2's fused MLP, on the same staged
  // residual. The fused block sums 80 partial slices in index order and the split-N pair sums whole
  // rows, so again the bound is f16 relative eps with headroom.
  splitNTolerance: 0.005,
  splitNLayerIndex: 2,
  // The correctness battery runs at both registered depths. 192 reuses B1.2's pinned reference; the
  // 1024 pin is generated in this unit's preflight and its sha256 recorded.
  correctnessPositions: [192, 1024],
  // The battery compares this many teacher-forced positions, centred on the depth it adjudicates.
  // The deeper battery is therefore the SAME 256-position comparison made at 1024 attended
  // positions, not a longer one made at 192: what changes between the two is the depth of the cache
  // every compared row was produced against, which is the only thing HIT-CONTEXT is about.
  correctnessCompareTokens: 256,
  bootstrapReplicates: 10_000,
  seedTiming: 111001,
  seedBootstrap: 111002,
  seedSampling: 111004,
  timestampCapacityPairs: 2048,
  liveGpuBytesGrowthMaxPerBlock: 16 * 1024 * 1024,
} as const;

// B1.0's published instrument constants, carried for the chain-bandwidth comparison only.
export const B1_0 = {
  aSubmitMs: 0.14156581472167018,
  bDispatchMs: 0.0018426071960334117,
  matvecGBps: 433.388098065848,
  streamGBps: 482.7521176014881,
  chainBytes: 144_572_416,
  chainDispatches: 83,
  chainP50Ms: 0.9575004577636719,
  chainEffectiveGBps: 182.53579475241384,
  tArMs: 2.6649999618530273,
  dispatchesPerStep: 311,
  submitsPerStep: 23,
} as const;

// B1.3's published A2 numbers: the baseline this unit's levers are priced against. A2 is rerun
// interleaved in this session, so every one of these is re-measured; the registered values are
// asserted against B1.3's artifacts at P0 and the session's own A2 arm is what the deltas use.
export const B1_3 = {
  arm: "A2",
  tArMs: 0.9134179688990116,
  tArCi: [0.9093359373509884, 0.9183837891556322],
  tokensPerSecond: 1094.789060483833,
  dispatches: 74,
  submits: 1,
  idleFraction: 0.002299657216024431,
  gpuBusyMs: 0.9114946247558594,
  perDispatchTotalGpuMs: 1.0283017656249998,
  overlapScale: 0.8864077211827549,
  weightBytes: 144_572_416,
  totalBytes: 157_643_932,
  chainGBps: 233.48743992265162,
  correctnessTop1: 1,
  // A2's attention split, the number this unit's flash core has to beat.
  attn: {
    layers: 6,
    projectionMs: 0.067964758016862,
    coreMs: 0.18237059101542955,
    outProjectionMs: 0.0575796605559674,
    blockMs: 0.30791500958825896,
    coreMsPerLayer: 0.03039509850257159,
    projectionGBps: 95.2350103327301,
    coreGBps: 13.433744916650094,
  },
  groupMs: {
    attention: 0.30791500958825896,
    mlp: 0.3024465803047141,
    conv: 0.18301351361569995,
    other: 0.11811952124718657,
  },
  // A2's measured per-kind table. `scaledMs` is the per-dispatch instrument's time scaled to the
  // token's own busy time, which is the figure every carried price in this unit is written against.
  // The two matvec_residual rows are the two shapes, separated by their occupancy: the attention
  // out-projection at 256 workgroups (1024 rows at 4 per workgroup) and the conv out-projection at
  // 64 (1024 rows at 16).
  perKind: [
    { kind: "mlp_fused", dispatches: 14, scaledMs: 0.24277595318507983, totalBytes: 73_572_352, workgroups: 80, workgroupSize: 256, effectiveGBps: 268.62257164596065 },
    { kind: "attn_core_qkv", dispatches: 6, scaledMs: 0.18237059101542955, totalBytes: 2_449_920, workgroups: 8, workgroupSize: 256, effectiveGBps: 11.907775218518227 },
    { kind: "norm_head", dispatches: 1, scaledMs: 0.10198958764506526, totalBytes: 43_261_952, workgroups: 4096, workgroupSize: 128, effectiveGBps: 375.99650289490285 },
    { kind: "norm_matvec", dispatches: 8, scaledMs: 0.0961166794382483, totalBytes: 15_843_328, workgroups: 192, workgroupSize: 128, effectiveGBps: 146.11041861317634 },
    { kind: "attn_proj", dispatches: 6, scaledMs: 0.067964758016862, totalBytes: 7_962_624, workgroups: 512, workgroupSize: 128, effectiveGBps: 103.84987161616901 },
    { kind: "fold", dispatches: 14, scaledMs: 0.0596706271196343, totalBytes: 4_702_208, workgroups: 32, workgroupSize: 256, effectiveGBps: 69.85134360077537 },
    { kind: "matvec_residual_attn", dispatches: 6, scaledMs: 0.0575796605559674, totalBytes: 3_993_600, workgroups: 256, workgroupSize: 128, effectiveGBps: 61.47931129039242 },
    { kind: "matvec_residual_conv", dispatches: 8, scaledMs: 0.052760483377827065, totalBytes: 5_324_800, workgroups: 64, workgroupSize: 128, effectiveGBps: 89.4598292429125 },
    { kind: "conv_core", dispatches: 8, scaledMs: 0.034136350799624564, totalBytes: 262_144, workgroups: 4, workgroupSize: 256, effectiveGBps: 6.807009543160885 },
    { kind: "sample_partial", dispatches: 1, scaledMs: 0.007967129548835106, totalBytes: 264_192, workgroups: 256, workgroupSize: 256, effectiveGBps: 29.393500788367962 },
    { kind: "sample_final", dispatches: 1, scaledMs: 0.004363314307280823, totalBytes: 2_056, workgroups: 1, workgroupSize: 256, effectiveGBps: 0.41767659774376426 },
    { kind: "embed", dispatches: 1, scaledMs: 0.0037994897460053734, totalBytes: 4_756, workgroups: 1, workgroupSize: 256, effectiveGBps: 1.1095582311749763 },
  ],
  // The per-dispatch prices this unit CARRIES rather than re-derives. Every kind here is a kernel
  // no arm of this unit changes and no kernel curve covers -- the fused MLP block, the conv core,
  // the fold, the embedding lookup and the two sampling scans -- so its P0d price is B1.3's own
  // measurement of that kernel in the real token, divided by the dispatch count. Instrument
  // ESTIMATED, because it is a measurement transported across sessions rather than one made here.
  perDispatchMs: {
    mlp_fused: 0.017341139513219989,
    conv_core: 0.0042670438499530705,
    fold: 0.00426218765140245,
    embed: 0.0037994897460053734,
    sample_partial: 0.007967129548835106,
    sample_final: 0.004363314307280823,
    // A2's streaming attention core at 192 positions. The flash core replaces exactly this.
    attn_core_qkv_192: 0.030395098502571591,
  },
  lambdaChainMs: 0.0008813484261433283,
  // P3's fitted context slope for A2: the core grows 0.536 ms per 1000 cached positions and
  // everything else 0.022. At 1024 positions that is 1.36 ms and at 2048 2.15 ms, which is what
  // HIT-CONTEXT asks this unit to take under 1.000 ms.
  contextSlope: {
    coreMsPer1000: 0.53635,
    coreMsPer1000Ci: [0.53244, 0.53893],
    otherMsPer1000: 0.02191,
    otherMsPer1000Ci: [0.01582, 0.02800],
    coreInterceptMs: 0.0792444002605992,
    otherInterceptMs: 0.7375029238470727,
    // B1.3's own P3 sweep on A2, 4 blocks x 64 tokens per point. These are the numbers HIT-CONTEXT
    // is measured against, and the core per layer is what CORE-FLOOR is read beside.
    tAr: [
      { positions: 192, tArMs: 0.9248046949505806, tArCi: [0.8981249928474426, 0.9612499922513962], coreMsPerLayer: 0.031067252073945936 },
      { positions: 1024, tArMs: 1.4068749994039536, tArCi: [1.3674218654632568, 1.420859381556511], coreMsPerLayer: 0.10348190399397217 },
      { positions: 2048, tArMs: 1.968945324420929, tArCi: [1.9391406327486038, 1.985624998807907], coreMsPerLayer: 0.19684772737657438 },
    ],
  },
  // The streaming surface B1.3 measured and this unit RETIRES as a pricing basis. It is carried so
  // the achieved fraction of it can be published per kernel cell, and so the retired model's
  // prediction can be published beside the kernel-curve one and the model-class change itself
  // adjudicated. Rate is the slope through two dispatch counts on cold windows, GB/s.
  streamingSurface: [
  { bytes: 524288, workgroups: 8, workgroupSize: 64, GBps: 20.1326592 },
  { bytes: 524288, workgroups: 16, workgroupSize: 64, GBps: 38.1300451790654 },
  { bytes: 524288, workgroups: 32, workgroupSize: 64, GBps: 66.2256664825177 },
  { bytes: 524288, workgroups: 64, workgroupSize: 64, GBps: 93.20593910180206 },
  { bytes: 524288, workgroups: 128, workgroupSize: 64, GBps: 125.82984000411989 },
  { bytes: 524288, workgroups: 256, workgroupSize: 64, GBps: 125.82864000183106 },
  { bytes: 524288, workgroups: 1024, workgroupSize: 64, GBps: 104.85686667179527 },
  { bytes: 524288, workgroups: 4096, workgroupSize: 64, GBps: 34.00779313384918 },
  { bytes: 1048576, workgroups: 8, workgroupSize: 64, GBps: 22.270619657001653 },
  { bytes: 1048576, workgroups: 16, workgroupSize: 64, GBps: 43.38935743162977 },
  { bytes: 1048576, workgroups: 32, workgroupSize: 64, GBps: 67.10903466710069 },
  { bytes: 1048576, workgroups: 64, workgroupSize: 64, GBps: 109.416390170641 },
  { bytes: 1048576, workgroups: 128, workgroupSize: 64, GBps: 179.75422042359097 },
  { bytes: 1048576, workgroups: 256, workgroupSize: 64, GBps: 201.32812801171883 },
  { bytes: 1048576, workgroups: 1024, workgroupSize: 64, GBps: 193.58309112441034 },
  { bytes: 1048576, workgroups: 4096, workgroupSize: 64, GBps: 61.380081380139366 },
  { bytes: 2097152, workgroups: 8, workgroupSize: 64, GBps: 24.314786977540294 },
  { bytes: 2097152, workgroups: 16, workgroupSize: 64, GBps: 46.603339917726224 },
  { bytes: 2097152, workgroups: 32, workgroupSize: 64, GBps: 78.03362278713824 },
  { bytes: 2097152, workgroups: 64, workgroupSize: 64, GBps: 124.27575601285696 },
  { bytes: 2097152, workgroups: 128, workgroupSize: 64, GBps: 218.8336877135322 },
  { bytes: 2097152, workgroups: 256, workgroupSize: 64, GBps: 305.0403614325232 },
  { bytes: 2097152, workgroups: 1024, workgroupSize: 64, GBps: 324.718871182231 },
  { bytes: 2097152, workgroups: 4096, workgroupSize: 64, GBps: 117.05052071416564 },
  { bytes: 4194304, workgroups: 8, workgroupSize: 64, GBps: 27.24310996574025 },
  { bytes: 4194304, workgroups: 16, workgroupSize: 64, GBps: 47.821086127966 },
  { bytes: 4194304, workgroups: 32, workgroupSize: 64, GBps: 83.88606000000478 },
  { bytes: 4194304, workgroups: 64, workgroupSize: 64, GBps: 145.8887218652418 },
  { bytes: 4194304, workgroups: 128, workgroupSize: 64, GBps: 232.7470535743335 },
  { bytes: 4194304, workgroups: 256, workgroupSize: 64, GBps: 359.51333878234334 },
  { bytes: 4194304, workgroups: 1024, workgroupSize: 64, GBps: 410.87158750759926 },
  { bytes: 4194304, workgroups: 4096, workgroupSize: 64, GBps: 234.10026262922207 },
  { bytes: 8388608, workgroups: 8, workgroupSize: 64, GBps: 26.8435456 },
  { bytes: 8388608, workgroups: 16, workgroupSize: 64, GBps: 52.77237592245614 },
  { bytes: 8388608, workgroups: 32, workgroupSize: 64, GBps: 91.3045800299259 },
  { bytes: 8388608, workgroups: 64, workgroupSize: 64, GBps: 153.68426026472224 },
  { bytes: 8388608, workgroups: 128, workgroupSize: 64, GBps: 238.25624660203354 },
  { bytes: 8388608, workgroups: 256, workgroupSize: 64, GBps: 432.95938286752164 },
  { bytes: 8388608, workgroups: 1024, workgroupSize: 64, GBps: 359.5115020410181 },
  { bytes: 8388608, workgroups: 4096, workgroupSize: 64, GBps: 394.7578463668615 },
  { bytes: 16777216, workgroups: 8, workgroupSize: 64, GBps: 26.794423920455785 },
  { bytes: 16777216, workgroups: 16, workgroupSize: 64, GBps: 53.12047001622115 },
  { bytes: 16777216, workgroups: 32, workgroupSize: 64, GBps: 100.91559075131471 },
  { bytes: 16777216, workgroups: 64, workgroupSize: 64, GBps: 169.53814869806777 },
  { bytes: 16777216, workgroups: 128, workgroupSize: 64, GBps: 272.0630463112021 },
  { bytes: 16777216, workgroups: 256, workgroupSize: 64, GBps: 385.31402354341907 },
  { bytes: 16777216, workgroups: 1024, workgroupSize: 64, GBps: 385.31402354341907 },
  { bytes: 16777216, workgroups: 4096, workgroupSize: 64, GBps: 434.1273060440151 },
  { bytes: 47185920, workgroups: 8, workgroupSize: 64, GBps: 27.108607787938077 },
  { bytes: 47185920, workgroups: 16, workgroupSize: 64, GBps: 52.24738270854051 },
  { bytes: 47185920, workgroups: 32, workgroupSize: 64, GBps: 99.07804459875594 },
  { bytes: 47185920, workgroups: 64, workgroupSize: 64, GBps: 188.90110118732534 },
  { bytes: 47185920, workgroups: 128, workgroupSize: 64, GBps: 315.44920154258364 },
  { bytes: 47185920, workgroups: 256, workgroupSize: 64, GBps: 471.85890000019066 },
  { bytes: 47185920, workgroups: 1024, workgroupSize: 64, GBps: 471.85927500001196 },
  { bytes: 47185920, workgroups: 4096, workgroupSize: 64, GBps: 468.92859010076967 },
  { bytes: 524288, workgroups: 8, workgroupSize: 128, GBps: 35.95133387823434 },
  { bytes: 524288, workgroups: 16, workgroupSize: 128, GBps: 62.914920002059944 },
  { bytes: 524288, workgroups: 32, workgroupSize: 128, GBps: 96.79083550949267 },
  { bytes: 524288, workgroups: 64, workgroupSize: 128, GBps: 119.83777959411442 },
  { bytes: 524288, workgroups: 128, workgroupSize: 128, GBps: 114.38947438368706 },
  { bytes: 524288, workgroups: 256, workgroupSize: 128, GBps: 125.82864000183106 },
  { bytes: 524288, workgroups: 1024, workgroupSize: 128, GBps: 59.91888979705721 },
  { bytes: 524288, workgroups: 4096, workgroupSize: 128, GBps: 17.722386510647556 },
  { bytes: 1048576, workgroups: 8, workgroupSize: 128, GBps: 41.25553689886966 },
  { bytes: 1048576, workgroups: 16, workgroupSize: 128, GBps: 62.91447000012874 },
  { bytes: 1048576, workgroups: 32, workgroupSize: 128, GBps: 125.82864000183106 },
  { bytes: 1048576, workgroups: 64, workgroupSize: 128, GBps: 152.5201807162616 },
  { bytes: 1048576, workgroups: 128, workgroupSize: 128, GBps: 218.8336877135322 },
  { bytes: 1048576, workgroups: 256, workgroupSize: 128, GBps: 228.7809322336336 },
  { bytes: 1048576, workgroups: 1024, workgroupSize: 128, GBps: 122.75959167387268 },
  { bytes: 1048576, workgroups: 4096, workgroupSize: 128, GBps: 28.43595196782724 },
  { bytes: 2097152, workgroups: 8, workgroupSize: 128, GBps: 43.20316811143509 },
  { bytes: 2097152, workgroups: 16, workgroupSize: 128, GBps: 79.26240922562876 },
  { bytes: 2097152, workgroups: 32, workgroupSize: 128, GBps: 124.27575601285696 },
  { bytes: 2097152, workgroups: 64, workgroupSize: 128, GBps: 223.69583407471708 },
  { bytes: 2097152, workgroups: 128, workgroupSize: 128, GBps: 272.0623450707934 },
  { bytes: 2097152, workgroups: 256, workgroupSize: 128, GBps: 305.0421245289283 },
  { bytes: 2097152, workgroups: 1024, workgroupSize: 128, GBps: 239.6733823182714 },
  { bytes: 2097152, workgroups: 4096, workgroupSize: 128, GBps: 55.30957053504128 },
  { bytes: 4194304, workgroups: 8, workgroupSize: 128, GBps: 49.89505626511715 },
  { bytes: 4194304, workgroups: 16, workgroupSize: 128, GBps: 81.84007677969464 },
  { bytes: 4194304, workgroups: 32, workgroupSize: 128, GBps: 124.66037253690364 },
  { bytes: 4194304, workgroups: 64, workgroupSize: 128, GBps: 224.94579779663198 },
  { bytes: 4194304, workgroups: 128, workgroupSize: 128, GBps: 359.5115020410181 },
  { bytes: 4194304, workgroups: 256, workgroupSize: 128, GBps: 291.7766371807165 },
  { bytes: 4194304, workgroups: 1024, workgroupSize: 128, GBps: 379.8610022078288 },
  { bytes: 4194304, workgroups: 4096, workgroupSize: 128, GBps: 112.47289889831599 },
  { bytes: 8388608, workgroups: 8, workgroupSize: 128, GBps: 53.12047335788528 },
  { bytes: 8388608, workgroups: 16, workgroupSize: 128, GBps: 91.51205553720108 },
  { bytes: 8388608, workgroups: 32, workgroupSize: 128, GBps: 150.24381519274695 },
  { bytes: 8388608, workgroups: 64, workgroupSize: 128, GBps: 232.74731018092453 },
  { bytes: 8388608, workgroups: 128, workgroupSize: 128, GBps: 317.05011306383915 },
  { bytes: 8388608, workgroups: 256, workgroupSize: 128, GBps: 383.4795014968011 },
  { bytes: 8388608, workgroups: 1024, workgroupSize: 128, GBps: 394.7578463668615 },
  { bytes: 8388608, workgroups: 4096, workgroupSize: 128, GBps: 226.20956667090854 },
  { bytes: 16777216, workgroups: 8, workgroupSize: 128, GBps: 52.96325692410904 },
  { bytes: 16777216, workgroups: 16, workgroupSize: 128, GBps: 102.19625047798306 },
  { bytes: 16777216, workgroups: 32, workgroupSize: 128, GBps: 167.5975352798806 },
  { bytes: 16777216, workgroups: 64, workgroupSize: 128, GBps: 278.171504645323 },
  { bytes: 16777216, workgroups: 128, workgroupSize: 128, GBps: 292.30689591702105 },
  { bytes: 16777216, workgroups: 256, workgroupSize: 128, GBps: 381.6613303749734 },
  { bytes: 16777216, workgroups: 1024, workgroupSize: 128, GBps: 391.87615618707406 },
  { bytes: 16777216, workgroups: 4096, workgroupSize: 128, GBps: 384.3942237559761 },
  { bytes: 47185920, workgroups: 8, workgroupSize: 128, GBps: 53.16723094624098 },
  { bytes: 47185920, workgroups: 16, workgroupSize: 128, GBps: 100.35110243815969 },
  { bytes: 47185920, workgroups: 32, workgroupSize: 128, GBps: 187.33863783411022 },
  { bytes: 47185920, workgroups: 64, workgroupSize: 128, GBps: 315.01037449247804 },
  { bytes: 47185920, workgroups: 128, workgroupSize: 128, GBps: 537.9868533353147 },
  { bytes: 47185920, workgroups: 256, workgroupSize: 128, GBps: 472.84457846696864 },
  { bytes: 47185920, workgroups: 1024, workgroupSize: 128, GBps: 469.9013077598545 },
  { bytes: 47185920, workgroups: 4096, workgroupSize: 128, GBps: 480.87571130676486 },
  { bytes: 524288, workgroups: 8, workgroupSize: 256, GBps: 66.22599889228978 },
  { bytes: 524288, workgroups: 16, workgroupSize: 256, GBps: 119.8366911591357 },
  { bytes: 524288, workgroups: 32, workgroupSize: 256, GBps: 104.85936669643239 },
  { bytes: 524288, workgroups: 64, workgroupSize: 256, GBps: 251.66208005859465 },
  { bytes: 524288, workgroups: 128, workgroupSize: 256, GBps: 125.82864000183106 },
  { bytes: 524288, workgroups: 256, workgroupSize: 256, GBps: 114.38947438368706 },
  { bytes: 524288, workgroups: 1024, workgroupSize: 256, GBps: 27.961979259339635 },
  { bytes: 524288, workgroups: 4096, workgroupSize: 256, GBps: 8.530785590348172 },
  { bytes: 1048576, workgroups: 8, workgroupSize: 256, GBps: 62.13787800642848 },
  { bytes: 1048576, workgroups: 16, workgroupSize: 256, GBps: 107.08833463177714 },
  { bytes: 1048576, workgroups: 32, workgroupSize: 256, GBps: 152.5201807162616 },
  { bytes: 1048576, workgroups: 64, workgroupSize: 256, GBps: 193.58309112441034 },
  { bytes: 1048576, workgroups: 128, workgroupSize: 256, GBps: 209.71373334359055 },
  { bytes: 1048576, workgroups: 256, workgroupSize: 256, GBps: 186.4131950622642 },
  { bytes: 1048576, workgroups: 1024, workgroupSize: 256, GBps: 58.52513055708735 },
  { bytes: 1048576, workgroups: 4096, workgroupSize: 256, GBps: 17.722410315413658 },
  { bytes: 2097152, workgroups: 8, workgroupSize: 256, GBps: 83.19300069712651 },
  { bytes: 2097152, workgroups: 16, workgroupSize: 256, GBps: 127.42174600241435 },
  { bytes: 2097152, workgroups: 32, workgroupSize: 256, GBps: 223.69583407471708 },
  { bytes: 2097152, workgroups: 64, workgroupSize: 256, GBps: 314.5749750150384 },
  { bytes: 2097152, workgroups: 128, workgroupSize: 256, GBps: 296.0683847751461 },
  { bytes: 2097152, workgroups: 256, workgroupSize: 256, GBps: 159.78322237367738 },
  { bytes: 2097152, workgroups: 1024, workgroupSize: 256, GBps: 114.38972231535736 },
  { bytes: 2097152, workgroups: 4096, workgroupSize: 256, GBps: 34.59219975201058 },
  { bytes: 4194304, workgroups: 8, workgroupSize: 256, GBps: 83.36504976402661 },
  { bytes: 4194304, workgroups: 16, workgroupSize: 256, GBps: 125.04752526524693 },
  { bytes: 4194304, workgroups: 32, workgroupSize: 256, GBps: 222.46038829097506 },
  { bytes: 4194304, workgroups: 64, workgroupSize: 256, GBps: 432.9584949096413 },
  { bytes: 4194304, workgroups: 128, workgroupSize: 256, GBps: 419.43080000038145 },
  { bytes: 4194304, workgroups: 256, workgroupSize: 256, GBps: 497.1041946089971 },
  { bytes: 4194304, workgroups: 1024, workgroupSize: 256, GBps: 228.78043636384444 },
  { bytes: 4194304, workgroups: 4096, workgroupSize: 256, GBps: 65.47210711087719 },
  { bytes: 8388608, workgroups: 8, workgroupSize: 256, GBps: 91.09804316866493 },
  { bytes: 8388608, workgroups: 16, workgroupSize: 256, GBps: 147.49203352252556 },
  { bytes: 8388608, workgroups: 32, workgroupSize: 256, GBps: 266.6573605022661 },
  { bytes: 8388608, workgroups: 64, workgroupSize: 256, GBps: 415.1066691042573 },
  { bytes: 8388608, workgroups: 128, workgroupSize: 256, GBps: 362.7504167519967 },
  { bytes: 8388608, workgroups: 256, workgroupSize: 256, GBps: 398.6679342075543 },
  { bytes: 8388608, workgroups: 1024, workgroupSize: 256, GBps: 462.8193052719329 },
  { bytes: 8388608, workgroups: 4096, workgroupSize: 256, GBps: 136.95684042760413 },
  { bytes: 16777216, workgroups: 8, workgroupSize: 256, GBps: 102.4562853912944 },
  { bytes: 16777216, workgroups: 16, workgroupSize: 256, GBps: 170.43526519864042 },
  { bytes: 16777216, workgroups: 32, workgroupSize: 256, GBps: 284.56036994855145 },
  { bytes: 16777216, workgroups: 64, workgroupSize: 256, GBps: 458.86349873195053 },
  { bytes: 16777216, workgroups: 128, workgroupSize: 256, GBps: 451.1515833820239 },
  { bytes: 16777216, workgroups: 256, workgroupSize: 256, GBps: 420.5253174922052 },
  { bytes: 16777216, workgroups: 1024, workgroupSize: 256, GBps: 485.1243626651224 },
  { bytes: 16777216, workgroups: 4096, workgroupSize: 256, GBps: 272.9851388911415 },
  { bytes: 47185920, workgroups: 8, workgroupSize: 256, GBps: 101.24829093182129 },
  { bytes: 47185920, workgroups: 16, workgroupSize: 256, GBps: 188.27301661059516 },
  { bytes: 47185920, workgroups: 32, workgroupSize: 256, GBps: 317.6611249230453 },
  { bytes: 47185920, workgroups: 64, workgroupSize: 256, GBps: 467.9598912645259 },
  { bytes: 47185920, workgroups: 128, workgroupSize: 256, GBps: 461.2882098216764 },
  { bytes: 47185920, workgroups: 256, workgroupSize: 256, GBps: 477.8321243390493 },
  { bytes: 47185920, workgroups: 1024, workgroupSize: 256, GBps: 475.82457114616466 },
  { bytes: 47185920, workgroups: 4096, workgroupSize: 256, GBps: 481.89887478499907 },
  ],
} as const;

// ---- P0a: the kernel bandwidth curves ----
//
// B1.3 priced every dispatch off a synthetic streaming kernel's surface and missed every lever:
// attention +242%, conv +88%, MLP +109%, total +32.4%. Its own per-kind table showed why -- the
// real kernels read at 0.1 to 0.8 of the streaming rate, monotone in occupancy -- so the streaming
// surface is retired here and the pricing basis is the kernel's own curve, measured on the kernel
// classes the runtime actually dispatches, at the shapes it actually reads.
//
// The chain half is carried unchanged: N dispatches in one command buffer, each reading the
// previous one's output, so the slope of time against N is lambda_chain.
//
// Every cell is measured at two dispatch counts and the rate taken from the SLOPE between them,
// which cancels the per-cell launch cost exactly. Every dispatch reads its own window of a
// 754 974 720 B footprint through a dynamic offset and the windows advance across warmups, reps and
// rounds, so no read is ever served from a line another dispatch of the same cell just brought in.
export const P0A = {
  chainCounts: [8, 16, 24, 32, 48, 64, 96, 128],
  rounds: 8,
  repsPerRound: 25,
  warmupReps: 20,
  workgroupSize: 64,
  // The six real dispatch shapes, one per (class, matrix) the plan contains. `weightBytes` is
  // quantised weights + f32 scales + packed zero points; `dispatchBytes` adds the gamma row, the
  // input vector and the output the kernel actually writes, which is the quantity the cost model
  // divides by BW_kernel. Every one is a multiple of 256 in every part, so a dynamic-offset window
  // is legal for each binding independently.
  shapes: [
    { key: "attn_o_proj", mode: "matvec_residual", rows: 1024, cols: 1024, emits: 1024, zeroPoints: false, preScale: 1, weightBytes: 655_360, dispatchBytes: 663_552 },
    { key: "mlp_down", mode: "matvec_residual", rows: 1024, cols: 2560, emits: 1024, zeroPoints: false, preScale: 1, weightBytes: 1_638_400, dispatchBytes: 1_651_712 },
    { key: "attn_qkv", mode: "norm_projection", rows: 2048, cols: 1024, emits: 2048, zeroPoints: false, preScale: 1, weightBytes: 1_310_720, dispatchBytes: 1_327_104 },
    { key: "conv_in_proj", mode: "norm_matvec", rows: 3072, cols: 1024, emits: 3072, zeroPoints: false, preScale: 1, weightBytes: 1_966_080, dispatchBytes: 1_980_416 },
    { key: "mlp_gate_up", mode: "norm_swiglu", rows: 5120, cols: 1024, emits: 2560, zeroPoints: false, preScale: 1, weightBytes: 3_276_800, dispatchBytes: 3_290_112 },
    { key: "head", mode: "norm_head", rows: 65536, cols: 1024, emits: 65536, zeroPoints: true, preScale: RUNTIME.headPreScale, weightBytes: 42_991_616, dispatchBytes: 43_257_856 },
  ],
  // REGISTERED DEVIATION on the occupancy axis. The spec asks for workgroup counts
  // {64 ... 4096} x sizes {64, 128, 256}. Without split-K -- which N1 forbids and no arm here uses
  // -- a matvec dispatch has exactly ceil(emits / rowsPerWorkgroup) workgroups, so the workgroup
  // count is not a free axis: at 1024 output rows it cannot exceed 1024 and 2048 and 4096 are
  // unreachable. The sweep is therefore parameterised by rows per workgroup, which IS free, and
  // the achieved workgroup count is published per cell. The two geometries B1.3 ran are 16 rows
  // per workgroup (its `base`) and 4 (its `wide`), both inside the sweep, so the baseline arm's
  // every dispatch is priced from a cell that was measured rather than interpolated.
  rowsPerWorkgroup: [1, 2, 4, 8, 16, 32],
  // The head emits 65 536 rows; one row per workgroup would need 65 536 workgroups against a
  // 4096 limit, so its sweep starts at 16 rows per workgroup (4096 workgroups) and halves down.
  rowsPerWorkgroupHead: [16, 32, 64, 128, 256, 512, 1024],
  workgroupSizes: [64, 128, 256],
  // REACHABILITY. A workgroup holds workgroupSize/subgroupSize subgroups. A cell asking for fewer
  // rows per workgroup than it has subgroups needs several subgroups to cooperate on one row,
  // which the generalised kernel does -- it partitions the row's quantisation blocks across
  // subgroupsPerRow subgroups and closes with an index-ordered reduction in workgroup memory. That
  // partition is only a real occupancy point when every subgroup gets at least one block, i.e.
  // subgroupsPerRow * subgroupSize <= cols / quantBlock. A cell failing this leaves subgroups
  // wholly idle and is SKIPPED, not measured at zero; the skipped set is published.
  reachabilityRule: "subgroupsPerRow * subgroupSize <= cols / quantBlock",
  // Two dispatch counts per cell: the rate is (bytes_hi - bytes_lo) / (t_hi - t_lo).
  dispatchCounts: [4, 16],
  footprintBytes: 754_974_720,
  fillChunkBytes: 67_108_864,
  curveRounds: 6,
  curveTargetBytesPerRound: 300 * 1024 * 1024,
  curveMinReps: 3,
  curveMaxReps: 25,
  curveWarmupReps: 8,
  // The occupancy knee at a shape: the smallest workgroup count reaching this fraction of the best
  // rate measured for that shape across every reachable cell. N1 sets each dispatch's geometry
  // from its shape's knee.
  kneeFraction: 0.9,
  seed: 111_007,
} as const;

// ---- P0b: the dequantisation variants ----
//
// Run only where P0a leaves a class below the KERNEL-STOP rate at the 1-3 MB sizes. The existing
// kernel already unpacks eight nibbles from one u32 and closes each row with a subgroupAdd, so the
// spec's two lines are read as the two things it does NOT do:
//
//   V1  vec4<u32> loads and f16 unpack/dot math -- one 16 B load per block instead of four 4 B
//       loads, and the eight-term dot in f16 instead of f32.
//   V2  four independent f32 accumulator lanes instead of one, so the inner loop's FMA chain is
//       four ways parallel, closed by a single subgroupAdd on the vec4.
//   V3  both.
//
// Best variant at best occupancy is measured, not argued. Adoption is kernel-internal and applies
// to EVERY arm including the baseline, so N1 and N2 are priced on the same matvec.
export const P0B = {
  variants: ["v1_vec4_f16", "v2_vec4_accumulator", "v3_both"],
  // The band the gate reads. The spec's "1-3 MB" sizes are this unit's 1.31, 1.64, 1.97 and
  // 3.28 MB shapes; 3.28 MB is the spec's "~3.0 MB" entry, so the upper edge admits it.
  bandBytes: [1_048_576, 3_500_000],
  // A variant is ADOPTED when it is the best by mean rate across the band's shapes AND its worst
  // shape in the band clears this. Same threshold as KERNEL-STOP, because they are the same claim.
  adoptionGBps: 300,
  rounds: 6,
  // Correctness of an adopted variant is not optional: its rows are compared against the carried
  // kernel's on the real weights before any arm runs. All five classes are checked, not only the
  // ones inside the band: an adoption replaces the matvec in EVERY dispatch of every arm, so a
  // variant that is exact on the 3 MB MLP and wrong on the 43 MB head is not adoptable either.
  // The section named for each class is the first real matrix of that shape in the blob -- layer 0
  // is a conv layer and layer 2 is the first attention layer, so those are where the two
  // projections live.
  checkSections: [
    { key: "attn_o_proj", section: "L2.attn.o_proj" },
    { key: "mlp_down", section: "L0.mlp.down" },
    { key: "attn_qkv", section: "L2.attn.qkv" },
    { key: "conv_in_proj", section: "L0.conv.in_proj" },
    { key: "mlp_gate_up", section: "L0.mlp.gate_up" },
    { key: "head", section: "head" },
  ],
  // The occupancy the check runs at. It is B1.3's carried `base` geometry rather than a swept cell:
  // the check measures the inner product, which is what the variant changes, and running it at one
  // fixed geometry keeps the comparison between variants a comparison of arithmetic alone.
  checkRowsPerWorkgroup: 16,
  checkWorkgroupSize: 128,
  // The staged input the checked matrices are run against, drawn once and shared by the carried
  // kernel and every variant.
  checkSeed: 111_009,
  variantTolerance: 0.005,
} as const;

// ---- P0c: the flash core microbench ----
//
// The standalone split-position core at three depths and seven block counts. A block stages K and V
// one 64-position tile at a time and keeps a running (maximum, denominator, accumulator), so S = 1
// is a legal cell at every depth and the workgroup allocation does not move with context length.
//
// Six cache slices, one per attention layer, cycled across the repetitions: a real token reads six
// different KV caches and then reads them all again on the next token, so a single re-read slice
// would measure a warmer cache than the token has and six is the token's own working set.
export const P0C = {
  positions: [192, 1024, 2048],
  blockCounts: [1, 2, 3, 4, 8, 16, 32],
  cacheSlices: 6,
  dispatchCounts: [4, 16],
  rounds: 6,
  repsPerRound: 25,
  warmupReps: 8,
  // The registered per-layer core predictions, written before the kernel exists.
  predictedCoreMsPerLayer: { 192: 0.010, 1024: 0.030, 2048: 0.055 },
  // The rule that picks S for an arm at a context length, registered before P0c runs. `positions`
  // means the attended count, position + 1, and it is evaluated at the LAST position of a run so
  // one encoded command buffer serves the whole run: S is fixed for a run and the block length
  // grows with the position inside it.
  blocksRule: "min(maxBlocks, max(1, ceil(positions / positionsPerBlockRule)))",
  seed: 111_008,
} as const;

// ---- P0d: the registered prediction ----
//
// Computed by the agent from P0a, P0b and P0c on the planned arms and written before any lever
// kernel. Claude's desk prior is recorded BESIDE it, not instead of it; the P0d number is the one
// PRICING adjudicates.
export const P0D = {
  deskPriorTAr192Ms: [0.60, 0.66],
  deskPriorTAr1024Ms: [0.75, 0.85],
  // The retired model is published beside the adopted one at the same two depths, so the change of
  // model class is itself adjudicated rather than assumed.
  publishRetiredStreamingPrediction: true,
} as const;

// The cost model, with the kernel curve in place of the streaming surface:
//
//   t_AR ~= sum over dispatches of max(lambda_kind, bytes_i / BW_kernel(class_i, bytes_i, wg_i, size_i))
//
// Every matvec dispatch is priced from the cell P0a measured for its own shape at its own
// occupancy -- an exact table lookup, no interpolation, because the plan's shapes ARE the measured
// shapes. Every other kind is priced from B1.3's measurement of that same kernel in the real token,
// and the flash core from P0c. Nothing is priced from a synthetic streaming kernel.
export const DESK = {
  lambdaKinds: ["chain", "attn_core", "conv_core", "sample"],
  // Which kinds are priced from B1_3.perDispatchMs rather than from a curve measured here.
  carriedKinds: ["mlp_fused", "conv_core", "fold", "embed", "sample_partial", "sample_final"],
  // PRICING names one lever. Two are named jointly only when the runner-up's error is within this
  // fraction of the leader's, i.e. when the evidence does not separate them.
  pricingMissTieFraction: 0.1,
} as const;

export const TARGET = {
  tokensPerSecond: 1400,
  tArMs: 0.714,
  // HIT-CONTEXT: a chat product lives at 500-2000 context, and 508 tok/s at 2048 is the number
  // users would actually see. Co-primary with HIT-1400.
  contextPositions: 1024,
  contextTArMs: 1.000,
  contextTokensPerSecond: 1000,
} as const;

export const GATES = {
  // HIT-1400: t_AR <= 0.714 ms with the CI upper bound <= 0.735, correctness passed, 192 positions.
  hitTArMs: 0.714,
  hitCiUpperMs: 0.735,
  // HIT-CONTEXT: t_AR <= 1.000 ms with the CI upper bound <= 1.030 at 1024 positions, correctness
  // battery rerun and passed at that depth. Fires independently of HIT-1400.
  contextPositions: 1024,
  contextTArMs: 1.000,
  contextCiUpperMs: 1.030,
  // NEAR-1400: 0.714 < t_AR(192) <= 0.80 -> name the residual kind.
  nearTArMs: 0.80,
  // CORE-FLOOR, a lever-level record and not a unit kill: the winner's core per layer above these
  // with S at its P0c optimum means the merge fold or the staging is the cap.
  coreFloorMsPerLayer192: 0.015,
  coreFloorMsPerLayer1024: 0.045,
  // PRICING: per lever +/-30%, total +/-20%. Inside the total band the kernel-curve model is
  // ADOPTED; outside it, the term that broke it is named.
  pricingLeverTolerance: 0.30,
  pricingTotalTolerance: 0.20,
  // KERNEL-STOP, pricing-first: if P0a and P0b leave EVERY matvec class below this at the 1-3 MB
  // sizes at EVERY reachable occupancy, the token is dequant/instruction-bound; the unit stops
  // before any lever kernel is built and names the dequant unit.
  kernelStopGBps: 300,
  // P0-GO: the P0d predicted t_AR at 192 positions must be at or under this for the arms to be
  // built. Above it, the priced shortfall is named and the unit stops.
  p0GoTArMs: 0.75,
  // IMPL-GAP: idle above this, or pipeline churn -> not a verdict.
  implGapIdle: 0.05,
  // SPLIT: the two GPU-time instruments disagreeing by more than this fraction of t_AR.
  splitTolerance: 0.15,
} as const;

export const CODES = {
  gate: {
    NONE: 0,
    HIT_1400: 1,
    NEAR_1400: 2,
    MISS: 3,
    IMPL_GAP: 4,
    SPLIT_NO_VERDICT: 5,
    NOT_ADJUDICABLE: 6,
    KERNEL_STOP: 7,
    P0_NO_GO: 8,
  },
  rung: {
    NONE: 0,
    L1_DROP_P0B: 1,
    L2_CONTEXT_SHORT: 2,
    L3_DROP_N2_ALONE: 3,
    L4_EIGHT_BLOCKS: 4,
  },
  arm: { A2: 0, A2N1: 1, A2N2: 2, A2N1N2: 3 },
  // The two levers, as flags on an arm and as names in the per-lever table.
  lever: { NONE: 0, SPLIT_N: 1, FLASH: 2, BOTH: 3, TOTAL: 4, MULTIPLE: 5 },
  // Which group a dispatch belongs to, unchanged from B1.3 so the two per-group tables read
  // side by side.
  group: { ATTENTION: 1, MLP: 2, CONV: 3, OTHER: 4 },
  pricingVerdict: { NONE: 0, ADOPTED: 1, MISSED: 2 },
  // Where a priced dispatch's rate came from. Every basis is a measurement: P0a's cell for that
  // shape at that occupancy, P0c's cell for an attention core, or B1.3's measurement of the same
  // kernel inside the real token. RETIRED_SURFACE is never an adopted basis; it labels the
  // comparison prediction P0d publishes beside the adopted one.
  priceBasis: { NONE: 0, KERNEL_CURVE: 1, CORE_MICROBENCH: 2, CARRIED: 3, RETIRED_SURFACE: 4 },
  instrument: { UNAVAILABLE: 0, MEASURED: 1, ESTIMATED: 2 },
  verdict: { FALSIFIED: 0, CONFIRMED: 1, UNMEASURED: -1 },
  divergingInstrument: { NONE: 0, PASS_TIMESTAMP: 1, PER_DISPATCH_SUM: 2 },
  implGapComponent: {
    NONE: 0,
    HOST_GAP: 1,
    UNFUSED_ATTENTION: 2,
    UNFUSED_MLP: 3,
    UNFUSED_CONV: 4,
    UNFOLDED_HOUSEKEEPING: 5,
    READBACK: 6,
    PIPELINE_CHURN: 7,
  },
  // Dispatch kinds. Codes 1 to 17 are B1.2's and B1.3's, unchanged so the three units' per-kind
  // tables can be read side by side; the holes left by kinds no arm here emits are kept rather
  // than renumbered. 18 and 19 are this unit's.
  kind: {
    NONE: 0,
    EMBED: 1,
    NORM_MATVEC: 2,
    NORM_MATVEC_SWIGLU: 3,
    MATVEC_RESIDUAL: 4,
    NORM_HEAD: 5,
    CONV_CORE: 8,
    SAMPLE_PARTIAL: 9,
    SAMPLE_FINAL: 10,
    MLP_FUSED: 12,
    FOLD: 14,
    ATTN_PROJ: 16,
    ATTN_CORE_QKV: 17,
    ATTN_CORE_FLASH: 18,
    ATTN_MERGE: 19,
  },
  // The four registered lambda kinds. A dispatch kind and a lambda kind are different enumerations
  // -- twelve kernels pay four floors -- so the lambda table gets its own codes.
  lambdaKind: { CHAIN: 1, ATTN_CORE: 2, CONV_CORE: 3, SAMPLE: 4 },
  // The five matvec kernel classes P0a measures a curve for, and the one shape each is measured at.
  matvecClass: {
    NONE: 0,
    MATVEC_RESIDUAL: 1,
    NORM_MATVEC: 2,
    NORM_PROJECTION: 3,
    NORM_SWIGLU: 4,
    NORM_HEAD: 5,
  },
  matvecShape: {
    NONE: 0,
    ATTN_O_PROJ: 1,
    MLP_DOWN: 2,
    ATTN_QKV: 3,
    CONV_IN_PROJ: 4,
    MLP_GATE_UP: 5,
    HEAD: 6,
  },
  // The dequantisation variants, and what P0b adopted. CARRIED is B1.3's kernel unchanged.
  dequantVariant: { CARRIED: 0, V1_VEC4_F16: 1, V2_VEC4_ACCUMULATOR: 2, V3_BOTH: 3 },
  // Which half of an attention layer a dispatch is, so the core and the projections are priced
  // apart in every arm. MERGE is the flash core's log-sum-exp fold, charged to the core.
  attnPart: { NONE: 0, PROJECTION: 1, CORE: 2, OUT_PROJECTION: 3, WHOLE: 4, MERGE: 5 },
  // The registered staging choice for K and V inside the attention core. NONE is what
  // `escalation_lever` and `next_lever` publish when no lever is named.
  staging: { NONE: 0, STREAM_GLOBAL: 1, STAGE_K_ONLY: 2, SPLIT_POSITIONS_FOLD: 3, TILE_STAGED_SPLIT_POSITIONS: 4 },
  // What the next unit's lever would be, named here and not adjudicated here.
  nextLever: {
    NONE: 0,
    DEQUANT_PACKED_F16_SUBGROUP: 1,
    SPLIT_POSITIONS_FOLD: 2,
    PAGED_KV_CACHE: 3,
    QUANTISED_KV_CACHE: 4,
    PERSISTENT_KERNEL: 5,
  },
  defect: {
    PIPELINE_CHURN_PER_STEP: 1,
    LIVE_GPU_BYTES_GROWTH: 2,
    THERMAL_DRIFT_PERSISTS: 3,
    CORRECTNESS_BELOW_GATE: 4,
    DISPATCH_COUNT_OFF_PLAN: 5,
    IDLE_INSTRUMENT_DISAGREEMENT: 6,
    DETERMINISM_HASH_MISMATCH: 7,
    DEFERRED_SUM_MISMATCH: 8,
    DEQUANT_MISMATCH: 9,
    LEVER_DID_NOT_APPLY: 10,
    ROPE_PREFIX_MISMATCH: 11,
    FLASH_CORE_MISMATCH: 12,
    REPACK_CONTENT_MISMATCH: 13,
    SPLIT_N_MISMATCH: 14,
    DEQUANT_VARIANT_MISMATCH: 15,
    P0A_CELL_UNREACHABLE: 16,
  },
  // A stage ledger entry. STOPPED is not TRIMMED and not FAILED: a rung gives a stage up for
  // time and a failure is an error, but a stage marked STOPPED was cancelled by a gate that
  // fired before it -- the registered consequence of KERNEL-STOP or a failed P0-GO is that no
  // lever kernel is written at all, so the stages that would have built and timed them never
  // become runnable and the ledger says which gate cancelled them.
  stage: { PENDING: 0, DONE: 1, FAILED: 2, TRIMMED: 3, STOPPED: 4 },
  ran: { NOT_RUN: 0, RUN: 1 },
} as const;

export const SENTINEL = -1.0;

// The context sweep. HIT-CONTEXT is adjudicated here, on the winner AND on A2 in the same session,
// and the core slope is re-fitted for both arms. The cache is primed by free-running the runtime to
// the position, never by writing a position into the state, so the cache holds the keys and values
// the model would actually have produced.
export const CONTEXT = {
  positions: [192, 1024, 2048],
  positionsAfterL2: [192, 1024],
  blocks: 4,
  tokensPerBlock: 64,
  warmupSteps: 20,
  // The gate's depth. 2048 is record only.
  gatePosition: 1024,
  // Above this at 1024 on the winner, the next unit's lever is named here, not adjudicated here.
  escalationTArMs1024: 1.0,
} as const;

// The blob is built BEFORE P0 because P0b's variant check reads real matrices out of it: a
// dequantisation variant trades the precision of an eight-term dot, and the size of that error
// depends on the magnitudes the real weights and scales carry, so a synthetic matrix would check
// nothing. Nothing in the blob depends on P0 -- the layout, the repack set and the section offsets
// are all geometry-independent -- so the swap costs the run no information.
export const STAGES = [
  "e1",
  "weights",
  "p0",
  "preflight",
  "correctness",
  "m",
  "context",
  "finalize",
  "closure",
] as const;
export type StageName = (typeof STAGES)[number];

// The stages that exist only to build, check and time the lever kernels. P0 decides whether
// they are ever runnable: if either gatekeeper fires, the unit stops at P0 with the next
// unit's lever named, these four are marked STOPPED, and the run goes straight to its
// deliverables. Registered here rather than derived from the position of `p0` in STAGES, so
// the set is a fact about what a lever arm needs and not an accident of stage order.
export const LEVER_STAGES: readonly StageName[] = ["preflight", "correctness", "m", "context"];

// The trim ladder, in registered order:
//   L1 drop the P0b dequant variants unless KERNEL-STOP would otherwise fire;
//   L2 context sweep at 192 and 1024 only, 2048 recorded NOT_MEASURED;
//   L3 drop the A2N2-alone arm, N2 still priced by difference from A2N1 and A2N1N2;
//   L4 16 -> 8 timing blocks.
// Each threshold is the remaining time at which the run can no longer afford what that rung drops.
export const DEADLINE = {
  totalSeconds: 12 * 3600,
  l1MinRemainingBeforeP0b: 10 * 3600,
  l2MinRemainingBeforeContext: 70 * 60,
  l3MinRemainingBeforeThirdArm: 3 * 3600,
  l4MinRemainingBeforeM: 4 * 3600,
  stageBudgetSeconds: {
    e1: 5 * 60,
    p0: 95 * 60,
    weights: 25 * 60,
    preflight: 30 * 60,
    correctness: 90 * 60,
    m: 230 * 60,
    context: 60 * 60,
    finalize: 10 * 60,
    closure: 5 * 60,
  },
} as const;
