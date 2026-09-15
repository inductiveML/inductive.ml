// Types shared by the injected hooks script, the page bundle and the node driver.
import type {
  AdapterRecord,
  CorrectnessConfig,
  CorrectnessResult,
  DeviceDescriptorRecord,
  GenerateConfig,
  GenerateResult,
  HookSnapshot,
  MConfig,
  MPageResult,
  P0aConfig,
  P0aPageResult,
  P0bConfig,
  P0bPageResult,
  P0cConfig,
  P0cPageResult,
  ContextConfig,
  ContextPageResult,
  PageEnv,
  ReferenceLoadConfig,
  ReferenceLoadResult,
  RuntimeConfig,
  RuntimeReady,
  SmokeConfig,
  SmokeResult,
  TimestampPass,
  TokenizeConfig,
  TokenizeResult,
} from "../schemas.ts";

export interface ProbeHooks {
  readonly installed: boolean;
  configure(extraFeatures: readonly string[]): void;
  beginStep(): void;
  endStep(): HookSnapshot;
  snapshot(): HookSnapshot;
  setInjection(commandBuffer: GPUCommandBuffer | null): void;
  enableTimestamps(device: GPUDevice, capacityPairs: number): void;
  disableTimestamps(): void;
  collectTimestamps(): Promise<TimestampPass[]>;
  devices(): readonly GPUDevice[];
  descriptors(): readonly DeviceDescriptorRecord[];
  errors(): readonly string[];
  lost(): readonly string[];
  liveBytes(): number;
  suspend<T>(fn: () => T): T;
  suspendAsync<T>(fn: () => Promise<T>): Promise<T>;
}

export interface ProbeApi {
  readonly ready: boolean;
  env(): PageEnv;
  adapterRecord(powerPreference: "high-performance" | "low-power"): Promise<AdapterRecord>;
  runtimeInit(config: RuntimeConfig): Promise<RuntimeReady>;
  runtimeSmoke(config: SmokeConfig): Promise<SmokeResult>;
  runtimeDispose(): Promise<void>;
  referenceLoad(config: ReferenceLoadConfig): Promise<ReferenceLoadResult>;
  referenceTokenize(config: TokenizeConfig): Promise<TokenizeResult>;
  referenceGenerate(config: GenerateConfig): Promise<GenerateResult>;
  referenceDispose(): Promise<void>;
  correctness(config: CorrectnessConfig): Promise<CorrectnessResult>;
  measure(config: MConfig): Promise<MPageResult>;
  p0a(config: P0aConfig): Promise<P0aPageResult>;
  p0b(config: P0bConfig): Promise<P0bPageResult>;
  p0c(config: P0cConfig): Promise<P0cPageResult>;
  context(config: ContextConfig): Promise<ContextPageResult>;
}

declare global {
  // Installed by hooks.ts before any page script runs.
  var __probeHooks: ProbeHooks;
  // Installed by main.ts.
  var probe: ProbeApi;
  // Installed by the node driver through exposeFunction.
  function probeEmit(kind: string, payload: string): Promise<void>;
}
