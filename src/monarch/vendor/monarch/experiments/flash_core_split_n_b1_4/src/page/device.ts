// One GPUDevice for the whole page: the fused runtime, the reference arm and every instrument
// share it, so B1.0's reference numbers and R1's numbers come from the same device object in the
// same session. Ownership runs from the reference arm outwards: ORT creates the device exactly as
// it does in B1.0 (its WebGPU EP cannot run a session on a device registered from outside) and
// the runtime adopts whatever device the session already holds, asserting every feature and limit
// its kernels need. When no reference arm is loaded — the depth sweep — the runtime creates it.
import type { AdapterRecord, DeviceDescriptorRecord, PageEnv } from "../schemas.ts";
import { requestableFeature } from "./features.ts";

export interface Gpu {
  readonly device: GPUDevice;
  readonly descriptor: DeviceDescriptorRecord;
  // False when the device belongs to the reference arm: destroying it is then ORT's business.
  readonly owned: boolean;
}

// Everything the kernels need from the device they end up on. The limits are minima: a device
// that clears them runs the kernels, and one that does not fails here by name rather than as a
// Dawn validation error inside a pipeline build.
export interface DeviceNeeds {
  readonly features: readonly string[];
  readonly limits: Readonly<Record<string, number>>;
}

// The weight blob is one 145 MB storage buffer bound in slices, so the 128 MiB default binding
// size would reject it; every limit here is raised to the adapter maximum, not to a guess.
const RAISED_LIMITS = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxStorageBuffersPerShaderStage",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupsPerDimension",
  "maxBindGroups",
  "maxBindingsPerBindGroup",
] as const;

function limitsRecord(limits: GPUSupportedLimits): Record<string, number> {
  const record: Record<string, number> = {};
  for (const key of Object.getOwnPropertyNames(Object.getPrototypeOf(limits))) {
    const value = Reflect.get(limits, key);
    if (typeof value === "number") {
      record[key] = value;
    }
  }
  return record;
}

export async function requestAdapter(powerPreference: GPUPowerPreference): Promise<GPUAdapter> {
  if (navigator.gpu === undefined) {
    throw new Error("navigator.gpu unavailable");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference });
  if (adapter === null) {
    throw new Error("requestAdapter returned null");
  }
  return adapter;
}

export function adapterRecordOf(adapter: GPUAdapter): AdapterRecord {
  return {
    vendor: adapter.info.vendor,
    architecture: adapter.info.architecture,
    device: adapter.info.device,
    description: adapter.info.description,
    isFallbackAdapter: adapter.info.isFallbackAdapter,
    subgroupMinSize: adapter.info.subgroupMinSize,
    subgroupMaxSize: adapter.info.subgroupMaxSize,
    features: Array.from(adapter.features).sort(),
    limits: limitsRecord(adapter.limits),
  };
}

export function assertDeviceMeets(gpu: Gpu, needs: DeviceNeeds, expectedSubgroupSize: number): void {
  const info = gpu.descriptor.adapter;
  if (info.isFallbackAdapter) {
    throw new Error("SOFTWARE_FALLBACK: adapter.info.isFallbackAdapter");
  }
  // The matvec kernels are compiled against one subgroup width. A device whose width is not
  // uniform, or is not the compiled width, would silently drop rows, so it fails here instead.
  if (info.subgroupMinSize !== expectedSubgroupSize || info.subgroupMaxSize !== expectedSubgroupSize) {
    throw new Error(`subgroup size ${info.subgroupMinSize}..${info.subgroupMaxSize} != compiled ${expectedSubgroupSize}`);
  }
  for (const feature of needs.features) {
    const name = requestableFeature(feature);
    if (!gpu.device.features.has(name)) {
      throw new Error(`device lacks required feature ${name}`);
    }
  }
  for (const [name, minimum] of Object.entries(needs.limits)) {
    const value = Reflect.get(gpu.device.limits, name);
    if (typeof value !== "number") {
      throw new Error(`device limit ${name} missing`);
    }
    if (value < minimum) {
      throw new Error(`device limit ${name} = ${value} < required ${minimum}`);
    }
  }
}

async function createDevice(needs: DeviceNeeds, powerPreference: GPUPowerPreference): Promise<Gpu> {
  const adapter = await requestAdapter(powerPreference);
  const requiredFeatures: GPUFeatureName[] = [];
  for (const feature of needs.features) {
    const name = requestableFeature(feature);
    if (!adapter.features.has(name)) {
      throw new Error(`adapter lacks required feature ${name}`);
    }
    requiredFeatures.push(name);
  }
  const requiredLimits: Record<string, number> = {};
  for (const name of RAISED_LIMITS) {
    const value = adapter.limits[name];
    if (typeof value !== "number") {
      throw new Error(`adapter limit ${name} missing`);
    }
    requiredLimits[name] = value;
  }
  const hooks = globalThis.__probeHooks;
  const before = hooks.devices().length;
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits, label: "probe" });
  const descriptor = hooks.descriptors()[before];
  if (descriptor === undefined || hooks.devices()[before] !== device) {
    throw new Error("hooks did not record the probe device");
  }
  return { device, descriptor, owned: true };
}

// The session holds exactly one GPUDevice. If one already exists it was created by the reference
// arm's session and is adopted; otherwise this call creates it with adapter-maximum limits.
export async function deviceForRuntime(
  needs: DeviceNeeds,
  powerPreference: GPUPowerPreference,
  expectedSubgroupSize: number,
): Promise<Gpu> {
  const hooks = globalThis.__probeHooks;
  const existing = hooks.devices();
  if (existing.length > 1) {
    throw new Error(`SHARED_DEVICE: ${existing.length} GPUDevices exist; the session must hold one`);
  }
  const device = existing[0];
  const descriptor = hooks.descriptors()[0];
  const gpu =
    device === undefined || descriptor === undefined
      ? await createDevice(needs, powerPreference)
      : { device, descriptor, owned: false };
  assertDeviceMeets(gpu, needs, expectedSubgroupSize);
  return gpu;
}

export async function waitIdle(device: GPUDevice): Promise<void> {
  await device.queue.onSubmittedWorkDone();
}

export async function withValidation<T>(device: GPUDevice, errors: string[], label: string, fn: () => Promise<T>): Promise<T> {
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("internal");
  try {
    return await fn();
  } finally {
    const internal = await device.popErrorScope();
    const oom = await device.popErrorScope();
    const validation = await device.popErrorScope();
    for (const error of [validation, oom, internal]) {
      if (error !== null) {
        errors.push(`${label}: ${error.constructor.name}: ${error.message}`);
      }
    }
  }
}

export function timerResolutionMs(): number {
  let smallest = Number.POSITIVE_INFINITY;
  let previous = performance.now();
  for (let i = 0; i < 20000; i += 1) {
    const now = performance.now();
    const delta = now - previous;
    if (delta > 0 && delta < smallest) {
      smallest = delta;
    }
    previous = now;
  }
  return smallest;
}

export function pageEnv(): PageEnv {
  return {
    user_agent: navigator.userAgent,
    cross_origin_isolated: globalThis.crossOriginIsolated,
    timer_resolution_ms: timerResolutionMs(),
    hardware_concurrency: navigator.hardwareConcurrency,
    webgpu: navigator.gpu !== undefined,
  };
}

export async function readBuffer(device: GPUDevice, source: GPUBuffer, offset: number, bytes: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "readback" });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, offset, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ, 0, bytes);
  const copy = staging.getMappedRange(0, bytes).slice(0);
  staging.unmap();
  staging.destroy();
  return copy;
}

// Resolves timestamp query pairs (ns) into a flat Float64Array of [begin, end] per pair.
export async function resolveTimestamps(device: GPUDevice, querySet: GPUQuerySet, pairs: number): Promise<Float64Array> {
  const bytes = pairs * 2 * 8;
  const resolved = device.createBuffer({ size: bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC, label: "timestamps" });
  const encoder = device.createCommandEncoder();
  encoder.resolveQuerySet(querySet, 0, pairs * 2, resolved, 0);
  device.queue.submit([encoder.finish()]);
  const raw = await readBuffer(device, resolved, 0, bytes);
  resolved.destroy();
  const values = new BigUint64Array(raw);
  const out = new Float64Array(pairs * 2);
  for (let i = 0; i < out.length; i += 1) {
    const value = values[i];
    out[i] = value === undefined ? 0 : Number(value);
  }
  return out;
}
