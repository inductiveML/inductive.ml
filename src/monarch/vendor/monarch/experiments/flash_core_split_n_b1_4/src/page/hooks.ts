// Injected through Playwright addInitScript before any page script. Patches the WebGPU
// prototypes so every dispatch, submit, pass, bind group, buffer and pipeline made by ORT (or
// by the probe's own kernels) is counted per step, captures the device ORT requests, appends
// pass-level timestamp queries, and can append one extra command buffer to a step's first submit.
import type { DeviceDescriptorRecord, HookSnapshot, TimestampPass } from "../schemas.ts";
import { requestableFeature } from "./features.ts";
import type { ProbeHooks } from "./api.ts";

interface Counters {
  dispatches: number;
  dispatchesIndirect: number;
  submits: number;
  commandBuffers: number;
  passes: number;
  bindGroups: number;
  writeBuffers: number;
  writeBufferBytes: number;
  createBuffers: number;
  createBufferBytes: number;
  destroyBuffers: number;
  pipelines: number;
  pipelinesAsync: number;
  pipelineCompileMs: number;
  shaderModules: number;
  mapAsyncs: number;
  copyBufferToBuffer: number;
  submitOffsets: number[];
  injected: number;
}

interface PassRecord {
  index: number;
  dispatches: number;
}

interface TimestampState {
  device: GPUDevice;
  querySet: GPUQuerySet;
  capacity: number;
  next: number;
  resolveBuffer: GPUBuffer;
  stagingBuffer: GPUBuffer;
  passes: PassRecord[];
}

function freshCounters(): Counters {
  return {
    dispatches: 0,
    dispatchesIndirect: 0,
    submits: 0,
    commandBuffers: 0,
    passes: 0,
    bindGroups: 0,
    writeBuffers: 0,
    writeBufferBytes: 0,
    createBuffers: 0,
    createBufferBytes: 0,
    destroyBuffers: 0,
    pipelines: 0,
    pipelinesAsync: 0,
    pipelineCompileMs: 0,
    shaderModules: 0,
    mapAsyncs: 0,
    copyBufferToBuffer: 0,
    submitOffsets: [],
    injected: 0,
  };
}

function limitsRecord(limits: GPUSupportedLimits): Record<string, number> {
  const record: Record<string, number> = {};
  const proto = Object.getPrototypeOf(limits);
  for (const key of Object.getOwnPropertyNames(proto)) {
    if (key === "constructor") {
      continue;
    }
    const descriptor = Object.getOwnPropertyDescriptor(proto, key);
    if (descriptor === undefined || typeof descriptor.get !== "function") {
      continue;
    }
    const value = descriptor.get.call(limits);
    if (typeof value === "number") {
      record[key] = value;
    }
  }
  return record;
}

function limitsFromDescriptor(limits: Record<string, GPUSize64 | undefined> | undefined): Record<string, number> {
  const record: Record<string, number> = {};
  if (limits === undefined) {
    return record;
  }
  for (const [key, value] of Object.entries(limits)) {
    if (typeof value === "number") {
      record[key] = value;
    }
  }
  return record;
}

(() => {
  if (typeof navigator === "undefined" || navigator.gpu === undefined) {
    globalThis.__probeHooks = {
      installed: false,
      configure: () => undefined,
      beginStep: () => undefined,
      endStep: () => {
        throw new Error("WebGPU unavailable");
      },
      snapshot: () => {
        throw new Error("WebGPU unavailable");
      },
      setInjection: () => undefined,
      enableTimestamps: () => undefined,
      disableTimestamps: () => undefined,
      collectTimestamps: () => Promise.resolve([]),
      devices: () => [],
      descriptors: () => [],
      errors: () => [],
      lost: () => [],
      liveBytes: () => 0,
      suspend: (fn) => fn(),
      suspendAsync: (fn) => fn(),
    };
    return;
  }

  let active = false;
  let suspended = false;
  let stepStart = 0;
  let stepEnd = 0;
  let counters = freshCounters();
  let injection: GPUCommandBuffer | null = null;
  let timestamps: TimestampState | null = null;
  let liveBytes = 0;
  let extraFeatures: string[] = [];
  const devices: GPUDevice[] = [];
  const descriptors: DeviceDescriptorRecord[] = [];
  const errors: string[] = [];
  const lost: string[] = [];
  const bufferSizes = new WeakMap<GPUBuffer, number>();
  const passRecords = new WeakMap<GPUComputePassEncoder, PassRecord>();

  const counting = (): boolean => active && !suspended;

  const adapterProto = GPUAdapter.prototype;
  const originalRequestDevice = adapterProto.requestDevice;
  adapterProto.requestDevice = async function (this: GPUAdapter, descriptor?: GPUDeviceDescriptor): Promise<GPUDevice> {
    const requested: GPUFeatureName[] = descriptor === undefined ? [] : [...(descriptor.requiredFeatures ?? [])];
    const effective: GPUFeatureName[] = [...requested];
    const added: GPUFeatureName[] = [];
    for (const feature of extraFeatures) {
      const name = requestableFeature(feature);
      if (!effective.includes(name) && this.features.has(name)) {
        effective.push(name);
        added.push(name);
      }
    }
    const augmented: GPUDeviceDescriptor = { ...(descriptor ?? {}), requiredFeatures: effective };
    const info = this.info;
    const record: DeviceDescriptorRecord = {
      requestedFeatures: requested.map(String),
      requiredFeatures: effective.map(String),
      addedFeatures: added.map(String),
      requiredLimits: limitsFromDescriptor(descriptor?.requiredLimits),
      label: descriptor?.label ?? "",
      adapter: {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
        isFallbackAdapter: info.isFallbackAdapter,
        subgroupMinSize: info.subgroupMinSize,
        subgroupMaxSize: info.subgroupMaxSize,
        features: [...this.features].map(String).sort(),
        limits: limitsRecord(this.limits),
      },
    };
    const device = await originalRequestDevice.call(this, augmented);
    devices.push(device);
    descriptors.push(record);
    device.addEventListener("uncapturederror", (event) => {
      errors.push(`${event.error.constructor.name}: ${event.error.message}`);
    });
    device.lost.then((info2) => {
      lost.push(`${info2.reason}: ${info2.message}`);
    });
    return device;
  };

  const deviceProto = GPUDevice.prototype;
  const originalCreateBuffer = deviceProto.createBuffer;
  deviceProto.createBuffer = function (this: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = originalCreateBuffer.call(this, descriptor);
    bufferSizes.set(buffer, descriptor.size);
    liveBytes += descriptor.size;
    if (counting()) {
      counters.createBuffers += 1;
      counters.createBufferBytes += descriptor.size;
    }
    return buffer;
  };
  const originalCreateBindGroup = deviceProto.createBindGroup;
  deviceProto.createBindGroup = function (this: GPUDevice, descriptor: GPUBindGroupDescriptor): GPUBindGroup {
    if (counting()) {
      counters.bindGroups += 1;
    }
    return originalCreateBindGroup.call(this, descriptor);
  };
  const originalCreateShaderModule = deviceProto.createShaderModule;
  deviceProto.createShaderModule = function (this: GPUDevice, descriptor: GPUShaderModuleDescriptor): GPUShaderModule {
    if (counting()) {
      counters.shaderModules += 1;
    }
    return originalCreateShaderModule.call(this, descriptor);
  };
  const originalCreateComputePipeline = deviceProto.createComputePipeline;
  deviceProto.createComputePipeline = function (this: GPUDevice, descriptor: GPUComputePipelineDescriptor): GPUComputePipeline {
    const start = performance.now();
    const pipeline = originalCreateComputePipeline.call(this, descriptor);
    if (counting()) {
      counters.pipelines += 1;
      counters.pipelineCompileMs += performance.now() - start;
    }
    return pipeline;
  };
  const originalCreateComputePipelineAsync = deviceProto.createComputePipelineAsync;
  deviceProto.createComputePipelineAsync = async function (this: GPUDevice, descriptor: GPUComputePipelineDescriptor): Promise<GPUComputePipeline> {
    const start = performance.now();
    const pipeline = await originalCreateComputePipelineAsync.call(this, descriptor);
    if (counting()) {
      counters.pipelinesAsync += 1;
      counters.pipelineCompileMs += performance.now() - start;
    }
    return pipeline;
  };

  const bufferProto = GPUBuffer.prototype;
  const originalDestroy = bufferProto.destroy;
  bufferProto.destroy = function (this: GPUBuffer): undefined {
    const size = bufferSizes.get(this);
    if (size !== undefined) {
      liveBytes -= size;
      bufferSizes.delete(this);
    }
    if (counting()) {
      counters.destroyBuffers += 1;
    }
    return originalDestroy.call(this);
  };
  const originalMapAsync = bufferProto.mapAsync;
  bufferProto.mapAsync = function (this: GPUBuffer, mode: GPUMapModeFlags, offset?: GPUSize64, size?: GPUSize64): Promise<undefined> {
    if (counting()) {
      counters.mapAsyncs += 1;
    }
    return originalMapAsync.call(this, mode, offset, size);
  };

  const queueProto = GPUQueue.prototype;
  const originalSubmit = queueProto.submit;
  queueProto.submit = function (this: GPUQueue, commandBuffers: Iterable<GPUCommandBuffer>): undefined {
    let list = [...commandBuffers];
    if (counting()) {
      if (injection !== null) {
        list = [...list, injection];
        injection = null;
        counters.injected += 1;
      }
      counters.submits += 1;
      counters.commandBuffers += list.length;
      counters.submitOffsets.push(performance.now() - stepStart);
    }
    return originalSubmit.call(this, list);
  };
  const originalWriteBuffer = queueProto.writeBuffer;
  queueProto.writeBuffer = function (
    this: GPUQueue,
    buffer: GPUBuffer,
    bufferOffset: GPUSize64,
    data: GPUAllowSharedBufferSource,
    dataOffset?: GPUSize64,
    size?: GPUSize64,
  ): undefined {
    if (counting()) {
      counters.writeBuffers += 1;
      const total = size !== undefined ? size : data.byteLength - (dataOffset ?? 0);
      counters.writeBufferBytes += total;
    }
    return originalWriteBuffer.call(this, buffer, bufferOffset, data, dataOffset, size);
  };

  const encoderProto = GPUCommandEncoder.prototype;
  const originalBeginComputePass = encoderProto.beginComputePass;
  encoderProto.beginComputePass = function (this: GPUCommandEncoder, descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder {
    let effective = descriptor;
    let record: PassRecord | null = null;
    if (counting()) {
      counters.passes += 1;
      if (timestamps !== null && timestamps.next < timestamps.capacity && (descriptor === undefined || descriptor.timestampWrites === undefined)) {
        const index = timestamps.next;
        timestamps.next += 1;
        effective = {
          ...(descriptor ?? {}),
          timestampWrites: {
            querySet: timestamps.querySet,
            beginningOfPassWriteIndex: 2 * index,
            endOfPassWriteIndex: 2 * index + 1,
          },
        };
        record = { index, dispatches: 0 };
        timestamps.passes.push(record);
      }
    }
    const pass = originalBeginComputePass.call(this, effective);
    if (record !== null) {
      passRecords.set(pass, record);
    }
    return pass;
  };
  const originalCopyBufferToBuffer = encoderProto.copyBufferToBuffer;
  const shortCopy: (this: GPUCommandEncoder, source: GPUBuffer, destination: GPUBuffer, size?: GPUSize64) => undefined = originalCopyBufferToBuffer;
  const patchedCopy: typeof encoderProto.copyBufferToBuffer = function (
    this: GPUCommandEncoder,
    source: GPUBuffer,
    second: GPUSize64 | GPUBuffer,
    third?: GPUBuffer | GPUSize64,
    destinationOffset?: GPUSize64,
    size?: GPUSize64,
  ): undefined {
    if (counting()) {
      counters.copyBufferToBuffer += 1;
    }
    if (typeof second === "number") {
      if (!(third instanceof GPUBuffer) || destinationOffset === undefined) {
        throw new TypeError("copyBufferToBuffer: bad five-argument form");
      }
      return originalCopyBufferToBuffer.call(this, source, second, third, destinationOffset, size);
    }
    if (third !== undefined && typeof third !== "number") {
      throw new TypeError("copyBufferToBuffer: bad three-argument form");
    }
    return shortCopy.call(this, source, second, third);
  };
  encoderProto.copyBufferToBuffer = patchedCopy;

  const passProto = GPUComputePassEncoder.prototype;
  const originalDispatch = passProto.dispatchWorkgroups;
  passProto.dispatchWorkgroups = function (this: GPUComputePassEncoder, x: GPUSize32, y?: GPUSize32, z?: GPUSize32): undefined {
    if (counting()) {
      counters.dispatches += 1;
      const record = passRecords.get(this);
      if (record !== undefined) {
        record.dispatches += 1;
      }
    }
    return originalDispatch.call(this, x, y, z);
  };
  const originalDispatchIndirect = passProto.dispatchWorkgroupsIndirect;
  passProto.dispatchWorkgroupsIndirect = function (this: GPUComputePassEncoder, buffer: GPUBuffer, offset: GPUSize64): undefined {
    if (counting()) {
      counters.dispatches += 1;
      counters.dispatchesIndirect += 1;
      const record = passRecords.get(this);
      if (record !== undefined) {
        record.dispatches += 1;
      }
    }
    return originalDispatchIndirect.call(this, buffer, offset);
  };

  const snapshot = (): HookSnapshot => {
    const end = active ? performance.now() : stepEnd;
    const first = counters.submitOffsets[0];
    const last = counters.submitOffsets[counters.submitOffsets.length - 1];
    return {
      dispatches: counters.dispatches,
      dispatches_indirect: counters.dispatchesIndirect,
      submits: counters.submits,
      command_buffers: counters.commandBuffers,
      passes: counters.passes,
      bind_groups: counters.bindGroups,
      write_buffers: counters.writeBuffers,
      write_buffer_bytes: counters.writeBufferBytes,
      create_buffers: counters.createBuffers,
      create_buffer_bytes: counters.createBufferBytes,
      destroy_buffers: counters.destroyBuffers,
      pipelines: counters.pipelines,
      pipelines_async: counters.pipelinesAsync,
      pipeline_compile_ms: counters.pipelineCompileMs,
      shader_modules: counters.shaderModules,
      map_asyncs: counters.mapAsyncs,
      copy_buffer_to_buffer: counters.copyBufferToBuffer,
      first_submit_ms: first === undefined ? -1 : first,
      last_submit_ms: last === undefined ? -1 : last,
      submit_offsets_ms: [...counters.submitOffsets],
      injected: counters.injected,
      live_gpu_bytes: liveBytes,
      step_ms: end - stepStart,
    };
  };

  const hooks: ProbeHooks = {
    installed: true,
    configure(features) {
      extraFeatures = [...features];
    },
    beginStep() {
      counters = freshCounters();
      if (timestamps !== null) {
        timestamps.next = 0;
        timestamps.passes = [];
      }
      stepStart = performance.now();
      stepEnd = stepStart;
      active = true;
    },
    endStep() {
      stepEnd = performance.now();
      active = false;
      return snapshot();
    },
    snapshot,
    setInjection(commandBuffer) {
      injection = commandBuffer;
    },
    enableTimestamps(device, capacityPairs) {
      if (timestamps !== null) {
        hooks.disableTimestamps();
      }
      // WebGPU caps a query set at 4096 entries; each timed pass consumes two.
      if (capacityPairs < 1 || capacityPairs > 2048) {
        throw new Error(`timestamp capacity ${capacityPairs} pairs outside 1..2048`);
      }
      const querySet = device.createQuerySet({ type: "timestamp", count: 2 * capacityPairs });
      const resolveBuffer = device.createBuffer({ size: 16 * capacityPairs, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const stagingBuffer = device.createBuffer({ size: 16 * capacityPairs, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      timestamps = { device, querySet, capacity: capacityPairs, next: 0, resolveBuffer, stagingBuffer, passes: [] };
    },
    disableTimestamps() {
      if (timestamps === null) {
        return;
      }
      timestamps.querySet.destroy();
      timestamps.resolveBuffer.destroy();
      timestamps.stagingBuffer.destroy();
      timestamps = null;
    },
    async collectTimestamps() {
      const state = timestamps;
      if (state === null || state.next === 0) {
        return [];
      }
      const previous = suspended;
      suspended = true;
      try {
        const encoder = state.device.createCommandEncoder();
        encoder.resolveQuerySet(state.querySet, 0, 2 * state.next, state.resolveBuffer, 0);
        encoder.copyBufferToBuffer(state.resolveBuffer, 0, state.stagingBuffer, 0, 16 * state.next);
        state.device.queue.submit([encoder.finish()]);
        await state.stagingBuffer.mapAsync(GPUMapMode.READ, 0, 16 * state.next);
        const values = new BigUint64Array(state.stagingBuffer.getMappedRange(0, 16 * state.next).slice(0));
        state.stagingBuffer.unmap();
        const result: TimestampPass[] = [];
        for (const record of state.passes) {
          const begin = values[2 * record.index];
          const end = values[2 * record.index + 1];
          if (begin === undefined || end === undefined) {
            throw new Error("timestamp index out of range");
          }
          result.push({ index: record.index, dispatches: record.dispatches, begin_ns: Number(begin), end_ns: Number(end) });
        }
        return result;
      } finally {
        suspended = previous;
      }
    },
    devices: () => devices,
    descriptors: () => descriptors,
    errors: () => errors,
    lost: () => lost,
    liveBytes: () => liveBytes,
    suspend(fn) {
      const previous = suspended;
      suspended = true;
      try {
        return fn();
      } finally {
        suspended = previous;
      }
    },
    async suspendAsync(fn) {
      const previous = suspended;
      suspended = true;
      try {
        return await fn();
      } finally {
        suspended = previous;
      }
    },
  };
  globalThis.__probeHooks = hooks;
})();
