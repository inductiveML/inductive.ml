// The weight blob: one GPU buffer that every matvec binds slices of. The real blob is fetched and
// its digest checked in-page against the manifest the weights stage pinned; the depth sweep's
// synthetic blobs are filled here from the same layout so it exercises identical bindings.
import { Rng } from "../pure/stats.ts";
import type { Layout } from "../pure/sections.ts";
import type { Section, SectionTable } from "./runtime.ts";

export interface WeightBlob {
  readonly buffer: GPUBuffer;
  readonly bytes: number;
  readonly sha256: string;
  readonly fetchMs: number;
}

export function sectionTableOf(sections: readonly { name: string; offset: number; bytes: number }[]): SectionTable {
  const table = new Map<string, Section>();
  for (const section of sections) {
    if (table.has(section.name)) {
      throw new Error(`duplicate weight section ${section.name}`);
    }
    table.set(section.name, { offset: section.offset, bytes: section.bytes });
  }
  return table;
}

function hex(digest: ArrayBuffer): string {
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

const WEIGHT_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

export async function fetchWeights(device: GPUDevice, url: string, expectedBytes: number): Promise<WeightBlob> {
  const start = performance.now();
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`weights fetch ${url}: ${response.status}`);
  }
  const bytes = await response.arrayBuffer();
  const fetchMs = performance.now() - start;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(`weights are ${bytes.byteLength} B, manifest says ${expectedBytes} B`);
  }
  const sha256 = hex(await crypto.subtle.digest("SHA-256", bytes));
  const buffer = device.createBuffer({ size: bytes.byteLength, usage: WEIGHT_USAGE, mappedAtCreation: true, label: "weights" });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(bytes));
  buffer.unmap();
  return { buffer, bytes: bytes.byteLength, sha256, fetchMs };
}

// The depth sweep's synthetic blob. Quantised payloads are pseudorandom bytes (the arithmetic is the same
// whatever the nibbles are), but every value the kernels interpret as a float is filled with a
// realistic magnitude so no dispatch runs on NaNs or denormals and mis-times the sweep.
export function synthesizeWeights(device: GPUDevice, layout: Layout, headDim: number, ropeTheta: number, seed: number): WeightBlob {
  const start = performance.now();
  const buffer = device.createBuffer({ size: layout.totalBytes, usage: WEIGHT_USAGE, mappedAtCreation: true, label: "weights_synthetic" });
  const mapped = buffer.getMappedRange();
  const bytes = new Uint8Array(mapped);
  const rng = new Rng(seed);
  const seen = new Set<number>();
  for (const section of layout.sections) {
    if (seen.has(section.offset)) {
      continue;
    }
    seen.add(section.offset);
    if (section.name.endsWith(".quant") || section.name.endsWith(".kquant") || section.name.endsWith(".zero_points")) {
      const fill = section.name.endsWith(".zero_points") ? null : rng;
      for (let i = 0; i < section.bytes; i += 1) {
        bytes[section.offset + i] = fill === null ? 0x88 : fill.uint32() & 0xff;
      }
    } else if (section.name.endsWith(".scales") || section.name.endsWith(".kscales")) {
      const values = new Float32Array(mapped, section.offset, section.bytes / 4);
      for (let i = 0; i < values.length; i += 1) {
        values[i] = 0.002 + rng.float() * 0.01;
      }
    } else if (section.name.endsWith(".norm") || section.name.endsWith(".q_norm") || section.name.endsWith(".k_norm")) {
      const values = new Float32Array(mapped, section.offset, section.bytes / 4);
      for (let i = 0; i < values.length; i += 1) {
        values[i] = 0.9 + rng.float() * 0.2;
      }
    } else if (section.name.endsWith(".taps")) {
      const values = new Float32Array(mapped, section.offset, section.bytes / 4);
      for (let i = 0; i < values.length; i += 1) {
        values[i] = rng.float() - 0.5;
      }
    } else if (section.name.startsWith("rope.cos") || section.name.startsWith("rope.sin")) {
      // Both the 1024-row tables and the 2560-row extensions the depth sweep indexes into.
      const half = headDim / 2;
      const values = new Float32Array(mapped, section.offset, section.bytes / 4);
      const useCos = section.name.startsWith("rope.cos");
      for (let row = 0; row * half < values.length; row += 1) {
        for (let i = 0; i < half; i += 1) {
          const angle = row / Math.pow(ropeTheta, (2 * i) / headDim);
          values[row * half + i] = useCos ? Math.cos(angle) : Math.sin(angle);
        }
      }
    } else {
      throw new Error(`synthesizeWeights: unclassified section ${section.name}`);
    }
  }
  buffer.unmap();
  return { buffer, bytes: layout.totalBytes, sha256: "", fetchMs: performance.now() - start };
}
