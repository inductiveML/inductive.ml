// The weight-blob layout, derived from the fusion plan alone. P3 lays out its synthetic stacks
// with this module and fills them with pseudorandom bytes; `extract` derives the same placement
// from the ONNX file, since both walk one plan at one alignment. tests/layout.test.ts pins that
// agreement on the real model, so the depth sweep cannot drift into a different blob shape than
// the measured run.
import { quantMatrixBytes, type ModelShape } from "./shape.ts";
import { planStep, type PlanOptions } from "./plan.ts";

export interface SectionSpec {
  readonly name: string;
  readonly bytes: number;
}

export interface PlacedSection {
  readonly name: string;
  readonly offset: number;
  readonly bytes: number;
}

export interface Layout {
  readonly sections: readonly PlacedSection[];
  readonly totalBytes: number;
  readonly quantBytes: number;
  readonly constantBytes: number;
  readonly repackBytes: number;
}

// A K-sliced repack of one quantised matrix. The fused blocks split the reduction dimension across
// workgroups, so workgroup s reads columns [s*cols/slices, (s+1)*cols/slices) of every row. In the
// row-major original those bytes sit at stride cols/2, which on a 128 B cache line costs a 4x read
// amplification and a strided gather. The repack is slice-major: slice s holds every row's slice
// contiguously, so a workgroup's whole read is one linear run and consecutive lanes read
// consecutive chunks. It is a pure permutation of the packed nibbles and the scales -- same bytes,
// same values, same total size -- so it changes no arithmetic and both copies live in the blob.
export interface RepackSpec {
  readonly section: string;
  readonly rows: number;
  readonly cols: number;
  readonly slices: number;
}

export interface RepackGeometry {
  readonly sliceCols: number;
  readonly blocksPerSlice: number;
  readonly wordsPerRowSlice: number;
  readonly quantBytes: number;
  readonly scaleBytes: number;
}

export function repackGeometry(spec: RepackSpec, shape: ModelShape): RepackGeometry {
  const { section, rows, cols, slices } = spec;
  if (!Number.isInteger(slices) || slices < 1) {
    throw new Error(`repackGeometry: ${section} slices must be a positive integer, got ${slices}`);
  }
  if (cols % slices !== 0) {
    throw new Error(`repackGeometry: ${section} cols ${cols} is not divisible by ${slices} slices`);
  }
  const sliceCols = cols / slices;
  if (sliceCols % shape.quantBlock !== 0) {
    throw new Error(`repackGeometry: ${section} slice of ${sliceCols} columns is not a whole number of ${shape.quantBlock}-wide quantisation blocks`);
  }
  const bitsPerRowSlice = sliceCols * shape.quantBits;
  if (bitsPerRowSlice % 32 !== 0) {
    throw new Error(`repackGeometry: ${section} slice of ${sliceCols} columns is not a whole number of u32 words`);
  }
  const bytes = quantMatrixBytes(rows, cols, false, shape.quantBits, shape.quantBlock);
  return {
    sliceCols,
    blocksPerSlice: sliceCols / shape.quantBlock,
    wordsPerRowSlice: bitsPerRowSlice / 32,
    quantBytes: bytes.quant,
    scaleBytes: bytes.scales,
  };
}

// Every matrix a fused arm reads through its K-sliced copy, deduplicated by section name and in
// plan order. Derived from the plan itself so a lever that changes its slice count cannot leave a
// stale repack behind.
export function repackSpecs(shape: ModelShape, options: PlanOptions): readonly RepackSpec[] {
  const specs = new Map<string, RepackSpec>();
  for (const dispatch of planStep(shape, options)) {
    for (const read of dispatch.reads) {
      if (read.kSlices === 0) {
        continue;
      }
      if (read.zeroPoints) {
        throw new Error(`repackSpecs: ${read.section} carries zero points, which the K-sliced repack does not carry`);
      }
      const existing = specs.get(read.section);
      if (existing !== undefined && existing.slices !== read.kSlices) {
        throw new Error(`repackSpecs: ${read.section} is read at ${existing.slices} and ${read.kSlices} slices`);
      }
      specs.set(read.section, { section: read.section, rows: read.rows, cols: read.cols, slices: read.kSlices });
    }
  }
  return [...specs.values()];
}

// Constants that a single dispatch reads as one contiguous binding must not be split by padding.
// The attention projection binds [norm, q_norm, k_norm] as one range, so the three are emitted as
// one group. The split arm applies the norm in the projection dispatch and the two head norms in
// the core, but both bind the same range of the same blob, so the group stays whole and keeps the
// three at the offsets B1.1 placed them at.
interface Group {
  readonly specs: readonly SectionSpec[];
}

function align(offset: number, alignment: number): number {
  const remainder = offset % alignment;
  return remainder === 0 ? offset : offset + (alignment - remainder);
}

export function layoutGroups(groups: readonly Group[], alignment: number, start: number): { placed: PlacedSection[]; end: number } {
  const placed: PlacedSection[] = [];
  let cursor = start;
  for (const group of groups) {
    cursor = align(cursor, alignment);
    for (const spec of group.specs) {
      placed.push({ name: spec.name, offset: cursor, bytes: spec.bytes });
      cursor += spec.bytes;
    }
  }
  return { placed, end: cursor };
}

// The layout for one model shape: every matmul's packed nibbles, scales and zero points first
// (the head's blob is shared with the tied embedding, exactly as the extractor dedupes it), then
// the constants, then the rotary tables.
export function weightLayout(
  shape: ModelShape,
  options: PlanOptions,
  ropeRows: number,
  ropeRowsExtended: number,
  alignment: number,
  repacks: readonly RepackSpec[],
): Layout {
  if (ropeRowsExtended < ropeRows) {
    throw new Error(`weightLayout: the extended rotary table (${ropeRowsExtended}) is shorter than the base (${ropeRows})`);
  }
  const plan = planStep(shape, options);
  const quantGroups: Group[] = [];
  const constantGroups: Group[] = [];
  for (const dispatch of plan) {
    const weight = dispatch.weight;
    if (weight !== null) {
      const bytes = quantMatrixBytes(weight.rows, weight.cols, weight.zeroPoints, shape.quantBits, shape.quantBlock);
      const specs: SectionSpec[] = [
        { name: `${dispatch.name}.quant`, bytes: bytes.quant },
        { name: `${dispatch.name}.scales`, bytes: bytes.scales },
      ];
      if (weight.zeroPoints) {
        specs.push({ name: `${dispatch.name}.zero_points`, bytes: bytes.zeroPoints });
      }
      quantGroups.push({ specs });
    }
    switch (dispatch.kind) {
      case "norm_matvec":
      case "norm_matvec_swiglu":
        constantGroups.push({ specs: [{ name: `${dispatch.name}.norm`, bytes: shape.hidden * 4 }] });
        break;
      case "conv_core":
        constantGroups.push({ specs: [{ name: `${dispatch.name}.taps`, bytes: shape.hidden * shape.convCache * 4 }] });
        break;
      case "attn_proj":
        constantGroups.push({
          specs: [
            { name: `${dispatch.name}.norm`, bytes: shape.hidden * 4 },
            { name: `${dispatch.name}.q_norm`, bytes: shape.headDim * 4 },
            { name: `${dispatch.name}.k_norm`, bytes: shape.headDim * 4 },
          ],
        });
        break;
      case "norm_head":
        constantGroups.push({ specs: [{ name: `${dispatch.name}.norm`, bytes: shape.hidden * 4 }] });
        break;
      // The fused MLP and the two attention-core forms read constants the layout arm already placed
      // under the same names -- the core's q_norm and k_norm sit in the projection's constant group,
      // which every arm binds as one range -- so they contribute nothing of their own; the layout
      // arm is the one that defines the section set.
      case "embed":
      case "matvec_residual":
      case "attn_core_qkv":
      case "sample_partial":
      case "sample_final":
      case "attn_core_flash":
      case "attn_merge":
      case "mlp_fused":
      case "fold":
        break;
    }
  }
  const quant = layoutGroups(quantGroups, alignment, 0);
  const constants = layoutGroups(constantGroups, alignment, quant.end);
  const ropeBytes = ropeRows * (shape.headDim / 2) * 4;
  const rope = layoutGroups(
    [{ specs: [{ name: "rope.cos", bytes: ropeBytes }] }, { specs: [{ name: "rope.sin", bytes: ropeBytes }] }],
    alignment,
    constants.end,
  );
  // The repacks are appended after everything the layout arm needs, so a blob built for the layout
  // arm alone is a byte identical prefix of the full blob and the offsets of every shared section
  // are the same in both. That prefix is exactly B1.1's blob.
  const repackGroups: Group[] = repacks.map((spec) => {
    const geometry = repackGeometry(spec, shape);
    return {
      specs: [
        { name: `${spec.section}.kquant`, bytes: geometry.quantBytes },
        { name: `${spec.section}.kscales`, bytes: geometry.scaleBytes },
      ],
    };
  });
  const repacked = layoutGroups(repackGroups, alignment, rope.end);
  // The extended rotary tables go last, after the repacks, so that adding them moves nothing. They
  // are a longer slice of the same exported tensors: their first `ropeRows` rows are byte identical
  // to rope.cos and rope.sin, which the extractor asserts rather than assumes.
  const extendedBytes = ropeRowsExtended * (shape.headDim / 2) * 4;
  const ropeExtended = layoutGroups(
    [
      { specs: [{ name: "rope.cos_ext", bytes: extendedBytes }] },
      { specs: [{ name: "rope.sin_ext", bytes: extendedBytes }] },
    ],
    alignment,
    repacked.end,
  );
  const placed = [...quant.placed, ...constants.placed, ...rope.placed, ...repacked.placed, ...ropeExtended.placed];
  // The tied embedding reads the head's blob; the extractor dedupes it by content digest and the
  // runtime binds both names, so the alias is part of the layout, not a runtime special case.
  const alias: PlacedSection[] = [];
  for (const section of placed) {
    if (section.name.startsWith("head.") && section.name !== "head.norm") {
      alias.push({ name: `embed.${section.name.slice("head.".length)}`, offset: section.offset, bytes: section.bytes });
    }
  }
  const sections = [...placed, ...alias];
  const quantBytes = quant.placed.reduce((sum, section) => sum + section.bytes, 0);
  const sum = (list: readonly PlacedSection[]): number => list.reduce((total, section) => total + section.bytes, 0);
  const constantBytes = sum(constants.placed) + sum(rope.placed) + sum(ropeExtended.placed);
  const repackBytes = sum(repacked.placed);
  return { sections, totalBytes: align(ropeExtended.end, alignment), quantBytes, constantBytes, repackBytes };
}
