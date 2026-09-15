// Seeded statistics used by every stage. The hierarchical bootstrap mirrors
// arc_closure_v2_3 p0._median_ci (resample blocks, then samples within each block, take the
// median, 2.5/97.5 percentiles of the replicate distribution). The generator is xoshiro128**
// seeded through splitmix32, so replicates are deterministic but not bit-identical to numpy.

export class Rng {
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;

  constructor(seed: number) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      throw new Error(`seed must be a uint32, got ${seed}`);
    }
    let state = seed >>> 0;
    const splitmix = (): number => {
      state = (state + 0x9e3779b9) >>> 0;
      let z = state;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.s0 = splitmix();
    this.s1 = splitmix();
    this.s2 = splitmix();
    this.s3 = splitmix();
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) {
      this.s0 = 1;
    }
  }

  uint32(): number {
    const result = (Math.imul(rotl(Math.imul(this.s1, 5) >>> 0, 7), 9) >>> 0);
    const t = (this.s1 << 9) >>> 0;
    this.s2 ^= this.s0;
    this.s3 ^= this.s1;
    this.s1 ^= this.s2;
    this.s0 ^= this.s3;
    this.s2 ^= t;
    this.s3 = rotl(this.s3, 11);
    return result;
  }

  float(): number {
    return this.uint32() / 4294967296;
  }

  int(bound: number): number {
    if (!Number.isInteger(bound) || bound <= 0) {
      throw new Error(`bound must be a positive integer, got ${bound}`);
    }
    const value = Math.floor(this.float() * bound);
    return value === bound ? bound - 1 : value;
  }

  shuffle<T>(items: T[]): T[] {
    for (let index = items.length - 1; index > 0; index -= 1) {
      const other = this.int(index + 1);
      const a = items[index];
      const b = items[other];
      if (a === undefined || b === undefined) {
        throw new Error("shuffle index out of range");
      }
      items[index] = b;
      items[other] = a;
    }
    return items;
  }
}

function rotl(value: number, shift: number): number {
  return ((value << shift) | (value >>> (32 - shift))) >>> 0;
}

export function sortedCopy(values: ArrayLike<number>): Float64Array {
  const copy = Float64Array.from(values);
  copy.sort();
  return copy;
}

// numpy.percentile with the default linear interpolation, on already sorted input.
export function percentileSorted(sorted: Float64Array, q: number): number {
  if (sorted.length === 0) {
    throw new Error("percentile of empty array");
  }
  if (q < 0 || q > 100) {
    throw new Error(`percentile q out of range: ${q}`);
  }
  const position = (q / 100) * (sorted.length - 1);
  const low = Math.floor(position);
  const high = Math.ceil(position);
  const a = sorted[low];
  const b = sorted[high];
  if (a === undefined || b === undefined) {
    throw new Error("percentile index out of range");
  }
  return a + (b - a) * (position - low);
}

export function percentile(values: ArrayLike<number>, q: number): number {
  return percentileSorted(sortedCopy(values), q);
}

export function median(values: ArrayLike<number>): number {
  return percentile(values, 50);
}

export function sum(values: ArrayLike<number>): number {
  let total = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === undefined) {
      throw new Error("sum index out of range");
    }
    total += value;
  }
  return total;
}

export function mean(values: ArrayLike<number>): number {
  if (values.length === 0) {
    throw new Error("mean of empty array");
  }
  let total = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === undefined) {
      throw new Error("mean index out of range");
    }
    total += value;
  }
  return total / values.length;
}

export function std(values: ArrayLike<number>): number {
  const centre = mean(values);
  let total = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === undefined) {
      throw new Error("std index out of range");
    }
    total += (value - centre) * (value - centre);
  }
  return Math.sqrt(total / values.length);
}

export function flatten(blocks: readonly (readonly number[])[]): Float64Array {
  let total = 0;
  for (const block of blocks) {
    total += block.length;
  }
  const flat = new Float64Array(total);
  let offset = 0;
  for (const block of blocks) {
    flat.set(block, offset);
    offset += block.length;
  }
  return flat;
}

// The hierarchical bootstrap needs blocks; a burst of sequential samples is cut into `count`
// contiguous chunks so the resample respects the order the samples were taken in.
export function contiguousBlocks(values: readonly number[], count: number): number[][] {
  if (count < 1 || values.length % count !== 0) {
    throw new Error(`contiguousBlocks: ${values.length} values do not divide into ${count} blocks`);
  }
  const width = values.length / count;
  const blocks: number[][] = [];
  for (let block = 0; block < count; block += 1) {
    blocks.push(values.slice(block * width, (block + 1) * width));
  }
  return blocks;
}

function requireMatrix(blocks: readonly (readonly number[])[]): number {
  if (blocks.length === 0) {
    throw new Error("timing values must be block x sample");
  }
  const width = blocks[0]?.length;
  if (width === undefined || width === 0) {
    throw new Error("timing values must be block x sample");
  }
  for (const block of blocks) {
    if (block.length !== width) {
      throw new Error("timing blocks must have equal sample counts");
    }
  }
  return width;
}

export interface MedianCi {
  readonly p50: number;
  readonly ci: readonly [number, number];
}

export function medianCi(blocks: readonly (readonly number[])[], seed: number, replicates: number): MedianCi {
  const samples = requireMatrix(blocks);
  const rng = new Rng(seed);
  const draws = new Float64Array(replicates);
  const scratch = new Float64Array(blocks.length * samples);
  for (let replicate = 0; replicate < replicates; replicate += 1) {
    let offset = 0;
    for (let slot = 0; slot < blocks.length; slot += 1) {
      const block = blocks[rng.int(blocks.length)];
      if (block === undefined) {
        throw new Error("block index out of range");
      }
      for (let sample = 0; sample < samples; sample += 1) {
        const value = block[rng.int(samples)];
        if (value === undefined) {
          throw new Error("sample index out of range");
        }
        scratch[offset] = value;
        offset += 1;
      }
    }
    scratch.sort();
    draws[replicate] = percentileSorted(scratch, 50);
  }
  const sortedDraws = sortedCopy(draws);
  return {
    p50: median(flatten(blocks)),
    ci: [percentileSorted(sortedDraws, 2.5), percentileSorted(sortedDraws, 97.5)],
  };
}

export interface Summary {
  readonly count: number;
  readonly mean_ms: number;
  readonly std_ms: number;
  readonly p50_ms: number;
  readonly p95_ms: number;
  readonly block_median_cv: number;
  readonly p50_ci: readonly [number, number];
}

// Port of nextlat_at_true_target_scale_v1_8 statistics.summarize plus the hierarchical CI.
export function summarize(blocks: readonly (readonly number[])[], seed: number, replicates: number): Summary {
  requireMatrix(blocks);
  const flat = flatten(blocks);
  const sorted = sortedCopy(flat);
  const medians = blocks.map((block) => median(block));
  const ci = medianCi(blocks, seed, replicates);
  return {
    count: flat.length,
    mean_ms: mean(flat),
    std_ms: std(flat),
    p50_ms: percentileSorted(sorted, 50),
    p95_ms: percentileSorted(sorted, 95),
    block_median_cv: std(medians) / Math.max(mean(medians), 1e-12),
    p50_ci: ci.ci,
  };
}

export interface OlsFit {
  readonly intercept: number;
  readonly slope: number;
}

export function olsFit(xs: readonly number[], ys: readonly number[]): OlsFit {
  if (xs.length !== ys.length || xs.length < 2) {
    throw new Error("olsFit needs at least two paired points");
  }
  const xMean = mean(xs);
  const yMean = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let index = 0; index < xs.length; index += 1) {
    const x = xs[index];
    const y = ys[index];
    if (x === undefined || y === undefined) {
      throw new Error("olsFit index out of range");
    }
    sxx += (x - xMean) * (x - xMean);
    sxy += (x - xMean) * (y - yMean);
  }
  if (sxx === 0) {
    throw new Error("olsFit needs varying x");
  }
  const slope = sxy / sxx;
  return { intercept: yMean - slope * xMean, slope };
}

export interface OlsCi {
  readonly intercept: number;
  readonly intercept_ci: readonly [number, number];
  readonly slope: number;
  readonly slope_ci: readonly [number, number];
}

// Rounds are the resampling unit: each round contributes one median per x level.
export function olsRoundBootstrap(
  xs: readonly number[],
  roundMedians: readonly (readonly number[])[],
  seed: number,
  replicates: number,
): OlsCi {
  if (roundMedians.length === 0) {
    throw new Error("olsRoundBootstrap needs rounds");
  }
  for (const round of roundMedians) {
    if (round.length !== xs.length) {
      throw new Error("each round must hold one median per x level");
    }
  }
  const pooledX: number[] = [];
  const pooledY: number[] = [];
  for (const round of roundMedians) {
    for (let level = 0; level < xs.length; level += 1) {
      const x = xs[level];
      const y = round[level];
      if (x === undefined || y === undefined) {
        throw new Error("olsRoundBootstrap index out of range");
      }
      pooledX.push(x);
      pooledY.push(y);
    }
  }
  const point = olsFit(pooledX, pooledY);
  const rng = new Rng(seed);
  const intercepts = new Float64Array(replicates);
  const slopes = new Float64Array(replicates);
  for (let replicate = 0; replicate < replicates; replicate += 1) {
    const sampleX: number[] = [];
    const sampleY: number[] = [];
    for (let slot = 0; slot < roundMedians.length; slot += 1) {
      const round = roundMedians[rng.int(roundMedians.length)];
      if (round === undefined) {
        throw new Error("round index out of range");
      }
      for (let level = 0; level < xs.length; level += 1) {
        const x = xs[level];
        const y = round[level];
        if (x === undefined || y === undefined) {
          throw new Error("olsRoundBootstrap index out of range");
        }
        sampleX.push(x);
        sampleY.push(y);
      }
    }
    const fit = olsFit(sampleX, sampleY);
    intercepts[replicate] = fit.intercept;
    slopes[replicate] = fit.slope;
  }
  const sortedIntercepts = sortedCopy(intercepts);
  const sortedSlopes = sortedCopy(slopes);
  return {
    intercept: point.intercept,
    intercept_ci: [percentileSorted(sortedIntercepts, 2.5), percentileSorted(sortedIntercepts, 97.5)],
    slope: point.slope,
    slope_ci: [percentileSorted(sortedSlopes, 2.5), percentileSorted(sortedSlopes, 97.5)],
  };
}

// Ratio of medians with the paired hierarchical bootstrap of statistics.paired_speedup.
export interface PairedRatio {
  readonly p50x: number;
  readonly p50x_ci: readonly [number, number];
}

export function pairedRatio(
  numerator: readonly (readonly number[])[],
  denominator: readonly (readonly number[])[],
  seed: number,
  replicates: number,
): PairedRatio {
  const samples = requireMatrix(numerator);
  if (requireMatrix(denominator) !== samples || numerator.length !== denominator.length) {
    throw new Error("paired timing arrays differ");
  }
  const rng = new Rng(seed);
  const draws = new Float64Array(replicates);
  const left = new Float64Array(numerator.length * samples);
  const right = new Float64Array(numerator.length * samples);
  for (let replicate = 0; replicate < replicates; replicate += 1) {
    let offset = 0;
    for (let slot = 0; slot < numerator.length; slot += 1) {
      const blockIndex = rng.int(numerator.length);
      const a = numerator[blockIndex];
      const b = denominator[blockIndex];
      if (a === undefined || b === undefined) {
        throw new Error("block index out of range");
      }
      for (let sample = 0; sample < samples; sample += 1) {
        const sampleIndex = rng.int(samples);
        const x = a[sampleIndex];
        const y = b[sampleIndex];
        if (x === undefined || y === undefined) {
          throw new Error("sample index out of range");
        }
        left[offset] = x;
        right[offset] = y;
        offset += 1;
      }
    }
    draws[replicate] = median(left) / median(right);
  }
  const sorted = sortedCopy(draws);
  return {
    p50x: median(flatten(numerator)) / median(flatten(denominator)),
    p50x_ci: [percentileSorted(sorted, 2.5), percentileSorted(sorted, 97.5)],
  };
}
