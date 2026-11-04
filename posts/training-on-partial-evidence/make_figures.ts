// Draws the figures for "Training on Partial Evidence" from the Abacus repo's result
// files and inlines each SVG into the post between its markers.
//
//   bun run posts/training-on-partial-evidence/make_figures.ts --abacus-root <abacus checkout at 85a5cf8>
//
// Colours are the set validated on the site's paper surface (#f3eee4) with the dataviz
// palette validator. Amber and aqua are under 3:1 contrast there, so every series also
// has a legend entry and a direct end label.

import {readFile, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {z} from 'zod';

const {values} = parseArgs({options: {'abacus-root': {type: 'string'}}, strict: true});
const abacusRoot = z.object({'abacus-root': z.string().min(1)}).parse(values)['abacus-root'];
const postPath = resolve(import.meta.dir, '../training-on-partial-evidence.md');

async function readJson<T>(path: string, schema: z.ZodType<T>): Promise<T> {
	return schema.parse(JSON.parse(await readFile(resolve(abacusRoot, path), 'utf8')));
}

const Interval = z.tuple([z.number(), z.number()]);
const Contrast = z.object({E: z.number(), CI_two_way: Interval});
const DropRaw = z.object({
	partA: z.object({
		E_vs_MASK0: z.object({
			MASK25: Contrast,
			MASK50: Contrast,
			MASK75: Contrast,
			MASK85: Contrast,
			MASK95: Contrast,
			MASK100: Contrast,
			MASK100_ALL: Contrast,
			DROP50: Contrast,
			DROP75: Contrast,
		}),
	}),
});
const RawArm = z.object({HO_ACC1_64: z.number()});
const RawBudget = z.object({
	summary: z.object({
		'RAW-NP-MASK0': RawArm,
		'RAW-NP-ALL75': RawArm,
		'RAW-NP-ALL100': RawArm,
		'RAW-NP-DROP75': RawArm,
	}),
});
const Scale = z.object({raw: z.object({'10000': RawBudget, '20000': RawBudget, '40000': RawBudget})});
const NS = ['1', '2', '4', '8', '16', '32'] as const;
const Union = z.object({'1': z.number(), '2': z.number(), '4': z.number(), '8': z.number(), '16': z.number(), '32': z.number()});
const Cover = z.object({analysis: z.object({verdict: z.object({UNION: z.object({'8': Union, '64': Union})})})});
const Rates = z.array(z.number().min(0).max(1)).length(24);
const SeedSplit = z.object({targets: z.array(z.number().int()).length(24), rates_new: z.record(z.string(), Rates)});

const FONT = 'Archivo, system-ui, sans-serif';
const MONO = 'IBM Plex Mono, ui-monospace, monospace';
const PAPER = '#f3eee4';
const BLUE = '#2a78d6';
const AMBER = '#eda100';
const AQUA = '#1baf7a';
const CRIMSON = '#b8381f';
const f1 = (n: number): string => n.toFixed(1);

function svg(w: number, h: number, label: string, body: string[]): string {
	return [`<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="${label}" style="max-width:640px;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg">`, ...body, '</svg>'].join('\n');
}
function text(x: number, y: number, s: string, opts: {size: number; anchor?: 'start' | 'middle' | 'end'; muted?: boolean; bold?: boolean; mono?: boolean; fill?: string}): string {
	const attrs = [
		`x="${f1(x)}"`,
		`y="${f1(y)}"`,
		`fill="${opts.fill ?? 'currentColor'}"`,
		`font-family="${opts.mono ? MONO : FONT}"`,
		`font-size="${opts.size}"`,
		opts.anchor ? `text-anchor="${opts.anchor}"` : '',
		opts.muted ? 'opacity="0.75"' : '',
		opts.bold ? 'font-weight="600"' : '',
	];
	return `<text ${attrs.filter(Boolean).join(' ')}>${s}</text>`;
}

// ------------------------------------------------------------------ the world
const OPS = ['NOP', 'READ', 'WRITE', 'SWAP', 'INC', 'MOVR', 'MOVL'] as const;
type Op = (typeof OPS)[number];

/** The Abacus K7 machine (abacus_b1/vendor/vm_hard.py): the 6 slots run once per tape cell. */
function run(program: Op[], input: number[]): number[] {
	const tape = [...input];
	const n = tape.length;
	let pointer = 0;
	let register = 0;
	for (let pass = 0; pass < n; pass++) {
		for (const op of program) {
			if (op === 'READ') register = tape[pointer];
			else if (op === 'WRITE') tape[pointer] = register;
			else if (op === 'SWAP') [tape[pointer], register] = [register, tape[pointer]];
			else if (op === 'INC') register = (register + 1) % 8;
			else if (op === 'MOVR') pointer = (pointer + 1) % n;
			else if (op === 'MOVL') pointer = (pointer - 1 + n) % n;
		}
	}
	return tape;
}

function worldFigure(): string {
	// Target 1878: the only program for this function, and a held-out target in every report.
	const program: Op[] = ['SWAP', 'MOVR', 'SWAP', 'MOVL', 'MOVL', 'MOVL'];
	const input = [3, 4, 6, 7, 0, 1, 6, 7];
	const output = run(program, input);
	// The Python interpreter's output for this program and input.
	if (output.join() !== [7, 0, 0, 1, 6, 7, 3, 4].join()) throw new Error(`interpreter mismatch: ${output.join()}`);
	if (output[2] !== input[4] || output[5] !== input[7]) throw new Error('the held-out reads do not hold');

	const parts: string[] = [];
	parts.push(text(12, 16, 'program: 6 slots, 7 possible operations', {size: 12, muted: true}));
	const slotW = 58;
	program.forEach((op, i) => {
		const x = 12 + i * (slotW + 5);
		parts.push(`<rect x="${x}" y="24" width="${slotW}" height="26" fill="none" stroke="currentColor" stroke-opacity="0.6"/>`);
		parts.push(text(x + slotW / 2, 41, op, {size: 12, anchor: 'middle', mono: true}));
	});
	parts.push(text(12, 76, 'one example: input tape and output tape', {size: 12, muted: true}));
	const cell = 36;
	const left = 56;
	const yIn = 92;
	const yOut = 176;
	input.forEach((value, i) => {
		const x = left + i * (cell + 5);
		const hot = i === 4 || i === 7;
		parts.push(text(x + cell / 2, yIn - 3, String(i), {size: 10, anchor: 'middle', muted: true}));
		parts.push(`<rect x="${x}" y="${yIn}" width="${cell}" height="26" fill="${hot ? CRIMSON : 'none'}" fill-opacity="${hot ? 0.14 : 0}" stroke="${hot ? CRIMSON : 'currentColor'}" stroke-opacity="${hot ? 1 : 0.5}"/>`);
		parts.push(text(x + cell / 2, yIn + 18, String(value), {size: 13, anchor: 'middle', mono: true}));
	});
	output.forEach((value, i) => {
		const x = left + i * (cell + 5);
		const hot = i === 2 || i === 5;
		parts.push(`<rect x="${x}" y="${yOut}" width="${cell}" height="26" fill="${hot ? CRIMSON : 'none'}" fill-opacity="${hot ? 0.14 : 0}" stroke="${hot ? CRIMSON : 'currentColor'}" stroke-opacity="${hot ? 1 : 0.5}"/>`);
		parts.push(text(x + cell / 2, yOut + 18, String(value), {size: 13, anchor: 'middle', mono: true}));
		parts.push(text(x + cell / 2, yOut + 38, String(i), {size: 10, anchor: 'middle', muted: true}));
	});
	parts.push(text(12, yIn + 18, 'input', {size: 12}));
	parts.push(text(12, yOut + 18, 'output', {size: 12}));
	const centre = (i: number): number => left + i * (cell + 5) + cell / 2;
	for (const [from, to] of [[4, 2], [7, 5]]) {
		const x1 = centre(from);
		const y1 = yIn + 28;
		const x2 = centre(to);
		const y2 = yOut - 3;
		const angle = Math.atan2(y2 - y1, x2 - x1);
		const head = (turn: number): string => `${f1(x2 - 9 * Math.cos(angle + turn))} ${f1(y2 - 9 * Math.sin(angle + turn))}`;
		parts.push(`<path d="M${x1} ${y1} L${x2} ${y2}" stroke="${CRIMSON}" stroke-width="2" fill="none"/>`);
		parts.push(`<path d="M${head(0.5)} L${x2} ${y2} L${head(-0.5)}" stroke="${CRIMSON}" stroke-width="2" fill="none"/>`);
	}
	return svg(400, 226, 'A six-slot program, SWAP MOVR SWAP MOVL MOVL MOVL, turns an eight-cell input tape into an output tape. Output cell 2 reads input cell 4, and output cell 5 reads input cell 7.', parts);
}

// ------------------------------------------------------------------ the split
function splitFigure(): string {
	// EXPECTED['A'] in abacus_b5/split.py: 25 held-out families, 603 in training, 9 and 17 with one read alone.
	const parts: string[] = [];
	const cols = [150, 278];
	const rows = [44, 118];
	const w = 122;
	const h = 68;
	parts.push(text(cols[0] + w / 2, 18, 'cell 5 reads', {size: 11, anchor: 'middle', muted: true}));
	parts.push(text(cols[0] + w / 2, 33, 'another cell', {size: 11, anchor: 'middle', muted: true}));
	parts.push(text(cols[1] + w / 2, 18, 'cell 5 reads', {size: 11, anchor: 'middle', muted: true}));
	parts.push(text(cols[1] + w / 2, 33, 'input cell 7', {size: 11, anchor: 'middle', muted: true}));
	parts.push(text(12, rows[0] + 30, 'cell 2 reads', {size: 11, muted: true}));
	parts.push(text(12, rows[0] + 45, 'another cell', {size: 11, muted: true}));
	parts.push(text(12, rows[1] + 30, 'cell 2 reads', {size: 11, muted: true}));
	parts.push(text(12, rows[1] + 45, 'input cell 4', {size: 11, muted: true}));
	const cells: {c: number; r: number; n: string; note: string; held: boolean}[] = [
		{c: 0, r: 0, n: '577', note: 'in training', held: false},
		{c: 1, r: 0, n: '17', note: 'in training', held: false},
		{c: 0, r: 1, n: '9', note: 'in training', held: false},
		{c: 1, r: 1, n: '25', note: 'held out · 24 tested', held: true},
	];
	for (const cell of cells) {
		const x = cols[cell.c];
		const y = rows[cell.r];
		parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${cell.held ? CRIMSON : 'currentColor'}" fill-opacity="${cell.held ? 0.14 : 0.07}" stroke="${cell.held ? CRIMSON : 'currentColor'}" stroke-opacity="${cell.held ? 1 : 0.35}"/>`);
		parts.push(text(x + w / 2, y + 32, cell.n, {size: 22, anchor: 'middle', bold: true}));
		parts.push(text(x + w / 2, y + 52, cell.note, {size: 11, anchor: 'middle', muted: !cell.held}));
	}
	return svg(400, 196, 'A two by two grid of function families. 577 have neither read, 17 have only the read of input 7 into cell 5, 9 have only the read of input 4 into cell 2, all in training. 25 have both, are held out, and 24 are tested.', parts);
}

// ------------------------------------------------------------------ writer × target grid
function gridFigure(split: z.infer<typeof SeedSplit>): string {
	const order = [60000, 60001, 60002].flatMap((i) => [60000, 60001, 60002].map((d) => `A_PLAIN_i${i}_d${d}`));
	const rows = order.map((name) => {
		const rates = split.rates_new[name];
		if (rates === undefined) throw new Error(`missing writer ${name}`);
		return rates;
	});
	const counts = rows.map((rates) => rates.filter((r) => r >= 0.5).length);
	if (Math.min(...counts) !== 5 || Math.max(...counts) !== 8) throw new Error(`writer counts changed: ${counts.join()}`);
	// Most-solved targets on the left, so the disagreement between writers is easy to see.
	const columns = split.targets.map((_, t) => t).sort((a, b) => rows.reduce((s, r) => s + r[b], 0) - rows.reduce((s, r) => s + r[a], 0));
	const parts: string[] = [];
	const left = 62;
	const cell = 13.5;
	const top = 26;
	parts.push(text(left, 14, '24 held-out targets, most solved on the left', {size: 11, muted: true}));
	rows.forEach((rates, r) => {
		const y = top + r * (cell + 1) + Math.floor(r / 3) * 8;
		if (r % 3 === 0) parts.push(text(12, y + cell + 9, `weights ${r / 3 + 1}`, {size: 11, muted: true}));
		columns.forEach((t, c) => {
			const x = left + c * (cell + 0.5);
			const rate = rates[t];
			const rateText = rate.toFixed(2);
			parts.push(`<rect x="${f1(x)}" y="${f1(y)}" width="${cell}" height="${cell}" fill="${CRIMSON}" fill-opacity="${rate === 0 ? 0 : (0.12 + 0.88 * rate).toFixed(2)}" stroke="currentColor" stroke-opacity="0.14"><title>target ${split.targets[t]}: ${rateText}</title></rect>`);
		});
	});
	const legendY = top + 9 * (cell + 1) + 2 * 8 + 12;
	parts.push(text(left, legendY + 9, '0', {size: 10, muted: true}));
	[0.25, 0.5, 0.75, 1].forEach((rate, i) => {
		parts.push(`<rect x="${left + 12 + i * 15}" y="${legendY}" width="14" height="10" fill="${CRIMSON}" fill-opacity="${(0.12 + 0.88 * rate).toFixed(2)}"/>`);
	});
	parts.push(text(left + 12 + 4 * 15 + 3, legendY + 9, 'share of fresh example sets solved', {size: 10, muted: true}));
	return svg(400, legendY + 16, 'A grid of nine writers against 24 held-out targets. Each writer solves a different subset, and writers sharing starting weights look more alike.', parts);
}

// ------------------------------------------------------------------ shared line-chart frame
interface Series {
	legend: string;
	color: string;
	dash: string;
	points: {x: number; y: number}[];
	end: string;
}
function lineChart(opts: {height: number; xs: {at: number; label: string}[]; xLabel: string; yMin: number; yMax: number; yTicks: number[]; yFormat: (v: number) => string; series: Series[]; label: string}): string {
	const W = 400;
	const legendRow = 16;
	const plotTop = 12 + opts.series.length * legendRow + 8;
	const plotBottom = opts.height - 38;
	const plotLeft = 36;
	const plotRight = W - 44;
	const xMin = opts.xs[0].at;
	const xMax = opts.xs[opts.xs.length - 1].at;
	const X = (x: number): number => plotLeft + ((x - xMin) / (xMax - xMin)) * (plotRight - plotLeft);
	const Y = (y: number): number => plotBottom - ((y - opts.yMin) / (opts.yMax - opts.yMin)) * (plotBottom - plotTop);
	const parts: string[] = [];
	for (const tick of opts.yTicks) {
		parts.push(`<path d="M${plotLeft} ${f1(Y(tick))} H${plotRight}" stroke="currentColor" stroke-opacity="${tick === opts.yMin ? 0.6 : 0.15}"/>`);
		parts.push(text(plotLeft - 6, Y(tick) + 4, opts.yFormat(tick), {size: 11, anchor: 'end', muted: true}));
	}
	for (const x of opts.xs) parts.push(text(X(x.at), plotBottom + 15, x.label, {size: 11, anchor: 'middle', muted: true}));
	parts.push(text((plotLeft + plotRight) / 2, plotBottom + 31, opts.xLabel, {size: 11, anchor: 'middle', muted: true}));
	opts.series.forEach((s, i) => {
		const y = 12 + i * legendRow;
		parts.push(`<path d="M${plotLeft} ${y} H${plotLeft + 22}" stroke="${s.color}" stroke-width="2" stroke-dasharray="${s.dash}"/>`);
		parts.push(`<circle cx="${plotLeft + 11}" cy="${y}" r="4" fill="${s.color}" stroke="${PAPER}" stroke-width="2"/>`);
		parts.push(text(plotLeft + 30, y + 4, s.legend, {size: 12}));
	});
	for (const s of opts.series) {
		const d = s.points.map((p, i) => `${i === 0 ? 'M' : 'L'}${f1(X(p.x))} ${f1(Y(p.y))}`).join(' ');
		parts.push(`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-dasharray="${s.dash}" stroke-linejoin="round"/>`);
	}
	for (const s of opts.series) {
		for (const p of s.points) parts.push(`<circle cx="${f1(X(p.x))}" cy="${f1(Y(p.y))}" r="4" fill="${s.color}" stroke="${PAPER}" stroke-width="2"/>`);
	}
	const ends = opts.series.map((s) => ({y: Y(s.points[s.points.length - 1].y), t: s.end})).sort((a, b) => a.y - b.y);
	for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
	for (const end of ends) parts.push(text(plotRight + 8, end.y + 4, end.t, {size: 11}));
	return svg(W, opts.height, opts.label, parts);
}

function poolFigure(cover: z.infer<typeof Cover>): string {
	const {verdict} = cover.analysis;
	const pick = (u: z.infer<typeof Union>): {x: number; y: number}[] => NS.map((n) => ({x: Math.log2(Number(n)), y: u[n]}));
	return lineChart({
		height: 250,
		xs: NS.map((n) => ({at: Math.log2(Number(n)), label: n})),
		xLabel: 'writers in the pool',
		yMin: 0,
		yMax: 24,
		yTicks: [0, 6, 12, 18, 24],
		yFormat: (v) => String(v),
		series: [
			{legend: '64 guesses per writer', color: CRIMSON, dash: 'none', points: pick(verdict.UNION['64']), end: verdict.UNION['64']['32'].toFixed(1)},
			{legend: '8 guesses per writer', color: BLUE, dash: 'none', points: pick(verdict.UNION['8']), end: verdict.UNION['8']['32'].toFixed(1)},
		],
		label: `Targets covered grow with pool size, reaching ${verdict.UNION['64']['32']} of 24 at 64 guesses and ${verdict.UNION['8']['32']} at 8 guesses with 32 writers.`,
	});
}

function rawFigure(scale: z.infer<typeof Scale>): string {
	const budgets = ['10000', '20000', '40000'] as const;
	const pick = (arm: 'RAW-NP-MASK0' | 'RAW-NP-ALL75' | 'RAW-NP-ALL100' | 'RAW-NP-DROP75'): {x: number; y: number}[] =>
		budgets.map((b, i) => ({x: i, y: scale.raw[b].summary[arm].HO_ACC1_64}));
	const end = (points: {y: number}[]): string => `${(points[points.length - 1].y * 100).toFixed(1)}%`;
	const full = pick('RAW-NP-MASK0');
	const all75 = pick('RAW-NP-ALL75');
	const all100 = pick('RAW-NP-ALL100');
	const drop = pick('RAW-NP-DROP75');
	return lineChart({
		height: 290,
		xs: [{at: 0, label: '10k'}, {at: 1, label: '20k'}, {at: 2, label: '40k'}],
		xLabel: 'training steps',
		yMin: 0.4,
		yMax: 0.8,
		yTicks: [0.4, 0.5, 0.6, 0.7, 0.8],
		yFormat: (v) => `${Math.round(v * 100)}%`,
		series: [
			{legend: 'All examples', color: AMBER, dash: '5 3', points: full, end: end(full)},
			{legend: 'Hide on 75% of rows, every length', color: AQUA, dash: 'none', points: all75, end: end(all75)},
			{legend: 'Hide on all rows, every length', color: BLUE, dash: 'none', points: all100, end: end(all100)},
			{legend: 'Drop 75% of rows', color: CRIMSON, dash: 'none', points: drop, end: end(drop)},
		],
		label: 'Raw-tape writer held-out accuracy at 10k, 20k and 40k steps. All examples stays near 59%. Hiding starts near 70% and falls slightly. Dropping rows rises from 51% to 71%.',
	});
}

// ------------------------------------------------------------------ what a row keeps
function masksFigure(): string {
	const panels: {name: string[]; keep: Record<number, number>; kind: 'full' | 'short' | 'all'}[] = [
		{name: ['all 40', 'at test'], keep: {4: 8, 5: 8, 6: 8, 7: 8, 8: 8}, kind: 'full'},
		{name: ['8 tapes,', 'length 4'], keep: {4: 8}, kind: 'short'},
		{name: ['1 tape,', 'lengths 4–5'], keep: {4: 1, 5: 1}, kind: 'short'},
		{name: ['1 tape,', 'length 4'], keep: {4: 1}, kind: 'short'},
		{name: ['4 tapes,', 'every length'], keep: {4: 4, 5: 4, 6: 4, 7: 4, 8: 4}, kind: 'all'},
		{name: ['1 tape,', 'every length'], keep: {4: 1, 5: 1, 6: 1, 7: 1, 8: 1}, kind: 'all'},
	];
	const parts: string[] = [];
	const panelW = 58;
	const gap = 7;
	const cw = 10;
	const ch = 8;
	const top = 50;
	panels.forEach((panel, p) => {
		const x0 = 8 + p * (panelW + gap);
		parts.push(text(x0 + panelW / 2 - 5, 14, panel.name[0], {size: 10, anchor: 'middle'}));
		parts.push(text(x0 + panelW / 2 - 5, 27, panel.name[1], {size: 10, anchor: 'middle', muted: true}));
		const color = panel.kind === 'all' ? CRIMSON : panel.kind === 'short' ? BLUE : 'currentColor';
		[4, 5, 6, 7, 8].forEach((length, c) => {
			parts.push(text(x0 + c * (cw + 1) + cw / 2, top - 6, String(length), {size: 8.5, anchor: 'middle', muted: true}));
			for (let t = 0; t < 8; t++) {
				const kept = t < (panel.keep[length] ?? 0);
				parts.push(`<rect x="${f1(x0 + c * (cw + 1))}" y="${f1(top + t * (ch + 1))}" width="${cw}" height="${ch}" fill="${kept ? color : 'none'}" fill-opacity="${kept ? 0.9 : 0}" stroke="currentColor" stroke-opacity="${kept ? 0 : 0.2}"/>`);
			}
		});
	});
	const y = top + 8 * (ch + 1) + 14;
	const label = (i: number, count: number): number => 8 + (i * (panelW + gap) + (i + count - 1) * (panelW + gap) + panelW - 5) / 2;
	parts.push(text(label(0, 1), y, 'full', {size: 11, anchor: 'middle', muted: true}));
	parts.push(text(label(1, 3), y, 'short lengths only', {size: 11, anchor: 'middle', muted: true}));
	parts.push(text(label(4, 2), y, 'every length', {size: 11, anchor: 'middle', muted: true}));
	return svg(400, y + 8, 'Six grids of five tape lengths by eight tapes showing which tapes a training row keeps: all 40, three short-length patterns, and two every-length patterns.', parts);
}

// ------------------------------------------------------------------ gains with intervals
function forestFigure(dropRaw: z.infer<typeof DropRaw>): string {
	const e = dropRaw.partA.E_vs_MASK0;
	const rows: {name: string; c: z.infer<typeof Contrast>; color: string}[] = [
		{name: 'Hide 25%, short', c: e.MASK25, color: BLUE},
		{name: 'Hide 50%, short', c: e.MASK50, color: BLUE},
		{name: 'Hide 75%, short', c: e.MASK75, color: BLUE},
		{name: 'Hide 85%, short', c: e.MASK85, color: BLUE},
		{name: 'Hide 95%, short', c: e.MASK95, color: BLUE},
		{name: 'Hide 100%, short', c: e.MASK100, color: BLUE},
		{name: 'Hide 100%, every length', c: e.MASK100_ALL, color: CRIMSON},
		{name: 'Drop 50% of rows', c: e.DROP50, color: AMBER},
		{name: 'Drop 75% of rows', c: e.DROP75, color: AMBER},
	];
	const left = 144;
	const right = 356;
	const top = 24;
	const rowH = 22;
	const lo = -50;
	const hi = 35;
	const X = (points: number): number => left + ((points - lo) / (hi - lo)) * (right - left);
	const bottom = top + rows.length * rowH;
	const parts: string[] = [];
	for (const tick of [-50, -25, 0, 25]) {
		parts.push(`<path d="M${f1(X(tick))} ${top - 4} V${bottom}" stroke="currentColor" stroke-opacity="${tick === 0 ? 0.7 : 0.15}"/>`);
		parts.push(text(X(tick), bottom + 15, tick > 0 ? `+${tick}` : tick < 0 ? `−${-tick}` : '0', {size: 11, anchor: 'middle', muted: true}));
	}
	parts.push(text((left + right) / 2, bottom + 31, 'points, against training on all examples', {size: 11, anchor: 'middle', muted: true}));
	rows.forEach((row, i) => {
		const y = top + i * rowH + rowH / 2;
		const [a, b] = row.c.CI_two_way;
		const clear = a > 0 || b < 0;
		const gain = row.c.E * 100;
		parts.push(text(12, y + 4, row.name, {size: 11.5}));
		parts.push(`<path d="M${f1(X(a * 100))} ${y} H${f1(X(b * 100))}" stroke="${row.color}" stroke-width="2"/>`);
		parts.push(`<circle cx="${f1(X(gain))}" cy="${y}" r="4.5" fill="${clear ? row.color : PAPER}" stroke="${row.color}" stroke-width="2"/>`);
		parts.push(text(right + 6, y + 4, `${gain >= 0 ? '+' : '−'}${Math.abs(gain).toFixed(1)}`, {size: 11, mono: true}));
	});
	return svg(400, bottom + 38, 'Gains in held-out accuracy with 95 percent intervals. Clear gains at 75 and 85 percent short-length hiding and at 100 percent every-length hiding, a clear loss at 100 percent short-length hiding, and unclear results elsewhere.', parts);
}

// ------------------------------------------------------------------ write into the post
const dropRaw = await readJson('experiments/abacus_dropraw0/rows.json', DropRaw);
const scale = await readJson('experiments/abacus_scale0/rows.json', Scale);
const cover = await readJson('experiments/abacus_cover0/rows.json', Cover);
const seedSplit = await readJson('experiments/abacus_seedsplit0/rows.json', SeedSplit);

const figures: Record<string, string> = {
	world: worldFigure(),
	split: splitFigure(),
	grid: gridFigure(seedSplit),
	pool: poolFigure(cover),
	masks: masksFigure(),
	forest: forestFigure(dropRaw),
	raw: rawFigure(scale),
};

let post = await readFile(postPath, 'utf8');
for (const [name, figure] of Object.entries(figures)) {
	const start = `<!-- fig:${name}:start (generated by posts/training-on-partial-evidence/make_figures.ts) -->`;
	const end = `<!-- fig:${name}:end -->`;
	const a = post.indexOf(start);
	const b = post.indexOf(end);
	if (a === -1 || b === -1 || b < a) throw new Error(`markers not found for ${name}`);
	post = `${post.slice(0, a + start.length)}\n${figure}\n${post.slice(b)}`;
}
await writeFile(postPath, post);
console.log(`inlined ${Object.keys(figures).length} figures into ${postPath}`);
