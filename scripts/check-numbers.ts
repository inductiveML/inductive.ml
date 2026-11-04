// Checks every row of a post's numbers_check.md: the post must contain the "In post" text,
// and the named source file must contain the "Source text". A source is `<root>:<path>`,
// where each root is passed on the command line, or `site:<path>` for this repo. `web:`
// and `none` rows are checked on the post side only (their source side is checked by hand).
//
//   bun run scripts/check-numbers.ts --post posts/<slug>.md --check posts/<slug>/numbers_check.md \
//     --root <name>=<checkout path> [--root <name>=<checkout path> ...]

import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {z} from 'zod';

const ArgsSchema = z.object({
	post: z.string().min(1),
	check: z.string().min(1),
	root: z.array(z.string().regex(/^[a-z0-9-]+=.+$/)).min(1),
});

const {values} = parseArgs({
	options: {
		post: {type: 'string'},
		check: {type: 'string'},
		root: {type: 'string', multiple: true},
	},
	strict: true,
});
const args = ArgsSchema.parse(values);
const roots = new Map(args.root.map((pair) => [pair.slice(0, pair.indexOf('=')), pair.slice(pair.indexOf('=') + 1)]));

const siteRoot = resolve(import.meta.dir, '..');
const postPath = resolve(siteRoot, args.post);
const checkPath = resolve(siteRoot, args.check);

const RowSchema = z.object({
	id: z.string().regex(/^\d+[a-z]?$/),
	inPost: z.string().min(1),
	source: z.string().regex(/^([a-z0-9-]+:[^ ]+|none)$/),
	sourceText: z.string().min(1),
	kind: z.string().regex(/^(exact|rounded|derived|unverified)\b/),
});
type Row = z.infer<typeof RowSchema>;

// Split a Markdown table row on unescaped pipes, then unescape `\|`.
function splitCells(line: string): string[] {
	const cells: string[] = [];
	let current = '';
	for (let i = 0; i < line.length; i++) {
		const char = line[i];
		if (char === '\\' && line[i + 1] === '|') {
			current += '|';
			i++;
		} else if (char === '|') {
			cells.push(current.trim());
			current = '';
		} else {
			current += char;
		}
	}
	cells.push(current.trim());
	// A row starts and ends with a pipe, which leaves an empty first and last cell.
	if (cells[0] !== '' || cells[cells.length - 1] !== '') throw new Error(`malformed table row: ${line}`);
	return cells.slice(1, -1);
}

function parseRows(markdown: string): Row[] {
	const rows: Row[] = [];
	for (const line of markdown.split('\n')) {
		if (!/^\| \d+[a-z]? \|/.test(line)) continue;
		const cells = splitCells(line);
		if (cells.length !== 5) throw new Error(`expected 5 cells, got ${cells.length}: ${line}`);
		rows.push(RowSchema.parse({id: cells[0], inPost: cells[1], source: cells[2], sourceText: cells[3], kind: cells[4]}));
	}
	return rows;
}

function sourcePath(source: string): string | null {
	if (source === 'none' || source.startsWith('web:')) return null;
	const colon = source.indexOf(':');
	const name = source.slice(0, colon);
	const path = source.slice(colon + 1);
	if (name === 'site') return resolve(siteRoot, path);
	const root = roots.get(name);
	if (root === undefined) throw new Error(`no --root given for ${name}`);
	return resolve(root, path);
}

const post = await readFile(postPath, 'utf8');
const rows = parseRows(await readFile(checkPath, 'utf8'));
const ids = new Set<string>();
const failures: string[] = [];
const manual: string[] = [];
const cache = new Map<string, string>();

for (const row of rows) {
	if (ids.has(row.id)) failures.push(`row ${row.id}: duplicate id`);
	ids.add(row.id);
	if (!post.includes(row.inPost)) failures.push(`row ${row.id}: post does not contain ${JSON.stringify(row.inPost)}`);
	const path = sourcePath(row.source);
	if (path === null) {
		manual.push(`row ${row.id}: ${row.source}`);
		continue;
	}
	let text = cache.get(path);
	if (text === undefined) {
		text = await readFile(path, 'utf8');
		cache.set(path, text);
	}
	if (!text.includes(row.sourceText)) failures.push(`row ${row.id}: ${row.source} does not contain ${JSON.stringify(row.sourceText)}`);
}

console.log(`${rows.length} rows; ${rows.length - manual.length} checked against source files; ${manual.length} checked by hand:`);
for (const line of manual) console.log(`  ${line}`);
if (failures.length > 0) {
	console.error(`\n${failures.length} FAILED:`);
	for (const line of failures) console.error(`  ${line}`);
	process.exit(1);
}
console.log('ALL ROWS MATCH');
