// Checks every row of numbers_check.md: the post must contain the "In post" text,
// and the named source file must contain the "Source text". `web:` and `none`
// rows are checked on the post side only (their source side was checked by hand).
//
//   bun run posts/sokoban-without-hints/check_numbers.ts \
//     --no-hints-root <inductiveML/sokoban-no-hints checkout> \
//     --old-root <avp1598/sokoban checkout at e189e5b>

import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {z} from 'zod';

const ArgsSchema = z.object({
	'no-hints-root': z.string().min(1),
	'old-root': z.string().min(1),
});

const {values} = parseArgs({
	options: {
		'no-hints-root': {type: 'string'},
		'old-root': {type: 'string'},
	},
	strict: true,
});
const args = ArgsSchema.parse(values);

const here = import.meta.dir;
const siteRoot = resolve(here, '../..');
const postPath = resolve(here, '../sokoban-without-hints.md');
const checkPath = resolve(here, 'numbers_check.md');

const RowSchema = z.object({
	id: z.string().regex(/^\d+[a-z]?$/),
	inPost: z.string().min(1),
	source: z.string().regex(/^(no-hints:|old:|site:|web:https:\/\/|none$)/),
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
	if (source.startsWith('no-hints:')) return resolve(args['no-hints-root'], source.slice('no-hints:'.length));
	if (source.startsWith('old:')) return resolve(args['old-root'], source.slice('old:'.length));
	if (source.startsWith('site:')) return resolve(siteRoot, source.slice('site:'.length));
	return null;
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
