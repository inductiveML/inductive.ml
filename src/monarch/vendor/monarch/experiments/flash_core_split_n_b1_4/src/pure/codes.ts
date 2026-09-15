// The one encoder from a registered name to its published integer code, and back.
//
// results.json carries codes, not names, so every enumeration in the run crosses this boundary
// exactly once. Two encoders would be two chances to publish a dispatch kind under one code and
// adjudicate it under another, so node, page and report all come through here.
import { CODES } from "../config.ts";

export type CodeTable = Readonly<Record<string, number>>;

// Names are written lower_snake in the plan and SCREAMING_SNAKE in the code tables, which is the
// only transformation applied: a name with no entry is an error, never a zero.
export function codeOf(table: CodeTable, key: string, label: string): number {
  const code = table[key.toUpperCase()];
  if (code === undefined) {
    throw new Error(`${label}: no code for ${key}`);
  }
  return code;
}

// The inverse, for the prose report. A code with no name is an error for the same reason: a table
// that gained an entry the renderer does not know about must not print as a bare integer.
export function nameOf(table: CodeTable, code: number, label: string): string {
  const entry = Object.entries(table).find(([, value]) => value === code);
  if (entry === undefined) {
    throw new Error(`${label}: no name for code ${code}`);
  }
  return entry[0];
}

export function attnPartCode(part: string): number {
  return codeOf(CODES.attnPart, part, "ATTN_PART");
}
