// Shared site content: the homepage and the inner pages render from the same records.

export interface Experiment {
	month: string;
	year: string;
	title: string;
	excerpt: string;
	/** A phrase inside `excerpt` that is set in the accent colour. */
	highlight: string;
	href: string;
}

export const experiments: Experiment[] = [
	{
		month: 'Sep',
		year: '2026',
		title: 'Faster Tokens. Same Small Model.',
		excerpt:
			'MONARCH: optimizing WebGPU kernels for LFM2.5-230M, measuring 1,574 tokens/s on an M4 Max, and testing the limits of a persistent decoder. Try the real model on your GPU.',
		highlight: '1,574 tokens/s',
		href: '/experiments/monarch',
	},
	{
		month: 'Jul',
		year: '2026',
		title: 'I Taught a 4B Model My Taste in Names',
		excerpt:
			'Personalizing Qwen3.5-4B with 100 GRPO steps on a MacBook using MLXRL, a learned preference scorer, and guardrails against reward hacking.',
		highlight: '100 GRPO steps',
		href: 'https://x.com/inductive_ml/status/2075995191824331044',
	},
	{
		month: 'Apr',
		year: '2026',
		title: 'Teaching Trees to Understand Geometry',
		excerpt:
			'Decision trees waste capacity on curved or rotated geometry. We put a small network in front of XGBoost that learns to straighten the data first — 601 leaves down to 28.',
		highlight: '601 leaves down to 28',
		href: '/staged-learned-coordinates',
	},
];

export interface ProfileLink {
	k: string;
	v: string;
	href: string;
}

export const profile: ProfileLink[] = [
	{ k: 'Email', v: 'aditya@inductive.ml', href: 'mailto:aditya@inductive.ml' },
	{ k: 'X', v: 'x.com/inductive_ml', href: 'https://x.com/inductive_ml' },
	{ k: 'GitHub', v: 'github.com/inductiveML', href: 'https://github.com/inductiveML' },
	{ k: 'Hugging Face', v: 'hf.co/inductiveML', href: 'https://hf.co/inductiveML' },
	{
		k: 'OpenReview',
		v: '~Aditya_Veer_Parmar1',
		href: 'https://openreview.net/profile?id=~Aditya_Veer_Parmar1',
	},
	{ k: 'ORCID', v: '0009-0000-3060-4372', href: 'https://orcid.org/0009-0000-3060-4372' },
];

export interface AuditCheck {
	k: string;
	v: string;
}

export const auditChecks: AuditCheck[] = [
	{
		k: 'Answer leakage',
		v: 'The reference solution, or something that points to it, is left where the agent can read it: version-control history, file timestamps, caches, build artefacts, backup files.',
	},
	{
		k: 'Grader tampering',
		v: 'The agent can pass by changing the tests, the test runner or the grading harness instead of solving the task.',
	},
	{
		k: 'Sandbox egress',
		v: 'The agent can reach the network or shared services it should not, and fetch the answer or affect the outside world.',
	},
	{
		k: 'Impossible tasks',
		v: 'No correct solution can pass, so the only reward available comes from exploiting the environment.',
	},
	{
		k: 'Solvability',
		v: 'A correct solution passes the grader, and doing nothing does not.',
	},
	{
		k: 'Pass-rate band',
		v: 'The suite is neither trivially easy nor never solved, so it falls in the pass-rate band labs buy.',
	},
];

export interface Audit {
	label: string;
	title: string;
	note: string;
	/** A phrase inside `note` that is set in the accent colour, or null for none. */
	highlight: string | null;
}

export const audits: Audit[] = [
	{
		label: 'Audit',
		title: 'MiMo RL environments (Xiaomi)',
		note: 'Write-up coming soon.',
		highlight: null,
	},
	{
		label: 'Disclosure',
		title: 'SWE Atlas (Scale AI)',
		note: 'Coordinated disclosure in progress, report on 4 Nov 2026.',
		highlight: '4 Nov 2026',
	},
];

export interface HighlightParts {
	before: string;
	mark: string;
	after: string;
}

/** Splits `text` around `mark`; a mark that is not in the text is a content error. */
export function splitHighlight(text: string, mark: string): HighlightParts {
	const at = text.indexOf(mark);
	if (at === -1) {
		throw new Error(`Highlight "${mark}" does not occur in "${text}"`);
	}
	return { before: text.slice(0, at), mark, after: text.slice(at + mark.length) };
}

export function isExternal(href: string): boolean {
	return href.startsWith('http');
}
