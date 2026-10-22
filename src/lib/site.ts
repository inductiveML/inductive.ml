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

/** The write-up of the GRPO personalisation run, the first public use of MLXRL. */
const namesExperimentHref = 'https://x.com/inductive_ml/status/2075995191824331044';

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
		href: namesExperimentHref,
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

/** The lab's code and weights live under these two organisations. */
export const github: ProfileLink = {
	k: 'GitHub',
	v: 'github.com/inductiveML',
	href: 'https://github.com/inductiveML',
};
export const huggingFace: ProfileLink = {
	k: 'Hugging Face',
	v: 'hf.co/inductiveML',
	href: 'https://huggingface.co/inductiveML',
};

export const profile: ProfileLink[] = [
	{ k: 'Email', v: 'aditya@inductive.ml', href: 'mailto:aditya@inductive.ml' },
	{ k: 'X', v: 'x.com/inductive_ml', href: 'https://x.com/inductive_ml' },
	github,
	huggingFace,
	{
		k: 'OpenReview',
		v: '~Aditya_Veer_Parmar1',
		href: 'https://openreview.net/profile?id=~Aditya_Veer_Parmar1',
	},
	{ k: 'ORCID', v: '0009-0000-3060-4372', href: 'https://orcid.org/0009-0000-3060-4372' },
];

export type ReleaseKind = 'Library' | 'Model' | 'Code';

export interface Release {
	kind: ReleaseKind;
	name: string;
	note: string;
	/** A phrase inside `note` that is set in the accent colour, or null for none. */
	highlight: string | null;
	host: ProfileLink;
	href: string;
}

export interface FeaturedRelease extends Release {
	highlight: string;
	/** An experiment that used the release. */
	usedIn: string;
}

/** MLXRL, the lab's flagship library; the home page features it above the other releases. */
export const mlxrl: FeaturedRelease = {
	kind: 'Library',
	name: 'MLXRL',
	note: 'Fast on-policy RL post-training for LLMs on Apple Silicon. GRPO, Dr. GRPO, DAPO, GSPO, RLOO and multi-turn GiGPO with QLoRA on local 4-bit MLX models, in one process with one model object.',
	highlight: 'multi-turn GiGPO',
	host: github,
	href: 'https://github.com/inductiveML/mlxrl',
	usedIn: namesExperimentHref,
};

export const releases: Release[] = [
	mlxrl,
	{
		kind: 'Library',
		name: 'Ternel',
		note: 'Lossless execution formats for ternary models: 128 weights packed into 28 bytes and run directly by custom Metal kernels.',
		highlight: '28 bytes',
		host: github,
		href: 'https://github.com/inductiveML/ternel',
	},
	{
		kind: 'Model',
		name: 'Ternary Bonsai 2 27B · MLX 1.75 bpw',
		note: 'Lossless repack of Bonsai 2 27B with vision; 1.68 GB less active memory than the official 2-bit checkpoint.',
		highlight: '1.68 GB less',
		host: huggingFace,
		href: 'https://huggingface.co/inductiveML/Ternary-Bonsai-2-27B-mlx-lossless-1.75bpw',
	},
	{
		kind: 'Model',
		name: 'Ternary Bonsai 27B · MLX 1.75 bpw',
		note: 'Bit-exact over all 26.9B ternary weights, with 22.19% less live memory than the official distribution.',
		highlight: '22.19% less',
		host: huggingFace,
		href: 'https://huggingface.co/inductiveML/Ternary-Bonsai-27B-mlx-lossless-1.75bpw',
	},
	{
		kind: 'Model',
		name: 'Qwen3.6-35B-A3B · evolved mixed-bit',
		note: '12.63 GB MLX quant, 16.8% smaller than the TurboQuant 3-bit quant at tied WikiText-2 perplexity.',
		highlight: '16.8% smaller',
		host: huggingFace,
		href: 'https://huggingface.co/inductiveML/Qwen3.6-35B-A3B-evolved-mxbit',
	},
	{
		kind: 'Model',
		name: 'LFM2.5-230M · MONARCH',
		note: 'Packed int4 WebGPU buffer behind the MONARCH browser decoder.',
		highlight: null,
		host: huggingFace,
		href: 'https://huggingface.co/inductiveML/LFM2.5-230M-MONARCH',
	},
	{
		kind: 'Code',
		name: 'Staged Learned Coordinates',
		note: 'Code and paper draft for learned coordinates in front of gradient-boosted trees.',
		highlight: null,
		host: github,
		href: 'https://github.com/inductiveML/staged-learned-coordinates',
	},
	{
		kind: 'Code',
		name: 'Cupel MiMo audit',
		note: 'Reports, pre-registrations, analysis scripts and result tables for the MiMo audit.',
		highlight: null,
		host: github,
		href: 'https://github.com/inductiveML/cupel-mimo-audit',
	},
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
	/** The write-up, when one is published; rows without one use the list's link. */
	href?: string;
}

export const audits: Audit[] = [
	{
		label: 'Audit',
		title: 'MiMo RL environments (Xiaomi)',
		note: 'Fix held 0/300 · timestamp side channel 85.3% · severity Low, efficiency gain only',
		highlight: 'severity Low',
		href: '/cupel/mimo',
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
