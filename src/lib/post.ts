import { z } from 'astro/zod';

// Lengths follow what search results and social cards show before truncating.
const PostFrontmatter = z.object({
	pageTitle: z.string().max(60),
	shareTitle: z.string().max(60),
	description: z.string().min(120).max(160),
	date: z.iso.datetime(),
});

export interface Post extends z.infer<typeof PostFrontmatter> {
	/** The publication date as the page prints it, e.g. "1 Nov 2026". */
	published: string;
}

/** Validates a post's Markdown frontmatter and formats its publication date. */
export function parsePost(frontmatter: Record<string, unknown>): Post {
	const post = PostFrontmatter.parse(frontmatter);
	const published = new Date(post.date).toLocaleDateString('en-GB', {
		day: 'numeric',
		month: 'short',
		year: 'numeric',
		timeZone: 'UTC',
	});
	return { ...post, published };
}
