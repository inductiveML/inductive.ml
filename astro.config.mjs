import alchemy from 'alchemy/cloudflare/astro';
import react from '@astrojs/react';
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  site: 'https://inductive.ml',
  output: 'server',
  redirects: {
    '/cupel/mimo-draft': { status: 301, destination: '/cupel/mimo' },
  },
  adapter: alchemy(),
  integrations: [react()],
  // Posts are plain text and SVG; code blocks take the paper styling, not a highlighter theme.
  markdown: {
    syntaxHighlight: false,
  },
  vite: {
    server: {
      headers: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Resource-Policy': 'same-origin',
      },
    },
  },
});
