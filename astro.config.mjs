import alchemy from 'alchemy/cloudflare/astro';
import react from '@astrojs/react';
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  output: 'server',
  redirects: {
    '/cupel/mimo-draft': { status: 301, destination: '/cupel/mimo' },
  },
  adapter: alchemy(),
  integrations: [react()],
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
