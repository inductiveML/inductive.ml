import alchemy from 'alchemy/cloudflare/astro';
import react from '@astrojs/react';
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  output: 'server',
  adapter: alchemy(),
  integrations: [react()],
});
