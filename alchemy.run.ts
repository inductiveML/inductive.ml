import alchemy from "alchemy";
import { Astro } from "alchemy/cloudflare";

const app = await alchemy("inductive-ml");

export const worker = await Astro("website", {
  entrypoint: "dist/server/entry.mjs",
  assets: "dist/client",
  domains: ["inductive.ml"]
});

console.log({
  url: worker.url,
});

await app.finalize();
