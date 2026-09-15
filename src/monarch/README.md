# MONARCH browser demo

The `/experiments/monarch` page runs the B1.8 `MIX_M40` LFM2.5-230M decoder in a dedicated module worker. It uses 72 ordinary dispatches per token. No persistent decoder or atomic visibility protocol is used by the demo.

## Build and assets

Use Bun and run `bun run build`. The build first verifies all model chunks, bundles `worker.ts`, and then builds Astro. Alchemy deployment uses this same command.

The large weight chunks under `public/monarch/weights/` are generated assets excluded from Git. On a fresh checkout, provide the canonical research weight blob using `MONARCH_SOURCE_DIR=/path/to/monarch` (defaults to the sibling `../monarch` repository). The required file within it is `experiments/flash_core_split_n_b1_4/data/weights/weights.bin`. The build rejects a blob whose SHA-256 differs from the pinned manifest. Existing valid chunks can be reused without the research checkout. Keep the manifest, tokenizer, model license, and NOTICE with all distributed model assets.

`source-manifest.json` pins the 28 vendored research source files by SHA-256. Files under `vendor/monarch/` are copied without changes. Runtime allocation and the selected candidate are in `model-config.json`; the browser integration is in `worker.ts`. The worker is built with Bun, without requiring a server inference endpoint. `vendor/tokenizers.mjs` is Hugging Face tokenizers.js 0.1.3, distributed under Apache 2.0.

Both the page and module worker need their configured isolation headers. Production page headers come from `src/middleware.ts`; worker and model asset headers come from `public/_headers`. Vite dev headers are set in `astro.config.mjs`. The experiment page disables the site's analytics script.

## Runtime contract

Before any model download, check WebGPU, shader-f16, fixed 32-lane subgroups, 1,024-thread workgroups, and model buffer limits. Each downloaded or cached chunk and the reconstructed blob are hashed. A known-answer check compares the prefill seed and a 32-token continuation before enabling generation. GPU errors stop the run. Unload terminates the worker, and Stop is checked between bounded token groups or benchmark runs.

Generation is greedy, single-turn ChatML with a 1,024-token prompt limit and up to 512 output tokens. The visible speed counter excludes time to first token, includes streaming overhead, and uses real generated tokens. A short answer can finish partway through the last computed 32-token group.

The quick benchmark warms 64 tokens, then performs four fresh 256-token free-running continuations with 32-token readback through a 64-token ring. All four continuations must match. The selected context is the center of the generated window (start 64 or 896). Report reciprocal median wall time per token and timing spread, excluding model loading and prefill. Downloaded JSON labels this `LOCAL_QUICK_BENCHMARK`.

## Research claims

The article's recorded lab results are separate from the visitor's live measurement. B1.8 retained `NO_VERDICT_SPLIT`; B1.11a's integrated pricing was not adjudicated because environmental/thermal conditions failed. Neither the website nor a quick local benchmark changes those research gates. Public evidence is in `public/monarch/research-results.json`.

## Verification performed

On 15 September 2026, headless Chromium in its normal configuration on the available Apple Silicon adapter passed the bundled known-answer check, generated a real response, and produced matching four-run continuations. Desktop/mobile rendering and an injected adapter lacking shader-f16/subgroups were checked; unsupported devices must make zero weight requests. These are functional checks, not new controlled throughput claims.

## Page controls and rendering

The page reuses `v9.css` and the existing experiment masthead; the lava shader renders a static frame so it does not run beside inference. Nine prompt presets live in `prompts.ts`. `MonarchMarkdown.tsx` renders streamed CommonMark and GFM output with react-markdown and remark-gfm. Raw HTML is disabled and model-supplied images render their alt text without remote requests. Run `bun run test:monarch` for formatting and untrusted-output regression checks.
