# Numbers check: "Sokoban Without Hints"

Every number in [`../sokoban-without-hints.md`](../sokoban-without-hints.md), with the file it comes from and the exact text in that file. `check_numbers.ts` verifies both sides of every row: the post contains the "In post" text, and the source file contains the "Source text".

```bash
bun run posts/sokoban-without-hints/check_numbers.ts \
  --no-hints-root <path to a checkout of inductiveML/sokoban-no-hints> \
  --old-root <path to a checkout of avp1598/sokoban at e189e5b91b25f9544f32dd557f8d977106a5b0cd>
```

Source prefixes:
- `no-hints:` is [inductiveML/sokoban-no-hints](https://github.com/inductiveML/sokoban-no-hints) at `362b889`.
- `old:` is the private working repo `avp1598/sokoban` at `e189e5b91b25f9544f32dd557f8d977106a5b0cd`.
- `site:` is this repo.
- `web:` is an outside page. The checker confirms only the post side, and the source side was checked by hand on 2026-11-04.

Kinds:
- `exact`: the post's number appears verbatim in the source text.
- `rounded`: the post rounds the source value, as shown.
- `derived`: computed from other rows, as shown.

The figure's values are generated from `site:posts/sokoban-without-hints/data/` by `make_chart.ts`. Its printed output matches rows 24–26 and 28 (WD3 16.13% at 0.5B and 76.17% at 8.5B; P-long 81.42% at 8B and 70.87% at 2B).

| # | In post | Source | Source text | Kind |
|---|---|---|---|---|
| 1 | 0.7M-parameter Boxoban policy | no-hints:docs/protocol.md | 733,120 parameters in total | rounded (733,120 → 0.7M) |
| 2 | It solves 80.7% of the official unfiltered test split | no-hints:results/unit18/table_u18.md | 80.7 [78.1, 83.0] | exact |
| 3 | On Oct 17, 2026, kache | web:https://x.com/yacineMTB/status/2111504874844127300 | created_at "Sat Oct 17 17:09:26 +0000 2026" (read via api.fxtwitter.com) | exact |
| 4 | +0.25 each time a box lands on a target, and −1 when time runs out | no-hints:docs/protocol.md | +0.25 per box pushed onto a target and −0.25 per box pushed off; −1 on a timeout | exact |
| 5 | Our agent gets +1 for solving a level and 0 for everything else | no-hints:docs/protocol.md | win-only, +1 on a solve and 0 otherwise | exact |
| 6 | no episode pays more than 1 | no-hints:docs/vtrace_bug.md | episodes pay at most 1 | exact |
| 7 | the largest target reached 3.62e10 | no-hints:docs/vtrace_bug.md | the largest \|V-trace return\| reached **3.62e10** | exact |
| 8 | about 36 billion | no-hints:docs/vtrace_bug.md | 3.62e10 | derived (3.62e10 = 36.2 billion) |
| 9 | used c̄ = 5 and ρ̄ = 3.09 | no-hints:docs/vtrace_bug.md | Its tuned Boxoban settings use `vtrace_c_clip 5`, larger than `vtrace_rho_clip 3.09` | exact |
| 10 | IMPALA's analysis of V-trace assumes ρ̄ ≥ c̄ | web:https://arxiv.org/abs/1802.01561 | §4.1: "In addition we assume that the truncation levels are such that ρ̄ ≥ c̄." (also in Theorem 1) | exact |
| 11 | resampled a median of 25 times, at most 2,104 | no-hints:docs/vtrace_bug.md | outlier origins: a median of 25 times, at most 2,104 | exact |
| 12 | median 2.03 at those rows | no-hints:docs/vtrace_bug.md | outlier origins: median 2.03 | exact |
| 13 | controlled pair of 2B-step runs | no-hints:docs/vtrace_bug.md | Validation, fixed (2B steps, same config otherwise) | exact |
| 14 | the largest target fell from 4.35e7 to 3.67 | no-hints:docs/vtrace_bug.md | `c_clip 5` \| 4.35e7 | exact (control 4.35e7; fixed **3.67**) |
| 15 | to 3.67 | no-hints:docs/vtrace_bug.md | `c_clip 1` \| **3.67** | exact |
| 16 | went from 9.7% to 35.6% | no-hints:docs/vtrace_bug.md | held-out solve rate at 2B was 35.55% with the fix, against 9.66% for the control | rounded (9.66 → 9.7; 35.55 → 35.6) |
| 17 | Mainline PufferLib's default for this setting is already 1 | no-hints:docs/vtrace_bug.md | Mainline's default `vtrace_c_clip` is already 1. | exact |
| 18 | (Sep 17, 2026), Suarez independently removed V-trace and prioritized replay | web:https://x.com/jsuarez/status/2100698384768307385 | X article "PufferLib 5.0's Streamlined RL Algorithm", created_at "Thu Sep 17 21:28:18 +0000 2026" (Sep 18 in UTC+5:30): "So in total, we removed prio replay, advantage normalization, and VTrace" (read via api.fxtwitter.com) | exact |
| 18a | "a bad interaction between our prioritized sampling and higher sample reuse." | web:https://x.com/jsuarez/status/2100698384768307385 | "We also noticed a bad interaction between our prioritized sampling and higher sample reuse." | exact |
| 18b | no form of off-policy correction improved results in his experiments | web:https://x.com/jsuarez/status/2100698384768307385 | "No form of off-pol correction improved results in any scenario, even in ones specifically constructed for them to shine, compared to our strong baseline." | exact |
| 19 | the same 20,000 held-out levels | old:results/unit14/REPORT_unit14.md | ### Held-out 20,000, sampled (greedy) | exact |
| 20 | At 2B steps its held-out solve rate was 69.9%, against 64.3% for our full curriculum | old:results/unit14/REPORT_unit14.md | The no-curriculum control S-plain beats WD3 at 2B** (69.87% vs 64.34% held-out | rounded (69.87 → 69.9; 64.34 → 64.3) |
| 21 | 60.9% for a state-reset curriculum modelled on Suarez's | old:results/unit14/REPORT_unit14.md | S-suarez: 60.86% at 2B | rounded (60.86 → 60.9) |
| 22 | 54.1% for our curriculum without its level generator | old:results/unit14/REPORT_unit14.md | S-nosetter scored 54.07% at 2B | rounded (54.07 → 54.1) |
| 23 | A second seed of plain training reached 70.0% | old:results/unit15/REPORT_unit15.md | P-seed2 (plain, new seed 8238) scored **70.02% at 2B | rounded (70.02 → 70.0) |
| 24 | at 0.5B our full curriculum led plain training, 16.1% to 11.8% | old:results/unit14/REPORT_unit14.md | \| 0.5B \| 16.13% (11.00%) \| 11.80% (8.51%) | rounded (16.13 → 16.1; 11.80 → 11.8) |
| 25 | plain training reached 81.4% at 8B | old:results/unit15/REPORT_unit15.md | It ended at **81.42% at 8B** | rounded (81.42 → 81.4) |
| 26 | levelled off at 76.2% by 8.5B | old:results/unit13/REPORT_unit13.md | WD3 levels off near 76% from about 6B (76.17% at 8.5B) | rounded (76.17 → 76.2) |
| 27 | reached 46.1% at 12B | old:results/unit13/REPORT_unit13.md | WD2 never got there: 46.12% at 12B | rounded (46.12 → 46.1) |
| 28 | (70.9% at 2B here) | old:results/unit15/REPORT_unit15.md | \| 2.0B \| 64.34 (60.08) \| 70.87 (69.68) | rounded (70.87 → 70.9) |
| 29 | a fork of the fixed run showed that cap was not limiting | old:results/unit13/REPORT_unit13.md | so **the cap was not limiting** | exact |
| 30 | mainline PufferLib 5.0 with a small patch | no-hints:docs/protocol.md | mainline PufferLib 5.0 at `ea7f9893` | exact |
| 31 | three-layer conv encoder | no-hints:docs/protocol.md | then applies 3 conv layers | exact |
| 32 | The network has 733,120 parameters | no-hints:docs/protocol.md | 733,120 parameters in total | exact |
| 33 | 849,981 of Boxoban's | no-hints:docs/protocol.md | That leaves 849,981 levels | exact |
| 34 | A 3B-step run took 4.8 hours on one RTX 6000 Ada that was shared with other jobs | no-hints:README.md | 3B steps took 4.8 h on an RTX 6000 Ada (48 GB, shared with other jobs | exact |
| 35 | (1,000 levels, one attempt each) | no-hints:docs/protocol.md | `unfiltered/test` (1,000) | exact |
| 36 | **80.7%** with sampled actions and a 150-step limit | no-hints:results/unit18/table_u18.md | 77.1 [74.4, 79.6] \| 80.7 [78.1, 83.0] | exact (B-ours, 150 sampled) |
| 37 | **74.7%** greedy at 120 steps | no-hints:results/unit18/table_u18.md | 80.7 [78.1, 83.0] \| 74.7 [71.9, 77.3] | exact (B-ours, 120 greedy) |
| 38 | DeepMind's DRC work also caps episodes at 120 steps | no-hints:docs/outside_results.md | "Each episode is capped at 120 frames." | exact |
| 39 | The 8B run from the `simplecl` branch reaches 79.9% greedy at 120 steps | no-hints:results/unit18/table_u18.md | 80.5 [77.9, 82.8] \| 80.5 [77.9, 82.8] \| 79.9 [77.3, 82.3] | exact (P-long 8B, 120 greedy) |
| 40 | \| **80.7** [78.1, 83.0] \| 74.7 [71.9, 77.3] \| **21.2** [20.6, 21.7] \| **8.0** [7.1, 9.0] \| | no-hints:results/unit18/table_u18.md | B-ours (3B, mainline, hint-free conv) \| 77.1 [74.4, 79.6] \| 80.7 [78.1, 83.0] \| 74.7 [71.9, 77.3] \| 78.6 [76.0, 81.0] \| 19.9 [19.3, 20.4] \| 21.2 [20.6, 21.7] \| 7.1 [6.3, 8.0] \| 8.0 [7.1, 9.0] | exact (table row: B-ours) |
| 41 | \| 79.7 [77.1, 82.1] \| 73.9 [71.1, 76.5] \| 17.8 [17.3, 18.3] \| 5.9 [5.1, 6.7] \| | no-hints:results/unit18/table_u18.md | B-ours-gae (3B) \| 75.8 [73.0, 78.4] \| 79.7 [77.1, 82.1] \| 73.9 [71.1, 76.5] \| 77.7 [75.0, 80.2] \| 15.9 [15.4, 16.4] \| 17.8 [17.3, 18.3] \| 4.8 [4.1, 5.6] \| 5.9 [5.1, 6.7] | exact (table row: B-ours-gae) |
| 42 | \| 80.5 [77.9, 82.8] \| **79.9** [77.3, 82.3] \| 15.4 [14.9, 15.9] \| 5.3 [4.6, 6.1] \| | no-hints:results/unit18/table_u18.md | P-long (8.0B, simplecl) \| 80.5 [77.9, 82.8] \| 80.5 [77.9, 82.8] \| 79.9 [77.3, 82.3] \| 80.1 [77.5, 82.5] \| 14.7 [14.2, 15.2] \| 15.4 [14.9, 15.9] \| 4.5 [3.8, 5.3] \| 5.3 [4.6, 6.1] | exact (table row: P-long 8B) |
| 43 | \| 74.4 [71.6, 77.0] \| 70.8 [67.9, 73.5] \| 15.3 [14.9, 15.9] \| 5.9 [5.1, 6.7] \| | no-hints:results/unit18/table_u18.md | M1-unf (shipped shaped reward + network, unfiltered-trained, 3B) \| 72.4 [69.5, 75.1] \| 74.4 [71.6, 77.0] \| 70.8 [67.9, 73.5] \| 72.2 [69.3, 74.9] \| 14.9 [14.4, 15.4] \| 15.3 [14.9, 15.9] \| 5.0 [4.3, 5.8] \| 5.9 [5.1, 6.7] | exact (table row: M1-unf) |
| 44 | \| 37.2 [34.3, 40.2] \| 32.2 [29.4, 35.2] \| 11.1 [10.6, 11.5] ¹ \| 4.7 [4.0, 5.4] \| | no-hints:results/unit18/table_u18.md | M1 (mainline as shipped, medium-trained, 3B) ¹ \| 33.4 [30.5, 36.4] \| 37.2 [34.3, 40.2] \| 32.2 [29.4, 35.2] \| 34.9 [32.0, 37.9] \| 10.5 [10.1, 11.0] \| 11.1 [10.6, 11.5] \| 4.5 [3.9, 5.3] \| 4.7 [4.0, 5.4] | exact (table row: M1) |
| 45 | medium/valid (20,000), sampled \| hard (3,332), sampled | no-hints:results/unit18/table_u18.md | medium/valid (20,000), 150: greedy \| sampled \| hard (3,332), 150: greedy \| sampled | exact (table header) |
| 46 | +0.25 per box onto a target, −0.25 per box off, −1 on a timeout | no-hints:docs/protocol.md | +0.25 per box pushed onto a target and −0.25 per box pushed off; −1 on a timeout | exact |
| 47 | its own 9.9M-parameter network | no-hints:README.md | its own 9.9M-parameter network | exact |
| 48 | It reaches 74.4%. The hint-free model is 6.3 points higher on test with sampled actions (paired McNemar p = 1.3e-5) | no-hints:README.md | Hint-free is **+6.3 points** on test (sampled; paired McNemar p = 1.3e-5) | exact |
| 49 | 3.9 points higher greedy at 120 steps (p = 0.015) | no-hints:README.md | +3.9 greedy at 120 steps (p = 0.015) | exact |
| 50 | It also solves more medium levels (21.2% vs 15.3%) and hard levels (8.0% vs 5.9%) | no-hints:README.md | It also solves more medium (21.2 vs 15.3) and hard (8.0 vs 5.9) levels. | exact |
| 51 | the shipped config doesn't use V-trace | no-hints:results/unit18/M1_unf_logged_config.ini | vtrace = 0 | exact |
| 52 | Our model trained without V-trace still scores 79.7% | no-hints:results/unit18/table_u18.md | B-ours-gae (3B) \| 75.8 [73.0, 78.4] \| 79.7 [77.1, 82.1] | exact |
| 53 | and scores 37.2% on unfiltered test | no-hints:results/unit18/table_u18.md | M1 (mainline as shipped, medium-trained, 3B) ¹ \| 33.4 [30.5, 36.4] \| 37.2 [34.3, 40.2] | exact |
| 54 | the same shaped config trained on unfiltered levels scores 74.4% | no-hints:results/unit18/table_u18.md | 72.4 [69.5, 75.1] \| 74.4 [71.6, 77.0] | exact |
| 55 | against 104.1 and | no-hints:results/unit17_games/table_g.md | \| 1 \| G-default \| 81.534 \| 97.020 \| 103.824 \| 104.453 \| 104.138 \| | rounded (104.138 → 104.1; snake seed 1, default) |
| 56 | 118.1 and 119.1 | no-hints:results/unit17_games/table_g.md | \| 1 \| G-conv \| 101.285 \| 111.382 \| 116.655 \| 119.541 \| 118.098 \| | rounded (118.098 → 118.1) |
| 57 | 119.1 with the conv encoder | no-hints:results/unit17_games/table_g.md | \| 2 \| G-conv \| 104.865 \| 113.294 \| 117.805 \| 120.439 \| 119.122 \| | rounded (119.122 → 119.1) |
| 58 | against 104.1 and 104.4 | no-hints:results/unit17_games/table_g.md | \| 2 \| G-default \| 81.953 \| 96.579 \| 103.260 \| 105.587 \| 104.424 \| | rounded (104.424 → 104.4) |
| 59 | 3× the shipped learning rate | no-hints:results/unit17_games/table_g.md | \| snake \| 0.00798577 \| 82.804 \| 98.804 \| 110.976 \| ×3 \| | exact |
| 60 | 38.70 and 38.72 trash collected per game (of 40), against 37.51 and 37.50 (2 seeds) | no-hints:results/unit17_games/table_g.md | mean trash collected per game (of 40) | exact (header; values rows 61–64) |
| 61 | 38.70 and 38.72 | no-hints:results/unit17_games/table_g.md | \| 1 \| G-conv \| 33.212 \| 35.842 \| 38.552 \| 38.848 \| 38.700 \| | rounded (38.700 → 38.70) |
| 62 | 38.72 trash | no-hints:results/unit17_games/table_g.md | \| 2 \| G-conv \| 31.087 \| 38.105 \| 38.620 \| 38.828 \| 38.724 \| | rounded (38.724 → 38.72) |
| 63 | against 37.51 and | no-hints:results/unit17_games/table_g.md | \| 1 \| G-default \| 28.765 \| 35.209 \| 37.033 \| 37.984 \| 37.509 \| | rounded (37.509 → 37.51) |
| 64 | and 37.50 (2 seeds) | no-hints:results/unit17_games/table_g.md | \| 2 \| G-default \| 27.425 \| 33.905 \| 37.400 \| 37.596 \| 37.498 \| | rounded (37.498 → 37.50) |
| 65 | Seeds split 2–1: final solve rates on 2,000 fresh maps were 84.17%, 84.95% and 85.78% with the conv encoder, against 84.15%, 82.25% and 87.40% | no-hints:results/unit17_games/table_g.md | maze: solved fraction of 2,000 fresh maps (%) | exact (header; values rows 66–67) |
| 66 | were 84.17%, | no-hints:results/unit17_games/table_g.md | \| 1 \| G-conv \| 86.95 (67.90) \| 87.15 (75.85) \| 84.55 (75.25) \| 83.80 (75.65) \| 84.17 \| | exact (maze seed 1, conv) |
| 67 | 84.95% and | no-hints:results/unit17_games/table_g.md | \| 2 \| G-conv \| 83.30 (63.55) \| 87.50 (74.70) \| 84.50 (74.40) \| 85.40 (76.10) \| 84.95 \| | exact (maze seed 2, conv) |
| 67a | 85.78% with the conv encoder | no-hints:results/unit17_games/table_g.md | \| 3 \| G-conv \| 86.05 (65.75) \| 86.20 (74.65) \| 85.85 (75.50) \| 85.70 (76.90) \| 85.78 \| | exact (maze seed 3, conv) |
| 67b | against 84.15%, | no-hints:results/unit17_games/table_g.md | \| 1 \| G-default \| 91.60 (76.60) \| 84.85 (73.95) \| 84.40 (71.05) \| 83.90 (73.00) \| 84.15 \| | exact (maze seed 1, default) |
| 67c | 82.25% and | no-hints:results/unit17_games/table_g.md | \| 2 \| G-default \| 91.75 (75.40) \| 84.55 (71.75) \| 82.55 (72.30) \| 81.95 (72.80) \| 82.25 \| | exact (maze seed 2, default) |
| 67d | 87.40% with the default one | no-hints:results/unit17_games/table_g.md | \| 3 \| G-default \| 92.10 (77.70) \| 87.15 (75.45) \| 87.70 (76.80) \| 87.10 (76.20) \| 87.40 \| | exact (maze seed 3, default) |
| 67e | Seeds split 2–1 | no-hints:results/unit17_games/table_g.md | maze: per-seed winner ['conv', 'conv', 'def'] | exact |
| 68 | its 0.5B-step budget | no-hints:README.md | at the Sokoban Speedrun's budget (0.5B steps) | exact |
| 69 | Hint-free training scored 0.0% and 5.8% (2 seeds), against a 72.8% average for the shaped setup (3 seeds) | no-hints:README.md | gives 0.0% and 5.8% (2 seeds) against 72.8% shaped (3 seeds) | exact |
| 70 | Our 3B model was at 0.45% on held-out levels at 0.5B | no-hints:results/unit17_b_ours/table_b.md | \| 0.5B \| 0.45 (0.01) | exact |
| 71 | took off around 0.7–1B | old:results/unit19/REPORT_unit19.md | only took off around 0.7–1B | exact |
| 72 | It reports 99% on unfiltered test after 1e9 steps, with a shaped reward | no-hints:README.md | reports 99% on unfiltered test after 1e9 steps with a shaped reward | exact |
| 73 | Our 3B model solves 21.2% of medium levels and 8.0% of hard ones | no-hints:results/unit18/table_u18.md | 21.2 [20.6, 21.7] \| 7.1 [6.3, 8.0] \| 8.0 [7.1, 9.0] | exact |
| 74 | It was at 0.05% after 1B steps | old:results/unit16/REPORT_unit16.md | 0.05% unfiltered at 1B, with policy entropy at | exact |
| 74a | with policy entropy at its maximum | old:results/unit16/REPORT_unit16.md | policy entropy was 1.609 (the maximum for 5 actions) | exact |
| 75 | 2 seeds per arm for snake and trash_pickup and 3 for maze | no-hints:results/unit17_games/table_g.md | (2 seeds; seed 3 not run) | exact (maze table has seeds 1–3) |
| 76 | the speedrun comparison has 2 hint-free seeds | no-hints:README.md | (2 seeds) against 72.8% shaped (3 seeds) | exact |
| 77 | 72.2% on unfiltered test, against 60.8% for PufferLib's shaped reward at the same budget, from one seed | web:https://x.com/dkta0/status/2113281605750186187 | "left it running overnight at puffer's full budget, 3b moves … win-only with backwards starts: 72.2% of unfiltered test puffer's shaped reward, same budget: 60.8% one seed at that length" (Oct 22, 2026; read via api.fxtwitter.com) | exact |
| 77b | The post doesn't say whether actions were greedy or sampled | no-hints:docs/outside_results.md | **Not stated (unverified):** greedy vs sampled | exact |
| 78 | games start one pull away from the goal | web:https://blog.dkta.dev/blog/playing-sokoban-backwards/ | "games start one pull from solved and step back whenever the agent solves half of them at the current depth" | exact |
| 79 | non-LLM record is 74.2% on unfiltered test, greedy at 120 steps | no-hints:README.md | greedy, 120 steps, scored on unfiltered/test, with Puffer's shaped reward. Its record scores 74.2% | exact |
| 80 | in 12 min 38 s on one H100 | no-hints:README.md | in 12:38 on one H100 | exact |
| 81 | Our 74.7% under the same protocol took 3B steps rather than 0.5B | no-hints:results/unit18/table_u18.md | 74.7 [71.9, 77.3] | exact (3B: B-ours; 0.5B: row 68) |
| 82 | runs/b_ours/ckpt/boxoban/b_ours/0000002999975936.bin | no-hints:README.md | scripts/eval.sh runs/b_ours/ckpt/boxoban/b_ours/0000002999975936.bin my_b_ours | exact |
| 83 | prints `REPRODUCED EXACTLY` | no-hints:README.md | It prints the table above and `REPRODUCED EXACTLY` | exact |
| 84 | Boxoban is DeepMind's large set of 10×10 Sokoban levels | no-hints:docs/protocol.md | It reads the 4 × 10 × 10 board | exact |
