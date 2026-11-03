# Chart data

Copied verbatim from the working repo `avp1598/sokoban` at commit `e189e5b91b25f9544f32dd557f8d977106a5b0cd`. `../make_chart.ts` reads these files.

| File here | Path in avp1598/sokoban | Run |
|---|---|---|
| `avp1598-sokoban/unit13_partB.json` | `results/unit13/partB.json` | WD2 (trace clip 5, curriculum) and WD3 (trace clip 1, curriculum): held-out 20,000 sampled, every 500M steps |
| `avp1598-sokoban/unit15_P_long_eval/P_long_s1_<step>_unf_heldout20k_summary.json` | `results/unit15/eval/` (same names) | P-long (trace clip 1, no curriculum): held-out 20,000; `sampled` = [solved, levels, Wilson low, Wilson high] |

All three runs use seed 8237, the same 733,120-parameter network and the same win-only reward, and are scored on the same 20,000 held-out unfiltered levels.
