# Numbers check: "Training on Partial Evidence"

Every number in [`../training-on-partial-evidence.md`](../training-on-partial-evidence.md), with the file it comes from and the exact text in that file. `scripts/check-numbers.ts` verifies both sides of every row: the post contains the "In post" text, and the source file contains the "Source text".

```bash
bun run scripts/check-numbers.ts \
  --post posts/training-on-partial-evidence.md --check posts/training-on-partial-evidence/numbers_check.md \
  --root abacus=<path to a checkout of avp1598/abacus at 85a5cf8>
```

`abacus:` is the private research repo `avp1598/abacus` at `85a5cf8fda40c81df9016b066f626a9aa38b950a`.

The figures are generated from the same repo's `rows.json` files by `make_figures.ts`, so they aren't listed row by row. The script checks the world figure's output tape against the repo's Python interpreter (target 1878: `SWAP MOVR SWAP MOVL MOVL MOVL`), and fails if the grid figure's per-writer counts stop being 5 to 8.

Kinds:
- `exact`: the post's number appears verbatim in the source text.
- `rounded`: the post rounds the source value, as shown.
- `derived`: computed from sourced values, as shown.

| # | In post | Source | Source text | Kind |
|---|---|---|---|---|
| 1 | from 54% to 73% | abacus:experiments/abacus_dropraw0/REPORT.md | \| MASK100_ALL \| 1.0 \| 0.317 \| 0.725 ± 0.071 | rounded (0.540 → 54%; 0.725 → 73%) |
| 2 | about 70% within 10,000 training steps | abacus:experiments/abacus_scale0/REPORT.md | \| RAW-NP-ALL75 \| 0.272 \| 0.696 ± 0.076 | rounded (0.691–0.696 → about 70%) |
| 3 | stays near 59% | abacus:experiments/abacus_scale0/REPORT.md | \| RAW-NP-MASK0 \| 0.272 \| 0.588 ± 0.079 | rounded (0.588–0.596 → near 59%) |
| 4 | A program is 6 slots, each filled with one of 7 operations | abacus:abacus_b1/census.py | itertools.product(range(7),repeat=6) | exact |
| 5 | 117,649 programs, which compute 2,143 distinct functions | abacus:abacus_b5/split.py | c['programs'].shape==(117649,6) and c['full'].shape==(2143,142) | exact |
| 6 | runs its 6 slots once per tape cell | abacus:abacus_b1/vendor/vm_hard.py | for _ in range(state.tape.shape[1] if repetitions is None else repetitions): | exact |
| 7 | 8 at each of the lengths 4 to 8, 40 in total | abacus:experiments/abacus_cover0/REPORT.md | \| FULL \| {4: 8, 5: 8, 6: 8, 7: 8, 8: 8} | exact (5 × 8 = 40) |
| 8 | keeping its 8 or 64 best guesses | abacus:experiments/abacus_cover0/REPORT.md | CHECK64 (single model, 32 models) | exact |
| 9 | including lengths 16, 32 and 64 | abacus:abacus_b5/constants.py | LENGTHS=(4,5,6,7,8,16,32,64) | exact |
| 10 | output cell 2 reads input cell 4 and output cell 5 reads input cell 7 | abacus:experiments/abacus_decomp0/REPORT.md | relation 1 = cell 2 <- source 4, relation 2 = cell 5 <- source 7 | exact |
| 11 | at length 8, output cell 2 | abacus:abacus_b5/split.py | r=np.asarray(routes); start=4+5+6+7 | exact (start = 22 is the length-8 block) |
| 12 | All 24 test targets come from the fourth | abacus:abacus_b5/split.py | EXPECTED={'A':(25,603,25,31,9,17) | exact (figure: 25 held out, 603 in training, 9 and 17 with one read alone; 577 = 603 − 9 − 17) |
| 13 | 20 fresh sets of examples per target | abacus:experiments/abacus_seedsplit0/REPORT.md | RATE grid (20 shared fresh episodes per target | exact |
| 14 | averages 32 writers | abacus:experiments/abacus_dropraw0/REPORT.md | Part A — relation-bit writer (32 seeds each, 70000–70031) | exact |
| 15 | the 12 solved targets where no training program contained both reads | abacus:experiments/abacus_invent0/REPORT.md | INVENT (12 targets): EXAMPLE-DRIVEN 12, HABIT 0, UNRESOLVED 0 | exact |
| 16 | the best writer found the right function on 100% of fresh examples, and on 0% of examples from unrelated functions | abacus:experiments/abacus_invent0/REPORT.md | \| INVENT (all) \| 12 \| 12 \| 0 \| 0 \| 1.000 \| 0.000 | rounded (mean RIGHT_FRESH 1.000; mean WRONG_FAR 0.000) |
| 17 | Swapping two batches halfway through training | abacus:experiments/abacus_seedsplit0/REPORT.md | Perturbed copies swap the batches of updates 5,000 and 5,001 | exact (of 10,000 updates) |
| 18 | (similarity 0.97) | abacus:experiments/abacus_seedsplit0/REPORT.md | LUCK_SIM = 0.9684 | rounded (0.9684 → 0.97) |
| 19 | agreed more (0.71) | abacus:experiments/abacus_seedsplit0/REPORT.md | \| SAME_INIT \| 9 \| 0.7129 | rounded (0.7129 → 0.71) |
| 20 | a data order (0.64) | abacus:experiments/abacus_seedsplit0/REPORT.md | \| SAME_DATA \| 9 \| 0.6363 | rounded (0.6363 → 0.64) |
| 21 | or neither (0.61) | abacus:experiments/abacus_seedsplit0/REPORT.md | \| NEITHER \| 18 \| 0.6132 | rounded (0.6132 → 0.61) |
| 22 | Starting weights were the only one of the two with a clear effect | abacus:experiments/abacus_seedsplit0/REPORT.md | P(null ≥ observed) INIT 0.0067, DATA 0.2542 | exact |
| 23 | Nine writers (three starting weights × three data orders) | abacus:experiments/abacus_seedsplit0/REPORT.md | \| i2 d2 \| . \| . | exact |
| 24 | Each writer solves 5 to 8 targets at least half the time | abacus:experiments/abacus_seedsplit0/REPORT.md | \| i2 d1 \| . \| . \| . \| .50 \| . \| .55 \| 1.00 \| . \| .20 \| .30 \| . \| . \| .05 \| . \| .05 \| .75 \| . \| 1.00 \| . \| 1.00 \| 1.00 \| .65 \| . \| . \| 8 \| | exact (max 8 here; min 5 in row i0 d0; make_figures.ts asserts both) |
| 25 | one writer alone solves 8 to 14 of the 24 targets | abacus:experiments/abacus_cover0/REPORT.md | \| 1 \| 4.4 (2–6) \| 11.5 (8–14) | exact |
| 26 | All 32 together | abacus:experiments/abacus_cover0/REPORT.md | \| 32 \| 22.0 (22–22) \| 24.0 (24–24) | exact |
| 27 | keeps only programs reproducing all 40 examples | abacus:experiments/abacus_cover0/REPORT.md | \| FULL \| {4: 8, 5: 8, 6: 8, 7: 8, 8: 8} \| 1.00 \| 0.000 \| 1.000 \| 1.000 \| **0.000** | exact (false accepts 0.000) |
| 28 | It sends 189 distinct functions to the checker on average and reaches 100% | abacus:experiments/abacus_cover0/REPORT.md | \| POOL(32, 64) \| 189.0 (0.088) \| 673.5 \| 1.000 \| 0.088 \| 11.3 \| | exact |
| 29 | would reach 8.8%, about 11 times less | abacus:experiments/abacus_cover0/REPORT.md | EFFICIENCY R of POOL(32, 64) = 11.3 | rounded (11.3 → about 11; 0.088 → 8.8%) |
| 30 | \| 0% \| everything \| 0.540 \| 0.995 \| | abacus:experiments/abacus_dropraw0/REPORT.md | \| MASK0 \| 0.0 \| 0.209 \| 0.540 ± 0.088 \| 13.06 \| 0.995 | exact |
| 31 | \| 25% \| short lengths only \| 0.571 \| 0.992 \| | abacus:experiments/abacus_dropraw0/REPORT.md | \| MASK25 \| 0.25 \| 0.253 \| 0.571 ± 0.077 \| 13.91 \| 0.992 | exact |
| 32 | \| 50% \| short lengths only \| 0.591 \| 0.976 \| | abacus:experiments/abacus_dropraw0/REPORT.md | \| MASK50 \| 0.5 \| 0.266 \| 0.591 ± 0.083 \| 14.25 \| 0.976 | exact |
| 33 | \| 75% \| short lengths only \| 0.666 \| 0.948 \| | abacus:experiments/abacus_dropraw0/REPORT.md | \| MASK75 \| 0.75 \| 0.285 \| 0.666 ± 0.101 \| 15.94 \| 0.948 | exact |
| 34 | \| 85% \| short lengths only \| 0.697 \| 0.912 \| | abacus:experiments/abacus_dropraw0/REPORT.md | \| MASK85 \| 0.85 \| 0.239 \| 0.697 ± 0.091 \| 17.00 \| 0.912 | exact |
| 35 | \| 95% \| short lengths only \| 0.618 \| 0.789 \| | abacus:experiments/abacus_dropraw0/REPORT.md | \| MASK95 \| 0.95 \| 0.183 \| 0.618 ± 0.085 \| 14.94 \| 0.789 | exact |
| 36 | \| 100% \| short lengths only \| 0.195 \| 0.175 \| | abacus:experiments/abacus_dropraw0/REPORT.md | \| MASK100 \| 1.0 \| 0.018 \| 0.195 ± 0.101 \| 4.69 \| 0.175 | exact |
| 37 | \| **100%** \| **1 or 4 tapes at every length** \| **0.725** \| **0.959** \| | abacus:experiments/abacus_dropraw0/REPORT.md | \| MASK100_ALL \| 1.0 \| 0.317 \| 0.725 ± 0.071 \| 17.41 \| 0.959 | exact |
| 38 | 1 or 4 tapes | abacus:abacus_dropraw0/parta.py | if arm == 'MASK100_ALL': return ['ALL_1' if tt < .5 else 'ALL_4' for tt in t] | exact |
| 39 | raised held-out accuracy by 18.5 points | abacus:experiments/abacus_dropraw0/REPORT.md | \| 0.959 \| 0.999 \| 20.5 / 22.0 / 22.5 \| 0.851 / 0.910 / 0.941 \| +0.185 [+0.068, +0.301] | exact |
| 40 | cost 3.6 points on taught functions | abacus:experiments/abacus_dropraw0/REPORT.md | 13.06 \| 0.995 | derived (0.995 − 0.959) |
| 41 | the gains at 25%, 50% and 95% could be zero | abacus:experiments/abacus_dropraw0/REPORT.md | +0.078 [-0.056, +0.222] | exact (MASK95; MASK25 is +0.031 [-0.051, +0.108], MASK50 is +0.051 [-0.036, +0.136]) |
| 42 | moved accuracy by only 0.8 points | abacus:experiments/abacus_dropraw0/REPORT.md | +0.008 [-0.057, +0.071] | exact |
| 43 | hiding (0.666) beat dropping (0.580) by 8.5 points with an interval of −0.5 to +17.9 | abacus:experiments/abacus_dropraw0/REPORT.md | \| M75: MASK75 − DROP75 \| +0.0851 \| [-0.005, +0.179] | rounded (+0.0851 → 8.5 points) |
| 44 | 576 hand-built yes/no features per length | abacus:abacus_b5/features.py | a.reshape(len(a),5,576) | exact |
| 45 | a writer of 1,363,335 parameters | abacus:experiments/abacus_scale0/REPORT.md | Raw-tape writer, no program input (GPU, lr 1e-3, 1,363,335 parameters, 32 seeds) | exact |
| 46 | Hiding on 75% of rows cut the raw writer's accuracy from 0.585 to 0.174 at 40,000 steps | abacus:experiments/abacus_dropraw0/REPORT.md | \| RAW-MASK75 \| 0.031 \| 0.174 ± 0.040 | exact (RAW-MASK0 at 40k is 0.585) |
| 47 | fixed the answer for 92.8% of training rows | abacus:experiments/abacus_dropraw0/REPORT.md | the program alone fixes the label for 92.8% of rows | exact |
| 48 | the first guess was exactly right on 95.1% of training rows, but in the right function on only 7.3% | abacus:experiments/abacus_dropraw0/REPORT.md | RAW-MASK75 writes the exact label for 95.1% of training rows, yet its greedy program lands in the target class on only 7.3% of TR episodes | exact |
| 49 | (0.596 without it, 0.585 with it) | abacus:experiments/abacus_scale0/REPORT.md | \| RAW-MASK0 (with program) \| 0.257 \| 0.585 ± 0.071 | exact |
| 50 | \| Hand-built features \| 10k \| 0.509 \| 0.754 \| +0.245 (clear) \| | abacus:experiments/abacus_scale0/REPORT.md | E_REL_NP +0.245 [+0.126, +0.363] (SURVIVES) | exact (arms 0.509 and 0.754 in section 1) |
| 51 | \| Raw tapes \| 10k \| 0.588 \| 0.691–0.696 \| +0.104 to +0.108 (clear) \| | abacus:experiments/abacus_scale0/REPORT.md | E_RAW_100 UNRESOLVED (E by budget +0.104 / +0.098 / +0.080) | exact (arms 0.588, 0.691 and 0.696 in section 2) |
| 52 | \| Raw tapes \| 20k \| 0.591 \| 0.689–0.703 \| +0.098 to +0.112 (clear) \| | abacus:experiments/abacus_scale0/REPORT.md | \| RAW-NP-ALL75 \| 0.292 \| 0.703 ± 0.097 | rounded (E_RAW_75 +0.1125 → +0.112) |
| 53 | \| Raw tapes \| 40k \| 0.596 \| 0.664–0.677 \| +0.067 to +0.080 (not clear) \| | abacus:experiments/abacus_scale0/REPORT.md | \| RAW-NP-ALL100 \| 0.292 \| 0.677 ± 0.100 | exact |
| 54 | +0.104 to +0.108 | abacus:experiments/abacus_scale0/REPORT.md | E_RAW_75 UNRESOLVED (E by budget +0.108 / +0.112 / +0.067) | exact |
| 55 | 0.506, then 0.685, then 0.714 | abacus:experiments/abacus_scale0/REPORT.md | E_RAW_DROP SURVIVES (E by budget -0.081 / +0.094 / +0.118) | exact (arm rows 0.506 ± 0.099, 0.685 ± 0.072, 0.714 ± 0.057) |
| 56 | which is clearly above full-example training at 40,000 steps | abacus:experiments/abacus_scale0/REPORT.md | E_RAW_DROP +0.118 [+0.031, +0.201] | exact |
| 57 | it recovered exact labels equally well from stored and fresh tapes | abacus:experiments/abacus_scale0/REPORT.md | \| RAW-NP-MASK0 \| 0.688 / 0.980 \| 0.684 / 0.979 | exact |
| 58 | 44 of 288 test cases solved at 8 guesses, the same as the plain writer | abacus:experiments/abacus_ram0_exact_memory/REPORT.md | best memory arm RAM_HARD TEST@8 44/288 vs PLAIN 44/288 (net +0) | exact |
| 59 | right 6.2% of the time with eight tapes and 6.0% with one | abacus:experiments/abacus_ambig0/REPORT.md | \| HO \| L4_8 \| 420 / 21 \| 0.121 / 0.062 | rounded (ambiguous-episode column; L4_1 is 0.062 / 0.060 in the next row) |
| 60 | (28.9% and 20.0%) | abacus:experiments/abacus_ambig0/REPORT.md | \| HO \| L4_1 \| 467 / 24 \| 0.062 / 0.060 \| 0.412 / 0.420 \| 0.052 / 0.054 \| 0.200 | exact (RANDOM_FIT; L4_8 is 0.289) |
| 61 | (62.1% and 42.0%) | abacus:experiments/abacus_ambig0/REPORT.md | \| HO \| L4_8 \| 420 / 21 \| 0.121 / 0.062 \| 0.610 / 0.621 | exact (GEN ambiguous column; L4_1 is 0.420) |
| 62 | solved 4 of 144 test cases there at 8 guesses, against 40 of 144 on this one | abacus:experiments/abacus_ram0_exact_memory/REPORT.md | \| PLAIN \| **44** \| 6 \| 184 \| 155 \| 10 / 14 / 16 / 0 / 4 / 0 \| | derived (A cells 10 + 14 + 16 = 40; B cells 0 + 4 + 0 = 4; each panel has 288 / 2 = 144 cases) |
| 63 | The largest writer has 1.4 million parameters | abacus:experiments/abacus_dropraw0/REPORT.md | raw-tape writer (1,363,335 parameters | rounded (1,363,335 → 1.4 million) |
| 64 | in 10,000 steps where dropping rows needs 20,000 to 40,000 | abacus:experiments/abacus_scale0/REPORT.md | **20,000 updates** | exact (DROP75 at 20k is 0.685; at 40k 0.714) |
