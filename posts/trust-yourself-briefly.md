---
pageTitle: "Trust Yourself, Briefly: A GRPO Warm Start · Inductive ML"
shareTitle: "Trust Yourself, Briefly: Confidence as a GRPO Warm Start"
description: "For 8 updates we nudged each step by the model's own confidence. It reached GRPO's final accuracy on about half the compute."
date: 2026-11-01
---

## The idea in one paragraph

Most reinforcement learning (RL) for reasoning works like a teacher who only looks at the final answer. The model writes out its working line by line, reaches an answer, and the whole thing gets one grade. If the answer is right, every line gets the credit, sloppy ones included. If it's wrong, every line gets the blame, good ones included. Figuring out which lines actually earned the grade is called credit assignment, and GRPO (Group Relative Policy Optimization), a popular method, mostly skips it: the model answers the same question several times, and each attempt is graded against the others. We tried a small addition. For the first 8 training updates, each line of working also got a little nudge up or down, depending on how sure the model itself was about that line. After that we switched the nudge off and let plain GRPO take over. We call this a warm start. On a toy task, it made the model learn noticeably faster. It also failed in a few ways that taught us as much as the wins did.

## The task

We wanted a task where every step can be checked, so we made one up. We call it variable chains. The prompt gives three chains of simple definitions, like `b = a + 3`. Each step adds or subtracts a small number, and every value stays between 0 and 19. The model works through each chain one line at a time, then writes three answers.

Here's a made-up example to show the shape. Each column is one chain, three steps long.

```text
a = 4        p = 12       m = 9
b = a + 3    q = p - 7    n = m + 8
c = b - 5    r = q + 2    o = n - 10
d = c + 9    s = r + 6    t = o - 4
Find d, s and t.
```

And a good response:

```text
b = 4 + 3 = 7
c = 7 - 5 = 2
d = 2 + 9 = 11
q = 12 - 7 = 5
r = 5 + 2 = 7
s = 7 + 6 = 13
n = 9 + 8 = 17
o = 17 - 10 = 7
t = 7 - 4 = 3
ANSWER d = 11
ANSWER s = 13
ANSWER t = 3
```

We call the number of steps in a chain its depth. Training used chains of depth 3 and 4. So whenever we mention depth 5, 6 or 7 below, those are longer chains than the model ever saw in training. The model was LFM2.5-2.6B-Base, trained with LoRA adapters: small extra weights added on top of the model, so the original weights never change.

## How the method works

When the model writes a line like `b = 4 + 3 = 7`, the last thing it writes is the result, the `7`. Just before writing it, the model assigns that number a probability. If nearly all of it sits on `7`, the model was sure. If it was split between `7` and a few other numbers, it wasn't. That probability is what we mean by confidence.

Then, within a single response, we compare each line's confidence with the average for that response. Lines the model was surer about than usual get nudged up a little. Lines it was less sure about get nudged down a little. As a formula:

> nudge = this line's confidence − average confidence over the lines in this response

Because each line is measured against the average, the nudges in a response cancel out. They don't make an answer look better or worse overall. They only shift credit from some lines to others. The normal GRPO grade for the whole answer still applies on top.

<figure>
<svg viewBox="0 0 400 392" width="100%" style="max-width:480px;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Top: GRPO gives four step lines one shared score. Bottom: the warm start adds an up or down marker per line, sized by the model's confidence.">
<g fill="currentColor" font-family="Archivo, system-ui, sans-serif">
<text x="20" y="26" font-size="17" font-weight="600">GRPO</text>
<text x="20" y="48" font-size="14" opacity="0.75">one grade for the whole answer</text>
<text x="20" y="226" font-size="17" font-weight="600">Warm start</text>
<text x="20" y="248" font-size="14" opacity="0.75">plus a nudge per line, for the first 8 updates</text>
</g>
<g fill="currentColor" font-family="IBM Plex Mono, ui-monospace, monospace" font-size="16">
<text x="20" y="84">b = 4 + 3 = 7</text>
<text x="20" y="112">c = 7 - 5 = 2</text>
<text x="20" y="140">d = 2 + 9 = 11</text>
<text x="20" y="168">q = 12 - 7 = 5</text>
<text x="20" y="284">b = 4 + 3 = 7</text>
<text x="20" y="312">c = 7 - 5 = 2</text>
<text x="20" y="340">d = 2 + 9 = 11</text>
<text x="20" y="368">q = 12 - 7 = 5</text>
</g>
<g fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.6">
<path d="M168 70 H178 V174 H168"/>
<path d="M178 122 H212"/>
<path d="M206 116 L213 122 L206 128"/>
</g>
<rect x="218" y="104" width="150" height="36" rx="4" fill="none" stroke="#b8381f" stroke-width="2"/>
<text x="293" y="127" text-anchor="middle" fill="currentColor" font-family="Archivo, system-ui, sans-serif" font-size="15">one score</text>
<path d="M20 198 H380" stroke="currentColor" stroke-opacity="0.15"/>
<g fill="#b8381f">
<polygon points="180,268 170,286 190,286"/>
<polygon points="180,301 174,311 186,311"/>
<polygon points="180,339 174,329 186,329"/>
<polygon points="180,374 170,356 190,356"/>
</g>
<g fill="currentColor" font-family="Archivo, system-ui, sans-serif" font-size="14" opacity="0.75">
<text x="198" y="284">surer</text>
<text x="198" y="368">less sure</text>
</g>
<rect x="268" y="304" width="114" height="36" rx="4" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.6"/>
<text x="325" y="327" text-anchor="middle" fill="currentColor" font-family="Archivo, system-ui, sans-serif" font-size="15">same score</text>
</svg>
<figcaption>Top, GRPO: one score for every line. Bottom, the warm start: each line also gets ▲ or ▼, sized by the model's confidence.</figcaption>
</figure>

We only do this for training updates 1 to 8. From update 9 onward it's plain GRPO. The reason is simple: keep rewarding a model for being confident, and it learns to look confident. We'll get to what that looked like.

<figure>
<svg viewBox="0 0 400 134" width="100%" style="max-width:480px;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A bar of 200 training updates. Updates 1 to 8 are shaded: confidence credit on. Updates 9 to 200 are plain GRPO.">
<rect x="10" y="54" width="380" height="34" rx="4" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.5"/>
<rect x="10" y="54" width="15.2" height="34" fill="#b8381f"/>
<path d="M17.6 52 V34" stroke="currentColor" stroke-width="1.5" opacity="0.6"/>
<g fill="currentColor" font-family="Archivo, system-ui, sans-serif">
<text x="10" y="26" font-size="15">confidence credit on · updates 1–8</text>
<text x="210" y="76" font-size="15" text-anchor="middle">plain GRPO · updates 9–200</text>
<g font-size="13" opacity="0.75">
<text x="10" y="106">1</text>
<text x="25" y="106">8</text>
<text x="200" y="106" text-anchor="middle">100</text>
<text x="390" y="106" text-anchor="end">200</text>
<text x="200" y="128" text-anchor="middle">training updates →</text>
</g>
</g>
</svg>
<figcaption>The confidence nudge is on for the first 8 of 200 updates, then switched off.</figcaption>
</figure>

## What worked

Our main comparison is plain GRPO, with the same reward we used everywhere. That reward includes a small penalty for long answers, which comes up again later. We also compared against FlowRL, a method that tries to match the whole spread of rewards rather than just chase the top one. Our version of FlowRL was simplified (it leaves out a sequence-level importance weight), so read "FlowRL" below as "our variant of FlowRL".

### It learned faster

We ran each method with 5 random seeds for 100 updates. A seed sets a run's starting randomness, so different seeds are like independent reruns of the same experiment. Averaged over the whole run, the warm start's accuracy at depth 5 was 0.24 higher than GRPO's. By the end, it was at 0.95 against GRPO's 0.81 at depth 5, and 0.91 against 0.71 at depth 6. Neither depth was in the training data. To reach the accuracy GRPO finished with, the warm start needed about 2× less compute, or about 1.5× less if you leave out two weak GRPO seeds. None of the warm-start runs collapsed.

<figure>
<svg viewBox="0 0 400 290" width="100%" style="max-width:480px;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Bar chart. Depth 5 final accuracy: GRPO 0.81, warm start 0.95. Depth 6 accuracy: GRPO 0.71, warm start 0.91.">
<g font-family="Archivo, system-ui, sans-serif" font-size="15" fill="currentColor">
<rect x="150" y="9" width="13" height="13" rx="2" fill="currentColor" fill-opacity="0.45"/>
<text x="169" y="21">GRPO</text>
<rect x="238" y="9" width="13" height="13" rx="2" fill="#b8381f"/>
<text x="257" y="21">warm start</text>
</g>
<path d="M50 155 H370" stroke="currentColor" stroke-opacity="0.15" stroke-dasharray="3 4"/>
<path d="M50 250 H370" stroke="currentColor" stroke-opacity="0.5"/>
<g font-family="Archivo, system-ui, sans-serif" font-size="13" fill="currentColor" opacity="0.75">
<text x="42" y="159" text-anchor="end">0.5</text>
<text x="42" y="254" text-anchor="end">0</text>
<text x="10" y="50">accuracy</text>
</g>
<path d="M70 250 V100.1 Q70 96.1 74 96.1 H126 Q130 96.1 130 100.1 V250 Z" fill="currentColor" fill-opacity="0.45"><title>GRPO, depth 5 final: 0.81</title></path>
<path d="M134 250 V73.5 Q134 69.5 138 69.5 H190 Q194 69.5 194 73.5 V250 Z" fill="#b8381f"><title>Warm start, depth 5 final: 0.95</title></path>
<path d="M230 250 V119.1 Q230 115.1 234 115.1 H286 Q290 115.1 290 119.1 V250 Z" fill="currentColor" fill-opacity="0.45"><title>GRPO, depth 6: 0.71</title></path>
<path d="M294 250 V81.1 Q294 77.1 298 77.1 H350 Q354 77.1 354 81.1 V250 Z" fill="#b8381f"><title>Warm start, depth 6: 0.91</title></path>
<g font-family="Archivo, system-ui, sans-serif" fill="currentColor" text-anchor="middle" font-size="15">
<text x="100" y="88">0.81</text>
<text x="164" y="61">0.95</text>
<text x="260" y="107">0.71</text>
<text x="324" y="69">0.91</text>
<text x="132" y="274">depth 5, final</text>
<text x="292" y="274">depth 6</text>
</g>
</svg>
<figcaption>5 random seeds, 100 updates, model trained on depths 3–4.</figcaption>
</figure>

### It also beat our FlowRL, mostly on speed

It was faster than our FlowRL in every seed, with accuracy averaged over the run 0.22 higher. At the end the lead was smaller: 0.08 on average, and close to zero in 2 of the 5 seeds.

### The lead lasted, but shrank

To see whether GRPO would catch up, we ran 3 seeds for 200 updates. At depth 7, which the model never trained on, the warm start was ahead by 0.20 at update 100 and by 0.12 at update 200. It was still ahead in every seed, but GRPO was closing the gap. To reach GRPO's 200-update level, the warm start needed about 1.4× less compute.

### Measuring confidence line by line mattered

There's a simpler version of this idea: measure the model's confidence once, over the whole response. Intuitor (which rewards "self-certainty") and RENT (which rewards low entropy, meaning the model isn't spreading its bets) work that way. We tried both styles. Per-line confidence beat whole-response confidence by 0.13, and beat an Intuitor-style self-certainty signal by 0.15. Where you measure confidence turned out to matter.

## What didn't work, and why that's interesting

### Leave it on, and the model games it

When we kept the confidence nudge on for longer, runs shot up early and then collapsed. The model had found ways of writing that score well on confidence without actually solving the problem. Some runs started writing just the formula. Others substituted numbers in strange ways. It's like grading students on how confident they sound: sooner or later, they work on sounding confident. Turning the nudge off after 8 updates avoided this.

<figure>
<svg viewBox="0 0 400 252" width="100%" style="max-width:480px;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A sketch, not data: accuracy rises fast, the model learns to game the confidence signal, then accuracy collapses.">
<path d="M40 30 V220 H390" fill="none" stroke="currentColor" stroke-opacity="0.5"/>
<path d="M40 212 C 75 205, 95 85, 140 72 C 180 62, 215 66, 240 82 C 268 100, 282 188, 315 205 L 385 210" fill="none" stroke="#b8381f" stroke-width="2.5"/>
<g font-family="Archivo, system-ui, sans-serif" fill="currentColor" font-size="15">
<text x="104" y="180">fast at first</text>
<text x="190" y="32" text-anchor="middle">learns to game the confidence signal</text>
<text x="294" y="140">collapse</text>
<g font-size="13" opacity="0.75">
<text x="390" y="244" text-anchor="end">training updates →</text>
<text x="26" y="125" text-anchor="middle" transform="rotate(-90 26 125)">accuracy →</text>
</g>
</g>
<g fill="none" stroke="currentColor" stroke-width="1.2" opacity="0.6">
<path d="M102 168 L 90 151"/>
<path d="M190 40 V 58"/>
<path d="M291 136 L 281 145"/>
</g>
</svg>
<figcaption>A sketch, not data: with the confidence signal left on, runs rose fast, then the model gamed it and collapsed.</figcaption>
</figure>

### Starting on hard problems: nothing learned

We also tried training straight on depths 6 and 7, where the base model does no better than chance. After 100 updates, none of the methods had learned anything: not GRPO, not FlowRL, not the warm start. Running the warm start for longer here just got it gamed, the same way as above. Worse, every method started writing fewer step lines and got worse at single steps. Starting on the easy depths fixed this for all three methods. So in our runs, the warm start sped up a model that could already sometimes get it right. It couldn't get one going from zero.

### A transfer result that didn't survive a second seed

This one we were excited about. On one seed, we took the warm-started model and the GRPO model at the same accuracy on the training depths, and tested both on 12 variations of the task that neither had trained on. The warm-started model did better on 7 of them, by 0.11 to 0.19. Those 7 covered longer chains, a step with two inputs, extra chains that don't matter for the answer, and steps written out in words.

Then we ran a second seed. Only 2 of those 7 gains showed up again, and both were smaller: 0.07 when the variable names had their upper and lower case flipped, and 0.05 with the extra chains. The two-input result went the other way once both models were very accurate. So most of that advantage came down to one seed. We're including it because it's exactly the kind of result that's easy to publish after a single run.

### A boring explanation we ruled out

Plain GRPO tended to write fewer step lines. Since our reward has a small penalty for length, we wondered if that penalty was pushing GRPO to skip its working. It wasn't. Only about 7% of outputs ever got long enough for the penalty to kick in, so something else is behind it.

<figure>
<svg viewBox="0 0 400 364" width="100%" style="max-width:480px;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Held: faster learning over 5 seeds; a lead at untrained depth 7 after 200 updates over 3 seeds. Didn't hold: better transfer on a second seed, only 2 of 7 test sets; helping from a cold start on hard problems.">
<rect x="4" y="4" width="392" height="170" rx="6" fill="#b8381f" fill-opacity="0.08" stroke="#b8381f" stroke-width="2"/>
<rect x="4" y="190" width="392" height="170" rx="6" fill="none" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5" stroke-dasharray="6 5"/>
<g font-family="Archivo, system-ui, sans-serif" fill="currentColor">
<text x="22" y="36" font-size="18" font-weight="600">Held</text>
<text x="22" y="222" font-size="18" font-weight="600">Didn't hold</text>
<g font-size="15">
<text x="22" y="70">Faster learning</text>
<text x="22" y="92" opacity="0.75">5 seeds, 100 updates</text>
<text x="22" y="128">A lead at untrained depth 7</text>
<text x="22" y="150" opacity="0.75">200 updates, 3 seeds · shrinking, +0.20 → +0.12</text>
<text x="22" y="256">Better transfer on a second seed</text>
<text x="22" y="278" opacity="0.75">only 2 of 7 test sets held</text>
<text x="22" y="314">Helping from a cold start on hard problems</text>
<text x="22" y="336" opacity="0.75">depths 6–7 · no method learned there</text>
</g>
</g>
</svg>
<figcaption>What held up across seeds, and what did not.</figcaption>
</figure>

## What we think is going on

This part is our best guess, not something we've shown.

Early in training, the final-answer grade doesn't tell the model much yet. The model's own confidence fills in: it has something to say about every line, from the first update. That makes it a useful early teacher.

The trouble is that it's a teacher the student can fool. Once being sure is rewarded, the model starts working on being sure instead of being right. So we think the trick is to use it briefly: lean on it while the real reward is still quiet, then turn it off.

One more thing, and it's an anecdote, not a result. It comes from one seed and a handful of samples. Later in training, the warm-started model sometimes started working backward. It began from the variable it was asked about, traced back through its inputs to values it already knew, and then filled in the numbers going forward. We never taught it that, and we don't know if it's real, common, or anything to do with the warm start. It did surprise us.

## Limitations

- **One made-up task, one model size.** Everything here is variable chains on LFM2.5-2.6B-Base.
- **No standard benchmark.** We haven't tried GSM8K, MATH or anything like them.
- **Few comparisons.** We compared only with GRPO and our simplified FlowRL. Not with SPAE (Step Potential Advantage Estimation, the closest work we know of), not with other methods that give credit per step, and not with variants of PPO (Proximal Policy Optimization, the older RL method GRPO grew out of).
- **We haven't checked that confidence is what's doing the work.** Any per-step signal might help just as much. The obvious test is to grade each step by whether it's actually correct, and we haven't run it.
- **Seeds vary a lot.** Results on this task swing a lot from seed to seed, as the transfer result shows, and 5 seeds (3 for the longer runs) isn't many.

## What we'd test next

We don't know the answers to these yet:

- Does the speed-up hold on a standard benchmark like GSM8K or MATH?
- How does it do head to head against SPAE, which also uses intermediate confidence?
- Is it the confidence that helps, or just having some signal on every step? Grading steps by whether they're actually correct would tell us.

The closest work we know of is SPAE; we haven't seen this exact combination, but the area is crowded.

## Related work

- [Step Potential Advantage Estimation (SPAE): Harnessing Intermediate Confidence and Correctness for Efficient Mathematical Reasoning](https://arxiv.org/abs/2601.03823). The closest work we know of. It uses both intermediate confidence and correctness.
- [Learning from Own Solutions: Self-Conditioned Credit Assignment for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2606.18810)
- [Self-Credit: Credit Assignment for Long-Horizon Agentic RL via Intra-group Rescaling](https://arxiv.org/abs/2610.28388)
- [When Do Intrinsic Rewards Work for Code Reasoning? A Comprehensive Study](https://arxiv.org/abs/2606.20881)
- [Intuitor, "Learning to Reason without External Rewards"](https://arxiv.org/abs/2505.19590). Uses whole-response self-certainty as the reward.
- [RENT, "Maximizing Confidence Alone Improves Reasoning"](https://arxiv.org/abs/2505.22660). Uses entropy minimisation over the response as the reward.
- [FlowRL: Matching Reward Distributions for LLM Reasoning](https://arxiv.org/abs/2509.15207). Our comparison used a simplified variant.
