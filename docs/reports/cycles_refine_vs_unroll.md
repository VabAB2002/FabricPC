# Graphs with loops: refine with sPC, or unroll? (issue #64), first look

A graph with a loop can be settled by unrolling it (`graph(..., unroll=U)`) and relaxing with ePC, or by relaxing one visit with ePC and refining with T2 steps of sPC on the true loop energy (the composed schedule), or with plain sPC. Issue #64 asks which should be the default, measured. **These first CPU runs do not settle it**: which arm is cheapest changes with the trained weights. This report says what was measured and what a real answer needs.

## Setup

`python -m fabricpc.bench cycles --out DIR` (code: `fabricpc/bench/cycles.py`). Two MNIST graphs: `loop2`, the demo's loop (hidden2 -> back -> hidden1), and `loop4`, with four hidden nodes in the loop. The weights are trained first with the demo's sPC (rate 0.15, 50 steps, unroll 2); every arm then settles those weights on the same 5 test batches of 200 images. Energy is the true full-graph energy (every node at its latents, back edges included) per sample, with input and label clamped as in a training step; the anchor is sPC run for 2000 steps, and an arm "lands" when it is within 1% of it. Accuracy uses the same solver with the label free. Matmuls per sample count one per weighted edge per pass. Settle time is the median of jitted settles on a MacBook CPU. ePC rate 0.05.

Arms: unroll (U in 1 to 8, by ePC steps in 5, 20, 50, 200), refine (ePC 5 steps on one visit, then sPC T2 in 5 to 200), unroll 2 + refine, and plain sPC (20 to 500 steps).

## Results: cheapest setting that lands, per arm (one seed each)

| weights | arm | setting | matmuls / sample | settle ms |
|---|---|---|---|---|
| 1 epoch (loop2) | unroll | U 3, ePC 20 | 559 | 27 |
| | unroll 2 + refine | sPC 50 | 617 | 43 |
| | plain sPC | 100 steps | 1005 | 76 |
| | refine | sPC 100 | 1065 | 79 |
| 1 epoch (loop4) | unroll | U 2, ePC 20 | 559 | 23 |
| | unroll 2 + refine | sPC 50 | 869 | 44 |
| | plain sPC | 100 steps | 1407 | 68 |
| | refine | sPC 100 | 1491 | 78 |
| 0.25 epoch (loop2) | unroll 2 + refine | sPC 20 | 317 | 21 |
| | plain sPC | 50 steps | 505 | 37 |
| | refine | sPC 50 | 565 | 40 |
| | unroll | U 2, ePC 50 | 927 | 49 |

## What it shows, and what it does not

1. **The ranking depends on the weights.** With the 1-epoch weights, unrolling alone was cheapest; with the 0.25-epoch weights it was the most expensive arm that landed (the 1-epoch winner, U 3 with 20 ePC steps, does not land there), and unroll 2 with a short sPC refinement was cheapest. One seed of one weight set per graph cannot pick a default.
2. **One visit never lands** (U = 1, even at 200 ePC steps). ePC's step-boundary truncation on a loop (see `EPCInference.derive_states`) leaves a bias a single visit cannot remove. Unrolled ePC also keeps a small bias: on loop2 with the 1-epoch weights U 2 sits about 0.5% above the anchor even at 200 steps, so it lands only because the tolerance is 1%. sPC converges exactly.
3. **The accuracy column says little about settling.** Accuracy barely moves with the step count within an arm and follows the starting state (the feedforward pass along the unroll schedule): a reviewer found the settled prediction agrees with the start for about 98% of images, and sPC started from the unrolled pass gets the unrolled arms' accuracy. 1,000 images also give about ±0.8 points of sampling error.
4. **Cost counted in matmuls is rough.** Every matmul counts the same, but the 784 x 256 input edge is about 84% of the FLOPs; settle time is the better cost measure here.

## What a real answer needs

- The missing arm: sPC started from the unrolled feedforward pass (nearly free, and in a reviewer's check the cheapest to land).
- Several weight sets (trained with different solvers, unroll counts and lengths) and at least 5 seeds each, with the spread of the cost ratios.
- A tolerance sweep (0.25, 0.5, 1, 2%) and each arm's leftover energy bias.
- FLOP-based cost next to time, and accuracy next to the starting pass's accuracy.
- The result written as a versioned, validated benchmark row (manifest, seeds, contrasts), as #59 and #64 ask.

Raw numbers: `cycles.json` and `cycles.md` per run. MacBook CPU, 2026-10-10 (1-epoch weights) and 2026-10-11 (0.25-epoch weights).
