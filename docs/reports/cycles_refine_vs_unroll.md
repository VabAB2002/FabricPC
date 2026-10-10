# Graphs with loops: refine with sPC, or unroll? (issue #64)

A graph with a loop can be settled by unrolling it (`graph(..., unroll=U)`) and relaxing with ePC, or by a composed schedule that relaxes with ePC on one visit and refines with T2 steps of sPC on the true loop energy. Issue #64 asks which is the better default, measured, not argued.

## Setup

`python -m fabricpc.bench cycles --out DIR` (code: `fabricpc/bench/cycles.py`). Two MNIST graphs: `loop2`, the demo's loop (hidden2 -> back -> hidden1), and `loop4`, with four hidden nodes in the loop. Each is trained for 1 epoch with the demo's sPC (rate 0.15, 50 steps, unroll 2); every arm then settles the same weights on the same 5 test batches of 200 images. Energy is the true full-graph energy (every node at its latents, back edges included) per sample, with input and label clamped as in a training step; the anchor is sPC run for 2000 steps, and an arm lands when it is within 1% of it. Accuracy uses the same solver with the label free. Matmuls per sample count one per weighted edge per pass (ePC: a start pass plus two per step plus one to finish, over the visit schedule; sPC: two per step over the edges). Settle time is the median of jitted settles on a MacBook CPU. ePC rate 0.05 throughout.

Arms: unroll (U in 1, 2, 3, 4, 6, 8 by ePC steps in 5, 20, 50, 200), refine (ePC 5 steps on one visit, then sPC T2 in 5 to 200), unroll 2 + refine, and plain sPC (20 to 500 steps).

## Result: the cheapest setting that lands, per arm

## loop2: anchor energy 0.0853 (sPC, 2000 steps); lands within 1%
- unroll: unroll 3, ePC 20, sPC 0: 559 matmuls/sample, 26.91 ms, accuracy 0.9340
- refine: unroll 1, ePC 5, sPC 100: 1065 matmuls/sample, 78.69 ms, accuracy 0.9260
- unroll2+refine: unroll 2, ePC 5, sPC 50: 617 matmuls/sample, 43.10 ms, accuracy 0.9320
- spc: unroll 1, ePC 0, sPC 100: 1005 matmuls/sample, 75.58 ms, accuracy 0.9240
## loop4: anchor energy 0.1003 (sPC, 2000 steps); lands within 1%
- unroll: unroll 2, ePC 20, sPC 0: 559 matmuls/sample, 22.99 ms, accuracy 0.9240
- refine: unroll 1, ePC 5, sPC 100: 1491 matmuls/sample, 77.98 ms, accuracy 0.9210
- unroll2+refine: unroll 2, ePC 5, sPC 50: 869 matmuls/sample, 44.33 ms, accuracy 0.9240
- spc: unroll 1, ePC 0, sPC 100: 1407 matmuls/sample, 68.19 ms, accuracy 0.9210

## Default

**Unroll the loop 2 or 3 times and relax with ePC (about 20 steps).** On both graphs it lands for about half the matmuls of plain sPC (559 against 1005 and 1407 per sample) and a third of its time, and it scores equal or higher accuracy (93.4% against 92.4% on loop2). One visit (U = 1) never lands, even at 200 ePC steps: a single traversal misses the loop. The composed schedule (ePC on one visit, then sPC) is no cheaper than plain sPC, because the sPC refinement does almost all the work; it is the path to use only when the loop cannot be unrolled. Unrolling 2 times and refining briefly with sPC lands at 617 to 869 matmuls, between the two.

## Limits

One seed and one set of trained weights per graph, two hand-built loops on MNIST, CPU timing, one ePC rate (0.05). Deeper or longer loops, the fuzzer's cyclic graphs (issue #62), a GPU, and more seeds would test how far the default holds. Run on 2026-10-10 in about 3 minutes; raw numbers in `cycles.json`.
