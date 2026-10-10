# Regime band: f-bar or f_max? A multi-seed check on the MNIST MLP (issue #73)

Issue #73 asks whether ePC's regime band should read f-bar (the gradient-weighted relaxed fraction) or f_max (the top mode's), and names what would settle it: more seeds, a second probe batch, and a second graph family. The ResNet-18 evidence in `epc_regime_and_stability_report.md` is one seed and one 64-sample probe batch. This report adds a second graph family at several seeds, on CPU.

## Setup

`scripts/regime_band_study.py`, defaults: the `mnist-mlp-epc` benchmark row (784-256-64-10, sigmoid, softmax with cross-entropy, AdamW 0.001, batch 200) with `EPCInference(eta_infer=eta, infer_steps=5)` and the rate safety net off, so eta stays what was asked. eta in 0.001, 0.01, 0.03, 0.1, 0.3, 1; 3 seeds each (0, 1000, 2000, the benchmark's); 2 epochs. Each trial runs through the benchmark's `run_trial`. f-bar and f_max are read at init on the first training batch and, separately, on the second, and again after training on the first. Command:

```bash
python scripts/regime_band_study.py --out regime-band-study
```

## Results (means over 3 seeds; accuracy ± SD)

| eta | n | accuracy | f_bar init (batch 1 / 2) | f_max init (batch 1 / 2) | f_bar end | f_max end |
|---|---|---|---|---|---|---|
| 0.001 | 3 | 0.9490 ± 0.0003 | 0.006 / 0.006 | 0.007 / 0.007 | 0.015 | 0.024 |
| 0.01 | 3 | 0.9494 ± 0.0002 | 0.058 / 0.059 | 0.070 / 0.070 | 0.150 | 0.236 |
| 0.03 | 3 | 0.9503 ± 0.0004 | 0.167 / 0.168 | 0.198 / 0.197 | 0.413 | 0.625 |
| 0.1 | 3 | 0.9496 ± 0.0011 | 0.471 / 0.473 | 0.539 / 0.539 | 0.835 | 0.996 |
| 0.3 | 3 | 0.9461 ± 0.0012 | 0.889 / 0.890 | 0.940 / 0.940 | 1.191 | 3.595 |
| 1 | 3 | 0.9577 ± 0.0009 | 1.002 / 1.003 | 1.016 / 1.016 | 6422.563 | 46843.077 |

## What it shows

1. **The probe batch does not matter on this graph.** f-bar and f_max at init on two different 200-sample batches agree to within 0.002 at every eta. One of the issue's limits (a single probe batch) is not what moves the reading here.
2. **This graph cannot separate f-bar from f_max.** On the MLP the two move together, f_max about 1.1 to 1.2 times f-bar at every eta, so any accuracy trend follows both equally. The separation the ResNet-18 showed needs most of the gradient weight sitting in low-curvature modes while a few stiff modes relax first; a 3-layer MLP does not have that spread.
3. **Accuracy is flat across the stable range.** From eta 0.001 (backprop-like, f-bar 0.006) to 0.3 (f-bar 0.89) accuracy stays within half a point (94.6% to 95.0%); Adam absorbs the hidden-layer rescaling. At eta 1 the run ends unstable (eta times lambda_max above 2: f-bar and f_max far above 1 at the end) yet scores 95.8% after 2 epochs, so on this graph instability shows in the spectrum before it shows in accuracy.

## What would settle the question

A graph where f_max and f-bar come apart: the depth-varied linear chains of `scripts/epc_analysis.py` (the oracle gives both exactly), or a deeper network (the benchmark's `mnist-fcresnet` rows), swept the same way. The script takes `--row`, so a deep row can be swept with the same command once its cost allows (a GPU for depth 32 and up).

Raw numbers: `study.csv` and `summary.md` in the output folder. Run on a MacBook CPU, 2026-10-10, about 10 minutes.
