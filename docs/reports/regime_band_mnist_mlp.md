# Regime band: f-bar or f_max? A multi-seed check on the MNIST MLP (issue #73)

Issue #73 asks whether ePC's regime band should read f-bar (the gradient-weighted relaxed fraction) or f_max (the top mode's), and names what would settle it: more seeds, a second probe batch, and a second graph family. The ResNet-18 evidence in `epc_regime_and_stability_report.md` is one seed and one 64-sample probe batch. This report adds a second graph family at three seeds, on CPU. **It does not settle the question** (see "What it shows").

## Setup

`scripts/regime_band_study.py`, defaults: the `mnist-mlp-epc` benchmark row (784-256-64-10, sigmoid, softmax with cross-entropy, AdamW 0.001, batch 200) with `EPCInference(eta_infer=eta, infer_steps=5)` and the rate safety net off, so eta stays what was asked. eta in 0.001, 0.01, 0.03, 0.1, 0.3, 1; 3 seeds each (0, 1000, 2000); 2 epochs, which is a tenth of the row's own 20. Each trial runs through the benchmark's `run_trial`. f-bar and f_max are read at init on the first training batch and, separately, on the second, and again after training on the first.

```bash
python scripts/regime_band_study.py --out regime-band-study
```

## How to read f

For one mode with curvature lambda, T steps at rate eta relax a fraction f = 1 - (1 - eta lambda)^T of it. While eta lambda < 1 this lies between 0 and 1. For 1 < eta lambda < 2 and odd T (here T = 5) it exceeds 1: the step overshoots and the output gradient reverses. For eta lambda > 2 the iteration diverges and f grows without bound, so its mean over seeds means nothing. The table therefore gives medians and eta lambda_max, the quantity that decides stability (unstable above 2).

## Results (3 seeds; medians; eta lambda_max per seed at the end)

| eta | accuracy | eta lambda_max at init | eta lambda_max at end (3 seeds) | unstable at end | f-bar init (batch 1 / 2) | f_max init (batch 1 / 2) | f-bar end | f_max end |
|---|---|---|---|---|---|---|---|---|
| 0.001 | 0.9492 | 0.0014 | 0.005, 0.004, 0.005 | 0 of 3 | 0.006 / 0.006 | 0.007 / 0.007 | 0.016 | 0.024 |
| 0.01 | 0.9494 | 0.0145 | 0.057, 0.048, 0.052 | 0 of 3 | 0.058 / 0.059 | 0.070 / 0.070 | 0.154 | 0.234 |
| 0.03 | 0.9503 | 0.044 | 0.199, 0.163, 0.175 | 0 of 3 | 0.167 / 0.168 | 0.198 / 0.197 | 0.423 | 0.618 |
| 0.1 | 0.9493 | 0.145 | 0.773, 0.624, 0.677 | 0 of 3 | 0.471 / 0.473 | 0.539 / 0.539 | 0.830 | 0.996 |
| 0.3 | 0.9459 | 0.435 | 2.371, 1.874, 2.195 | **2 of 3** | 0.889 / 0.890 | 0.940 / 0.940 | 1.193 (overshoot) | 3.44 (diverging) |
| 1 | 0.9577 | 1.449 | 10.91, 7.52, 9.02 | 3 of 3 | 1.002 / 1.003 | 1.016 / 1.016 | diverging | diverging |

f-bar and f_max at init are means over seeds in the script's `summary.md`; the medians above come from the trial files in the output folder.

## What it shows

1. **The second probe batch agrees at init, with random weights.** f-bar and f_max on two different 200-sample batches match within 0.002 at every eta. This was measured at init only, and the second batch also used a different RNG key, so it shows the reading is not sensitive to the batch on an untrained MLP; it says nothing about trained networks.
2. **At init this graph cannot separate f-bar from f_max.** They move together (f_max about 1.1 to 1.2 times f-bar). At the end of training they come apart (f_max 1.5 to 1.7 times f-bar at eta 0.01 to 0.03, about 3 times at 0.3), but accuracy barely moves across that range, so the data cannot say which one accuracy follows.
3. **The stable range is eta 0.1 and below.** The stiffness grows during training (eta lambda_max rises about 4 to 5 times in 2 epochs). At eta 0.3 two of three seeds end unstable (eta lambda_max above 2) and accuracy is the lowest of the sweep (94.6%). At eta 1 all three end far past the bound yet score 95.8% after 2 epochs. Two epochs and no probe history during training cannot say whether instability would cost accuracy later, or whether Adam simply rescales the exploded ePC gradients; both need a longer run with the regime probed during training.

## What would settle the question

- A graph where f_max and f-bar come apart at init: the depth-varied linear chains of `scripts/epc_analysis.py` (the oracle gives both exactly), or the deep `mnist-fcresnet` rows, swept the same way (`--row`).
- The full 20 epochs with `RegimeProbe` recording f-bar, f_max and eta lambda_max during training, so a fall in accuracy can be lined up with each.
- An SGD-optimizer control, to test the "Adam absorbs it" explanation.
- The second probe batch measured after training too, with the same RNG key.

Raw numbers: `study.csv`, `summary.md` and `trials/` in the output folder. MacBook CPU, 2026-10-10, about 10 minutes.
