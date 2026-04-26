# Research Plan: Depth-Uncertainty Weighting in Mono SLAM

## Goal
Quantify how different depth-uncertainty signals affect mono ORB-SLAM accuracy. The eventual contribution is a learned covariance-prediction model; the experimental scaffold isolates whether such uncertainty signals can improve mono SLAM at all.

## Experimental design: four configurations
All differ *only* in how the per-keypoint scalar weight (`aux_depth` → `aux_uncertainty`) is computed. Sequences, RNG seed, optimizer, feature extractor, everything else: identical.

1. **Vanilla mono ORB-SLAM** — baseline, no per-keypoint weighting
2. **Mono + GT depth-derived uncertainty** — sanity check that the *weighting recipe* works at all
3. **Mono + monocular depth network** — uncertainty derived from a depth net (e.g. Metric3D-v2)
4. **Mono + covariance-prediction model** — research contribution

Config 2 (GT) is the linchpin: if a perfect depth oracle doesn't beat vanilla, the recipe is the bottleneck and no prediction model can rescue it. Config 2 sets the upper bound; configs 3 and 4 are evaluated against it, not just against vanilla.

All four hit the same architectural seam: per-keypoint scalar weights injected into the optimizer's information matrix at the four sites in `optimizer_g2o.py`. Adding a fifth config is a one-function change.

## The headroom problem (project risk)
On easy sequences, vanilla ORB-SLAM2 is at ceiling — sub-decimeter ATE leaves no room for *any* uncertainty method to demonstrate improvement. Null results on easy data mean the experiment is uninterpretable, not that the method failed.

Sanity checks confirmed: random per-keypoint weights only occasionally degrade ATE on abandonedfactory/Easy/P000. Huber kernel + ~1200 redundant features per frame + level-0 octave dominance absorb perturbations.

### Where uncertainty *should* matter
- Long trajectories with severe scale drift (mono accumulates error → late-trajectory points have huge depth uncertainty → upweighting is actively harmful)
- Sparse-feature / low-texture scenes
- Scenes with extreme depth ranges (close + far → Jacobian propagation differs by orders of magnitude)
- Dynamic content (down-weight moving objects)
- Initialization-critical moments (mono init is fragile; depth priors could stabilize)
- Loop closure (Sim(3) on noisy distant matches)

### Sequence selection criteria
Target sequences where vanilla ATE ∈ [0.3, 2m] OR has nontrivial failure rate. Pick from prior work (ORB-SLAM2/3 papers, TartanAir benchmark) — challenging-by-others is challenging-here.

## Metrics to report (always)
- ATE on successful runs (mean ± std over N≥5 reps)
- Success rate (fraction completing without lost tracking)
- Per-frame error tail (max, p99) — method may not improve mean but may reduce tail
- Never single-run numbers

## If full-trajectory ATE shows no gap, pivot the angle
- Initialization quality (success rate of mono init across seeds)
- Map quality (number/distribution of triangulated points, depth accuracy)
- Outlier rejection rate
- Robustness to perturbation (image noise, motion blur, exposure)
- Convergence speed (optimizer iterations to reach a tolerance)

All publishable findings if methodology is clean.

## Scope decisions
- pyslam ≡ ORB-SLAM2 (not ORB-SLAM3) — intentional, mono-only scope
- Python core (USE_CPP_CORE=False) — slower but adequate; C++ pybind extension is a future TODO if speed becomes limiting
- TartanAir as primary benchmark
- Pre-compute depth + uncertainty predictions per (model, dataset, sequence) in `uncertainty_estimation`, save to disk, load via dataset wrapper — cleaner ablations, cheaper sweeps

## Methodology rules
- One change per run
- N≥5 reps minimum per cell
- Seed RNG (SEED=42) for partial determinism
- Mean ± std + success rate in every report
- Negative results are publishable if the experiment was clean — the risk isn't "method doesn't work," it's "experiment is uninterpretable"
