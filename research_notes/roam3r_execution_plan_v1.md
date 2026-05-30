# RoAM3R Execution Plan v1

Date: 2026-05-13
Goal: turn the RoAM3R idea into a paper-grade prototype and evidence package
Base repo: LingBot-MAP

## 1. Objective

Build the smallest credible prototype that can support the following paper claim:

- a streaming reconstruction model can be made substantially more robust to projection shift and temporal sparsity through projection-aware geometry, adaptive memory, and online recovery

This plan intentionally does not target a full polished system in the first phase.
It targets evidence.

## 2. Success Criteria

The project is worth continuing only if it clears these gates:

### Gate A: problem is real

Baseline LingBot must show clear degradation under:

- temporal subsampling
- 360-derived or wide-FoV inputs
- viewpoint discontinuity stress

If the baseline does not degrade meaningfully, the paper framing weakens.

### Gate B: at least one module yields a strong signal

At least one of these should produce a measurable gain:

- ray camera interface
- adaptive memory
- relocalization

If all three modules produce tiny or noisy gains, the paper should be reframed.

### Gate C: dense-video regression stays controlled

The prototype must not collapse on standard perspective-video benchmarks.

Some regression is acceptable.
Large regression is not.

## 3. Core Hypotheses

### H1

Projection-aware ray conditioning will improve robustness on wide-FoV and 360-derived sequences.

### H2

Overlap-aware retrieval will slow performance collapse under low-FPS and interval-sampled inputs.

### H3

Uncertainty-gated relocalization will reduce catastrophic failure after viewpoint breaks.

## 4. Phase Plan

## Phase 0: baseline evidence

Duration:

- 2 to 4 days

Goal:

- prove the boundary with hard numbers

Deliverables:

- baseline temporal subsampling results
- baseline projection-shift results
- baseline viewpoint-break results

Decision:

- if the baseline already survives all stress tests, the current paper story is weak

## Phase 1: evaluation harness

Duration:

- 3 to 5 days

Goal:

- create reproducible scripts and metrics for all stress axes

Deliverables:

- temporal subsampling pipeline
- projection-shift evaluation pipeline
- viewpoint-break evaluation pipeline
- table-ready CSV outputs

Decision:

- if evaluation remains noisy or non-reproducible, stop method expansion until fixed

## Phase 2: camera interface prototype

Duration:

- 1 to 2 weeks

Goal:

- add a minimal projection-aware geometry pathway

Deliverables:

- ray descriptor generation
- backward-compatible perspective path
- first wide-FoV robustness ablation

Decision:

- if no gain appears on projection-shift evaluation, revisit the paper emphasis

## Phase 3: adaptive memory prototype

Duration:

- 1 to 2 weeks

Goal:

- replace recency-only memory with overlap-aware retrieval

Deliverables:

- keyframe scoring
- retrieval policy
- low-FPS and interval-capture ablations

Decision:

- if gains appear only on synthetic subsampling but not on real interval captures, narrow the claim

## Phase 4: relocalization prototype

Duration:

- 1 to 2 weeks

Goal:

- reduce catastrophic breakdown

Deliverables:

- uncertainty/risk score
- relocalization trigger
- recovery metrics

Decision:

- if recovery reduces average error but not failure rate, the module story is weak

## Phase 5: full integration

Duration:

- 1 week

Goal:

- evaluate the full RoAM3R system

Deliverables:

- full ablation table
- qualitative figures
- submission-ready claim-evidence table

## 5. Dataset and Evaluation Plan

## 5.1 Local pilot data already available

Use current local evidence first.

### Perspective video

- `/home/slam/datasets/zhuyevideos/video_20260418_190605(1).mp4`

Use this for:

- dense baseline
- FPS subsampling
- viewpoint-break simulation

### 360 still captures

- `/media/slam/My Passport/360imgs/DCIM/Camera01/outdoor`
- `/media/slam/My Passport/360imgs/DCIM/Camera01/outdoor_perspectives_8`

Use this for:

- projection-shift pilot
- interval-style evaluation

## 5.2 Benchmark families to build

### Family A: dense perspective

Question:

- do we preserve the original strong regime?

### Family B: temporal subsampling

Variants:

- full fps
- 10 fps
- 5 fps
- 2 fps
- 1 fps

Question:

- how sharply does each method degrade?

### Family C: interval captures

Variants:

- true interval images
- synthetic long-gap frame subsets

Question:

- does adaptive retrieval beat recency-based streaming?

### Family D: projection shift

Variants:

- perspective
- wide-FoV or fisheye
- 360-derived perspective

Question:

- does camera-aware geometry improve robustness?

### Family E: recovery stress

Variants:

- abrupt turn
- dropped segment
- occlusion
- repeated texture

Question:

- can the system recover instead of drifting blindly?

## 6. Exact Baseline Experiments to Run First

These are the first experiments that should be run before major model refactors.

### E0.1 Temporal collapse curve

Input:

- one or more normal perspective videos

Runs:

- baseline LingBot at dense, `10`, `5`, `2`, and `1` FPS

Record:

- ATE
- RPE
- depth error
- completeness
- runtime

Expected signal:

- steady but significant degradation as FPS decreases

### E0.2 Projection mismatch pilot

Input:

- normal perspective sequence
- 360-derived perspective sequence from same or similar environment

Runs:

- same baseline settings where possible

Record:

- geometry quality
- trajectory stability
- qualitative distortion

Expected signal:

- large robustness gap between perspective and 360-derived inputs

### E0.3 Viewpoint-break stress test

Input:

- normal video with artificial frame-drop segments or abrupt-turn splice

Record:

- failure rate
- recovery or non-recovery
- post-break ATE

Expected signal:

- catastrophic failure cases appear more clearly than in average metrics

## 7. Code Attack Plan

## 7.1 Evaluation harness first

Add or extend scripts under:

- `scripts/`

Suggested scripts:

- `scripts/eval_temporal_subsampling.sh`
- `scripts/eval_projection_shift.sh`
- `scripts/eval_viewpoint_break.sh`

Outputs:

- per-run logs
- CSV summary
- figure-ready artifacts

## 7.2 Camera interface implementation

Primary files:

- `lingbot_map/utils/pose_enc.py`
- `lingbot_map/utils/geometry.py`
- `lingbot_map/heads/camera_head.py`
- `lingbot_map/models/gct_base.py`

Minimal prototype path:

1. add a camera-mode abstraction
2. add ray descriptor utilities
3. keep original perspective path as fallback
4. inject ray descriptors before geometry reasoning

## 7.3 Adaptive memory implementation

Primary files:

- `lingbot_map/models/gct_stream.py`
- `lingbot_map/models/gct_stream_window.py`
- `lingbot_map/layers/attention.py`
- `lingbot_map/layers/flashinfer_cache.py`

Minimal prototype path:

1. score cached frames with overlap and uncertainty
2. maintain a fixed-capacity candidate set
3. retrieve non-local support when local support is weak

## 7.4 Relocalization implementation

Primary files:

- `lingbot_map/models/gct_stream.py`
- `lingbot_map/models/gct_stream_window.py`
- add a new helper module if needed under `lingbot_map/models/` or `lingbot_map/utils/`

Minimal prototype path:

1. compute instability signal
2. threshold or rank risk
3. retrieve anchor frame(s)
4. reattach streaming state

## 8. Metrics Package

Use the same metrics across all phases whenever possible.

### Trajectory

- ATE
- RPE

### Geometry

- depth error
- point accuracy
- point completeness

### Stability

- failure rate
- recovery success rate
- post-break error

### Efficiency

- FPS
- peak GPU memory
- wall-clock runtime

## 9. Ablation Matrix

The minimum paper-grade ablation matrix should be:

1. LingBot baseline
2. `+` ray camera interface
3. `+` adaptive memory
4. `+` relocalization
5. full RoAM3R

Secondary ablations:

- mixed-camera training vs perspective-only training
- mixed-FPS training vs dense-only training
- local-only memory vs retrieval memory
- no-risk-trigger vs heuristic trigger vs learned trigger

## 10. Kill Criteria

Stop or pivot if:

1. projection-shift evaluation is too noisy to support a clean claim
2. local 360 data is too weak for meaningful comparison and no better dataset can be added
3. camera-aware prototype gives negligible benefit
4. retrieval memory increases complexity but produces no temporal-robustness gain
5. relocalization improves visuals only but not measurable failure rate

## 11. What to Write in Parallel with Experiments

Do not wait for final results to write.

Write in parallel:

1. benchmark protocol section
2. method figure sketch
3. ablation table skeleton
4. related-work notes
5. failure-case figure panels

This reduces writing risk and exposes missing evidence early.

## 12. Best-Case Outcome

Best-case result pattern:

1. dense perspective performance remains near LingBot
2. low-FPS degradation curve is clearly flatter
3. wide-FoV or 360-derived results improve materially
4. failure rate after viewpoint breaks drops noticeably

If that pattern appears, the paper is strong enough to target a top-tier venue.

## 13. Most Valuable Immediate Next Step

The single most valuable next step is not implementing all three modules at once.

It is:

- building the evaluation harness and proving the baseline boundary with hard numbers

Without that evidence, the paper remains plausible but ungrounded.
