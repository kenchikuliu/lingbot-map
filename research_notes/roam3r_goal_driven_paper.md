# RoAM3R: A Goal-Driven Top-Tier Paper Idea Built from LingBot-MAP

Date: 2026-05-13
Base project: LingBot-MAP
Working mode: goal-driven research design

## 1. Executive Summary

This note turns LingBot-MAP from a runnable project into a concrete top-tier paper direction.

The main conclusion is:

- LingBot-MAP is strong as a feed-forward streaming 3D reconstruction model for continuous perspective video.
- Its most important unsolved boundary is not raw accuracy on standard benchmarks, but robustness under projection shift and temporal shift.
- The best next paper is not "make LingBot a bit bigger", and not "hallucinate unseen views".
- The best next paper is a camera-agnostic, temporally robust streaming mapping model that can handle perspective, fisheye, and 360-derived inputs while remaining stable under low-FPS and interval captures.

This note proposes that paper under the working title:

- RoAM3R: Robust Camera-Agnostic Streaming 3D Reconstruction

## 2. Goal-Driven Problem Statement

### Goal

Build a streaming 3D reconstruction system that remains geometrically stable when the input departs from the "continuous perspective video" regime.

### Why this matters

Real deployment data is not always:

- pinhole or near-pinhole
- temporally dense
- smoothly moving
- continuously overlapping

Many practical captures are instead:

- fisheye or wide-FoV
- 360-derived perspective cuts
- low-FPS videos
- interval still captures
- trajectories with abrupt viewpoint jumps

Existing streaming models are still disproportionately optimized for the easiest regime: monocular perspective video with dense temporal continuity.

### Precise gap

The open problem is:

- How do we make feed-forward streaming reconstruction robust to both camera projection shift and temporal continuity shift without falling back to expensive global optimization?

That is a cleaner and more defensible paper target than "general SLAM".

## 3. Evidence from This Reproduction

These observations were obtained while reproducing LingBot-MAP in this workspace.

### What worked

- Standard monocular perspective video ran successfully after replacing OpenCV frame extraction with an `ffmpeg -> image_folder` workflow.
- The video `/home/slam/datasets/zhuyevideos/video_20260418_190605(1).mp4` reconstructed reliably at:
  - `1 fps`, `210` frames
  - `2 fps`, `421` frames

### What failed qualitatively

- Insta360 X5 raw `.insp` captures were not a good fit for the model.
- Conservative center crops could run.
- Perspective-converted cuts could also run.
- But the reconstruction quality remained clearly worse than the normal video path.

### Why this matters scientifically

This is useful pilot evidence because it separates:

- runtime engineering issues
- from method boundary issues

The 360 failure mode is not mainly "the repo is broken". It is much closer to:

- camera-model mismatch
- preprocessing mismatch
- temporal-overlap mismatch

That gives a real, observed boundary from which the next paper can grow.

## 4. Boundary Review of LingBot-MAP

### Core claim of the current project

LingBot-MAP is a feed-forward model for streaming 3D reconstruction with compact streaming state and long-horizon inference.

### Where the claim holds

- perspective imagery
- continuous video
- high overlap between adjacent frames
- streaming reconstruction where forward progress is more important than explicit global recovery

### Where the claim weakens

- fisheye and non-rectilinear optics
- raw 360 or dual-fisheye imagery
- low-FPS or interval-sampled sequences
- wide-baseline still captures
- abrupt viewpoint transitions
- cases requiring strong relocalization or explicit recovery

### Why the current implementation suggests this boundary

The public code is centered around a narrow camera representation and pinhole geometry:

- pose encoding uses translation, quaternion, and FoV
- intrinsics are reconstructed from image size plus FoV
- depth unprojection and projection are built around pinhole assumptions
- preprocessing relies on canonical crop/pad logic
- streaming memory is optimized mainly for efficient causal operation, not robust global retrieval under temporal breakage

## 5. Literature Slot

This proposal is timely because recent work has split into two largely separate directions.

### Streaming and long-horizon reconstruction

- LingBot-MAP focuses on streaming reconstruction with compact geometric context and long-range memory.
- LONG3R focuses on recurrent long-sequence streaming with spatio-temporal memory.
- InfiniteVGGT focuses on infinite-horizon streaming via rolling memory.
- PAS3R focuses on pose-adaptive updates for long videos.

### Camera-model generalization

- CAM3R tackles camera-agnostic reconstruction with ray-aware modeling.
- Wid3R tackles wide-FoV reconstruction with camera-model conditioning.

### The missing intersection

What is still not cleanly solved is the intersection of both axes:

- streaming
- long-horizon
- camera-agnostic
- robust under sparse temporal continuity
- equipped with explicit recovery behavior

That intersection is where this paper should sit.

## 6. Paper Thesis

### One-sentence thesis

Streaming 3D reconstruction should be robust not only to long duration, but also to projection shift and temporal sparsity.

### Strong version of the claim

We introduce the first feed-forward streaming 3D reconstruction framework that jointly addresses:

- camera model shift across perspective, fisheye, and 360-derived imagery
- temporal continuity shift across dense video, low-FPS video, and interval captures
- online failure recovery without expensive full global optimization

### Safer reviewer-facing claim

We extend streaming 3D reconstruction from continuous perspective video to camera-agnostic and temporally sparse observations, while preserving real-time or near-real-time inference.

The safer claim is more believable and easier to defend.

## 7. Proposed Method: RoAM3R

RoAM3R should have three method contributions and one evaluation contribution.

### 7.1 Unified Ray Camera Interface

#### Problem

LingBot-style FoV encoding is too narrow for wide-FoV and distorted cameras.

#### Design

Replace the camera representation with a unified ray interface:

- each pixel or patch is associated with a viewing ray
- camera metadata is represented through ray direction statistics or learned ray tokens
- projection type is not baked into a single FoV parameterization

#### Practical implementation

- Introduce a ray-generation stage before geometry reasoning
- Replace FoV-only pose encoding with:
  - pose token
  - camera-model token
  - ray field summary or patch-level ray embeddings

#### Why this is better than just adding distortion coefficients

Adding distortion parameters still keeps the model architecturally centered on pinhole projection.
Ray-based conditioning makes the geometry pathway itself projection-aware.

### 7.2 Overlap-Aware Adaptive Memory

#### Problem

Current streaming memory policies are still too tied to temporal adjacency.

#### Design

Use adaptive memory that selects or retrieves keyframes by overlap, novelty, and confidence instead of only recency.

Each incoming frame computes:

- geometric novelty score
- overlap score with cached keyframes
- uncertainty score

The model then decides whether to:

- attach to local recent context
- retrieve a more distant but more overlapping keyframe
- promote the current frame to memory

#### Main intuition

Dense video and interval stills should not use the same memory policy.

### 7.3 Uncertainty-Gated Relocalization

#### Problem

Streaming models often degrade silently after viewpoint breaks.

#### Design

Add an explicit online recovery branch:

- estimate tracking risk from geometric inconsistency and prediction uncertainty
- if risk is high, trigger retrieval from global memory
- perform lightweight relocalization against stored high-value keyframes
- resume streaming updates from the recovered state

#### Key point

This does not need full classical pose graph optimization.
It only needs a reliable "do not keep drifting blindly" mechanism.

### 7.4 New Evaluation Protocol

The paper needs an evaluation contribution, even if not a brand-new dataset.

Use a benchmark protocol with controlled stress axes:

- projection shift
- temporal sparsity
- viewpoint discontinuity

This can be built from existing datasets plus controlled subsampling and re-projection recipes.

## 8. Architecture Sketch

### Input pathway

1. image encoder
2. ray-aware camera adapter
3. streaming geometry backbone

### Memory pathway

1. local recent memory
2. global retrieval memory
3. uncertainty-aware keyframe promotion

### Recovery pathway

1. risk estimator
2. relocalization trigger
3. lightweight alignment update

### Outputs

- camera pose
- depth or pointmap
- confidence
- recovery state

## 9. What Is Actually Novel Relative to Existing Papers

This is the most important part. Without this section, the project will collapse into "LingBot plus some tricks".

### Relative to LingBot-MAP

Novel because:

- LingBot is streaming, but not camera-agnostic
- LingBot is efficient, but still assumes dense temporal continuity
- LingBot lacks explicit recovery behavior

### Relative to LONG3R / InfiniteVGGT / PAS3R

Novel because:

- those works mainly attack long-horizon stability
- they do not center the projection-shift problem
- they are not framed as camera-agnostic streaming reconstruction

### Relative to CAM3R / Wid3R

Novel because:

- those works attack camera-model generalization
- but not streaming long-horizon reconstruction with recovery under temporal sparsity

### Relative to Spann3R / SLAM3R

Novel because:

- those works emphasize spatial memory or global alignment
- but not the combined problem of camera-model shift plus sparse-temporal streaming robustness

### The actual novelty statement

The paper is not:

- "a better camera model only"
- "a better streaming memory only"

It is:

- the first robust streaming reconstruction system designed for the joint boundary of projection shift and temporal shift

That is the angle that can make it top-tier rather than incremental.

## 10. Minimal Code-Level Attack Plan in This Repo

This is the shortest path to a prototype using the current LingBot-MAP codebase.

### Stage A: Camera representation refactor

Touch first:

- `lingbot_map/utils/pose_enc.py`
- `lingbot_map/utils/geometry.py`
- `lingbot_map/heads/camera_head.py`
- `lingbot_map/models/gct_base.py`

Deliverables:

- new camera representation abstraction
- ray-aware unprojection and projection utilities
- backward compatibility for original perspective mode

### Stage B: Adaptive memory and retrieval

Touch next:

- `lingbot_map/models/gct_stream.py`
- `lingbot_map/models/gct_stream_window.py`
- memory-related cache utilities in `lingbot_map/layers/`

Deliverables:

- overlap-aware keyframe scoring
- retrieval from non-local memory
- memory promotion and pruning policy

### Stage C: Recovery behavior

Add:

- uncertainty or failure score head
- relocalization trigger logic
- keyframe reattachment logic

Deliverables:

- explicit failure state
- recovery metrics

## 11. Experiment Plan

## 11.1 Main experimental question

Can one streaming reconstruction model remain robust when both the camera model and the temporal sampling regime shift away from the standard perspective-video setting?

## 11.2 Evaluation suites

### Suite A: Dense perspective video

Purpose:

- verify that the new method does not damage the original strong regime

### Suite B: Temporal subsampling

Take normal videos and evaluate:

- 30 fps
- 10 fps
- 5 fps
- 2 fps
- 1 fps

Purpose:

- measure performance collapse under decreasing continuity

### Suite C: Interval captures

Use still-image sequences captured at intervals rather than continuous video.

Purpose:

- test whether adaptive retrieval beats recency-based memory

### Suite D: Wide-FoV and 360-derived inputs

Evaluate:

- perspective
- fisheye
- panoramic or 360-derived perspective cuts

Purpose:

- verify camera-model robustness

### Suite E: Recovery stress test

Insert:

- abrupt turns
- repeated textures
- transient occlusion
- dropped frame segments

Purpose:

- test relocalization and recovery

## 11.3 Baselines

Primary baselines:

- LingBot-MAP
- LONG3R
- InfiniteVGGT
- PAS3R
- SLAM3R

Camera-generalization baselines:

- CAM3R
- Wid3R

Supporting references:

- LingBot-MAP: https://arxiv.org/abs/2604.14141
- LONG3R: https://arxiv.org/abs/2507.18255
- InfiniteVGGT: https://arxiv.org/abs/2601.02281
- PAS3R: https://arxiv.org/abs/2603.21436
- SLAM3R: https://arxiv.org/abs/2412.09401
- CAM3R: https://arxiv.org/abs/2603.22631
- Wid3R: https://arxiv.org/abs/2602.05321
- Spann3R: https://arxiv.org/abs/2408.16061

## 11.4 Metrics

Use hard metrics, not only visualizations.

- camera trajectory: ATE, RPE
- depth or geometry: depth error, point accuracy, completeness
- reconstruction stability: failure rate, recovery success rate
- efficiency: latency, memory, throughput

## 11.5 Ablations

Required ablations:

1. baseline LingBot
2. + ray camera interface only
3. + adaptive memory only
4. + relocalization only
5. full model

Critical cross-ablation:

- perspective only training vs mixed-camera training
- dense-only training vs mixed temporal sampling training

## 12. Claim-Evidence Map

### Claim 1

RoAM3R preserves strong performance on standard perspective streaming benchmarks.

Evidence needed:

- parity or near-parity on dense perspective video against LingBot

### Claim 2

RoAM3R is substantially more robust under temporal sparsity.

Evidence needed:

- slower degradation under FPS subsampling
- better performance on interval captures

### Claim 3

RoAM3R generalizes to wide-FoV and 360-derived inputs better than perspective-centered streaming models.

Evidence needed:

- clear gains over LingBot, LONG3R, PAS3R on fisheye and 360-derived evaluation

### Claim 4

RoAM3R reduces catastrophic drift after viewpoint disruptions.

Evidence needed:

- lower failure rate
- higher recovery success rate
- better post-break trajectory consistency

## 13. Likely Reviewer Attacks

These need to be addressed early.

### Attack 1

This is just LingBot plus camera conditioning.

Response path:

- show the paper solves the joint problem of projection shift and temporal shift
- show memory and recovery are essential, not optional

### Attack 2

CAM3R and Wid3R already do camera-agnostic reconstruction.

Response path:

- those are not the same problem as long-horizon streaming with recovery

### Attack 3

This is engineering, not science.

Response path:

- define the new problem clearly
- show systematic failure boundaries
- provide controlled stress tests that isolate each failure source

### Attack 4

The gains might come from more data or broader augmentation.

Response path:

- include training-data controlled ablations
- isolate the benefit of architecture and policy changes

## 14. Why This Is Better Than a View-Completion Paper

A tempting alternative is to focus on unseen-view completion or hallucinating missing geometry.

That is weaker as the main top-tier story for this project because:

- it shifts the task from reconstruction to generation
- it introduces ambiguity about truthfulness
- it is much harder to defend geometrically
- it becomes easy for reviewers to argue that the model is just inventing plausible structure

Novel view completion can still appear as:

- an auxiliary visualization
- a downstream application
- a secondary branch

But it should not be the paper's main claim.

## 15. Proposed Title Options

Preferred title:

- RoAM3R: Robust Camera-Agnostic Streaming 3D Reconstruction

Alternative titles:

- Beyond Perspective Video: Robust Streaming 3D Reconstruction under Projection and Temporal Shift
- Camera-Agnostic Streaming 3D Reconstruction with Adaptive Memory and Online Relocalization
- Robust Streaming Mapping for Wide-FoV and Sparse-Temporal Observations

## 16. Draft Abstract

Streaming 3D reconstruction has recently advanced through feed-forward models that process monocular video in real time. However, existing methods remain strongly biased toward continuous perspective video, and often degrade under projection shift, temporal sparsity, and abrupt viewpoint changes. We present RoAM3R, a robust camera-agnostic streaming 3D reconstruction framework designed to operate beyond the standard perspective-video regime. RoAM3R introduces a unified ray camera interface that replaces narrow FoV-centric camera encoding, enabling a single model to process perspective, fisheye, and 360-derived observations. To handle sparse or discontinuous temporal input, we further propose an overlap-aware adaptive memory that retrieves informative historical keyframes based on geometric novelty, confidence, and overlap rather than temporal recency alone. Finally, RoAM3R incorporates an uncertainty-gated online relocalization mechanism that detects unstable updates and recovers the streaming state without expensive global optimization. We evaluate RoAM3R under three controlled axes of difficulty: projection shift, temporal subsampling, and viewpoint discontinuity. Experiments show that RoAM3R preserves strong performance on standard perspective streaming benchmarks while substantially improving robustness on wide-FoV imagery, low-FPS sequences, and interval captures. These results suggest that robust streaming reconstruction requires jointly modeling camera geometry, memory, and recovery behavior rather than optimizing any single axis in isolation.

## 17. Paper Outline

### 1. Introduction

- Streaming 3D reconstruction is useful for live perception.
- Existing methods mainly assume continuous perspective video.
- Real deployment data violates both the camera assumption and the temporal continuity assumption.
- We propose a robust camera-agnostic streaming model.

### 2. Related Work

- feed-forward 3D reconstruction
- streaming reconstruction
- camera-agnostic and wide-FoV reconstruction
- memory and relocalization in online geometry systems

### 3. Method

- unified ray camera interface
- overlap-aware adaptive memory
- uncertainty-gated relocalization

### 4. Experiments

- dense video
- temporal subsampling
- interval still capture
- wide-FoV and 360-derived data
- recovery stress tests

### 5. Limitations

- very extreme occlusion
- severe domain shift if training data is narrow
- truly raw dual-fisheye may still benefit from upstream projection handling

## 18. Immediate Next Experiments

If building this from the current repo, the first two experiments should be:

### Experiment 1

Create a controlled benchmark from one normal video:

- original dense frames
- 5 fps
- 2 fps
- 1 fps
- interval-image subset

Goal:

- quantify temporal sparsity collapse in the current LingBot baseline

### Experiment 2

Create matched perspective and 360-derived perspective sequences from the same capture environment.

Goal:

- quantify projection-shift collapse in the current baseline

These two experiments justify the whole paper before any large refactor.

## 19. Final Recommendation

This project is worth pursuing if the goal is a top-tier paper.

The reason is not that LingBot itself is already enough.
The reason is that LingBot exposes a clean and important unsolved boundary:

- streaming models are still too tied to continuous perspective video

RoAM3R is the most natural and strongest next step because it grows directly from that boundary while remaining close enough to the current codebase to prototype quickly.
