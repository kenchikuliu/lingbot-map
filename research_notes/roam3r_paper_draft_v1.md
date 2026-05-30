# RoAM3R Paper Draft v1

Date: 2026-05-13
Target venues: CVPR / ICCV / NeurIPS
Project status: idea-to-paper framing, pre-result draft
Related note: `research_notes/roam3r_goal_driven_paper.md`

## 1. Paper Story in One Page

### Core story

Recent streaming 3D reconstruction systems have become fast and effective on continuous perspective video, but they still break when two assumptions are violated:

- the camera model is no longer near-pinhole
- temporal continuity is no longer dense and smooth

This paper argues that robust streaming reconstruction must jointly address:

- projection shift
- temporal sparsity
- online recovery

instead of optimizing only one axis at a time.

### Main thesis

We propose a camera-agnostic streaming 3D reconstruction framework that remains stable on perspective, fisheye, and 360-derived imagery, and degrades more gracefully under low-FPS and interval-sampled observations.

### Reviewer-facing contribution summary

1. We define a practically important but under-studied setting: streaming 3D reconstruction under joint projection and temporal shift.
2. We introduce a unified ray camera interface for projection-aware streaming geometry.
3. We introduce overlap-aware adaptive memory and uncertainty-gated relocalization to handle sparse or discontinuous observations.
4. We propose a controlled evaluation protocol covering projection shift, temporal sparsity, and viewpoint discontinuity.

## 2. Title Candidates

Preferred:

- RoAM3R: Robust Camera-Agnostic Streaming 3D Reconstruction

Alternatives:

- Beyond Perspective Video: Robust Streaming 3D Reconstruction under Projection and Temporal Shift
- Camera-Agnostic Streaming 3D Reconstruction with Adaptive Memory and Online Relocalization
- Robust Streaming Mapping for Wide-FoV and Sparse-Temporal Observations

## 3. Abstract Package

## 3.1 Safe Abstract for Early Draft

Streaming 3D reconstruction has recently advanced through feed-forward models that process monocular video in real time. However, existing methods remain strongly biased toward continuous perspective video, and often degrade under projection shift, temporal sparsity, and abrupt viewpoint changes. We present RoAM3R, a robust camera-agnostic streaming 3D reconstruction framework designed to operate beyond the standard perspective-video regime. RoAM3R introduces a unified ray camera interface that replaces narrow FoV-centric camera encoding, enabling a single model to process perspective, fisheye, and 360-derived observations. To handle sparse or discontinuous temporal input, we further propose an overlap-aware adaptive memory that retrieves informative historical keyframes based on geometric novelty, confidence, and overlap rather than temporal recency alone. Finally, RoAM3R incorporates an uncertainty-gated online relocalization mechanism that detects unstable updates and recovers the streaming state without expensive global optimization. We evaluate RoAM3R under three controlled axes of difficulty: projection shift, temporal subsampling, and viewpoint discontinuity. RoAM3R is designed to preserve strong performance on standard perspective streaming benchmarks while improving robustness on wide-FoV imagery, low-FPS sequences, and interval captures.

Role:

- challenge
- contribution
- high-level expected result

## 3.2 Submission-Oriented Abstract with Result Slots

Streaming 3D reconstruction has recently become practical through feed-forward models that process monocular video in real time. Yet existing methods remain strongly biased toward continuous perspective video and degrade sharply under projection shift, temporal sparsity, and abrupt viewpoint changes. We present RoAM3R, a robust camera-agnostic streaming 3D reconstruction framework for perspective, fisheye, and 360-derived observations. RoAM3R introduces a unified ray camera interface that replaces FoV-centric camera encoding, an overlap-aware adaptive memory that retrieves informative historical keyframes beyond temporal recency, and an uncertainty-gated online relocalization module that recovers the streaming state after unstable updates. Together, these components extend streaming reconstruction beyond the standard dense-video setting while preserving efficient inference. Across controlled benchmarks spanning projection shift, temporal subsampling, and viewpoint discontinuity, RoAM3R improves over prior streaming baselines by `[X]` on wide-FoV inputs, `[Y]` under low-FPS evaluation, and reduces catastrophic tracking failures by `[Z]`. These results suggest that robust streaming reconstruction requires jointly modeling camera geometry, memory, and recovery behavior rather than optimizing any single axis in isolation.

Role:

- final submission abstract
- add numbers after experiments are finished

## 4. Introduction Outline

Recommended paragraph plan:

1. Opening paragraph:
   application value of streaming 3D reconstruction and immediate statement of the real challenge
2. Prior-work paragraph:
   why recent feed-forward streaming methods succeed on dense perspective video
3. Boundary paragraph:
   where those methods fail and why that boundary matters
4. Our-method paragraph:
   summary of RoAM3R and its three modules
5. Experiment-and-contribution paragraph:
   what we test and what the paper contributes

This is a five-paragraph introduction. That is enough for a strong conference intro if each paragraph carries one message.

## 4.1 Introduction Draft

### Paragraph 1: Opening / task / challenge

Streaming 3D reconstruction is a core capability for embodied perception, mobile mapping, robotics, and real-time scene understanding. Recent feed-forward systems have shown that dense geometry and camera motion can be predicted online from monocular video with impressive efficiency. However, these systems are still built around a narrow operating regime: continuous perspective video with smooth motion and strong temporal overlap. Once the input departs from this regime, such as with fisheye imagery, 360-derived views, low-FPS video, or interval-sampled captures, reconstruction quality often degrades sharply. This gap limits the practical deployment of streaming reconstruction systems outside curated perspective-video benchmarks.

Role:

- task
- importance
- direct exposure of target challenge

### Paragraph 2: Prior methods and what they solve well

Recent streaming reconstruction models improve online geometry estimation by combining strong image features with temporal memory, geometric reasoning, and efficient causal inference. This line of work has substantially reduced the gap between streaming and optimization-heavy reconstruction, making long-horizon online processing increasingly practical. Nevertheless, these advances mainly target stability and efficiency within the dense monocular-video setting. Their camera representations, geometry pipelines, and memory policies remain strongly coupled to perspective projection and local temporal continuity. As a result, current progress on streaming reconstruction does not yet translate into robustness under projection shift or temporally sparse observations.

Role:

- place the paper in the recent method landscape
- state what prior methods solved
- isolate the remaining gap

### Paragraph 3: Boundary and technical reason

This remaining gap is technically nontrivial for three reasons. First, projection shift changes the geometry of image formation itself, making FoV-centric or pinhole-centered representations poorly matched to fisheye or 360-derived inputs. Second, temporal sparsity weakens the local overlap assumptions that support recency-based memory and frame-to-frame streaming updates. Third, once local updates become unreliable, a purely forward streaming system lacks an explicit mechanism to detect failure and recover. Therefore, robust streaming reconstruction requires not a single patch, but a joint treatment of camera representation, memory selection, and online recovery.

Role:

- decompose the exact challenge
- justify why the paper needs multiple modules

### Paragraph 4: Our method

To address this problem, we propose RoAM3R, a robust camera-agnostic streaming 3D reconstruction framework for observations beyond continuous perspective video. RoAM3R introduces a unified ray camera interface that replaces narrow FoV-centric encoding with projection-aware ray-based conditioning. It further uses an overlap-aware adaptive memory that retrieves informative historical keyframes according to overlap, novelty, and uncertainty rather than temporal recency alone. Finally, RoAM3R incorporates an uncertainty-gated online relocalization module that detects unstable streaming updates and reattaches the current state to reliable memory without expensive global optimization. Together, these components extend streaming reconstruction toward a more realistic regime of mixed camera models and sparse temporal continuity.

Role:

- method summary
- insight
- module list

### Paragraph 5: Experiments and contributions

We evaluate RoAM3R under three controlled axes of difficulty: projection shift, temporal subsampling, and viewpoint discontinuity. Our experiments are designed to answer three questions: whether RoAM3R preserves performance on standard perspective-video benchmarks, whether it degrades more gracefully under low-FPS and interval-sampled inputs, and whether it recovers more reliably after abrupt viewpoint breaks. The results are intended to show that robust streaming reconstruction requires jointly modeling camera geometry, adaptive memory, and recovery behavior. In summary, our contributions are a camera-agnostic streaming reconstruction framework, a new adaptive memory and relocalization design for temporally sparse inputs, and a stress-test protocol for evaluating robustness beyond the standard dense-video setting.

Role:

- experiment promise
- contribution list
- ending paragraph

## 4.2 Contribution Bullets for the Paper

Use these near the end of the introduction:

1. We identify and formalize streaming 3D reconstruction under joint projection and temporal shift as a practically important but under-studied setting.
2. We propose RoAM3R, a camera-agnostic streaming reconstruction framework built on a unified ray camera interface, overlap-aware adaptive memory, and uncertainty-gated online relocalization.
3. We design a controlled evaluation protocol spanning perspective, wide-FoV, temporally subsampled, and viewpoint-discontinuous inputs, enabling rigorous analysis of robustness beyond standard dense-video benchmarks.

## 5. Method Outline

Recommended section structure:

1. Problem Formulation
2. Method Overview
3. Unified Ray Camera Interface
4. Overlap-Aware Adaptive Memory
5. Uncertainty-Gated Online Relocalization
6. Training Objectives and Implementation Details

## 5.1 Method Overview Paragraph

We consider streaming 3D reconstruction from an input sequence of monocular observations that may vary in projection model and temporal continuity. Given a sequence of frames, our goal is to estimate camera motion and scene geometry online while maintaining robustness under wide-FoV imagery, low-FPS sampling, and abrupt viewpoint changes. Figure 2 provides an overview of RoAM3R. The method contains three key components: a unified ray camera interface for projection-aware geometry modeling, an overlap-aware adaptive memory for non-local historical retrieval, and an uncertainty-gated relocalization module for online recovery. Sections 3.1 to 3.3 describe these components, followed by training objectives and implementation details.

Role:

- method overview
- section map

## 5.2 Section 3.1 Problem Formulation

### Motivation

A clear formulation is needed because this paper is not about generic SLAM and not about offline global reconstruction. The paper specifically studies robust streaming reconstruction under mixed camera projection and mixed temporal continuity.

### Design points to define

- input sequence `I_1, ..., I_T`
- optional per-frame camera metadata
- projection family `c_t` for each frame or sequence
- streaming memory state `M_t`
- outputs:
  - pose `P_t`
  - depth or pointmap `D_t`
  - confidence `U_t`

### Draft paragraph

We study streaming 3D reconstruction from a sequence of monocular observations `\{I_t\}_{t=1}^T`, where frames may differ in projection characteristics and temporal spacing. At each step `t`, the model receives the current frame, camera metadata when available, and a bounded streaming memory state `M_{t-1}`. The model predicts the current camera state `P_t`, dense geometry `D_t`, and uncertainty estimates `U_t`, while updating the memory to `M_t`. Unlike standard streaming settings that assume perspective projection and dense temporal continuity, we explicitly target mixed regimes including wide-FoV imagery, low-FPS video, and interval captures. The goal is to maintain accurate online geometry estimation without resorting to full global optimization.

Role:

- setting
- boundary
- symbols

## 5.3 Section 3.2 Unified Ray Camera Interface

### Motivation

FoV-centric pose encoding is too narrow when the image formation model changes.

### Module design

- compute ray directions for pixels or patches
- encode camera-model identity or calibration metadata
- fuse image tokens with ray-aware tokens
- use the resulting representation for geometry prediction and pose updates

### Technical advantage

- projection-aware without hard-coding everything into pinhole intrinsics
- reusable across perspective, fisheye, and 360-derived perspective views

### Draft subsection opening

A remaining problem in existing streaming systems is that camera geometry is represented through a narrow FoV-centric parameterization, which is poorly matched to wide-FoV and non-rectilinear imagery. To address this issue, we introduce a unified ray camera interface that associates each image region with an explicit viewing direction representation. Concretely, for each frame we compute patch-level ray descriptors from the available camera model or projection metadata and fuse them with visual tokens before geometry reasoning. This design allows the backbone to reason over a projection-aware representation instead of implicitly treating all inputs as near-pinhole images. As a result, the same streaming pipeline can process perspective, fisheye, and 360-derived observations with a shared geometry interface.

Role:

- motivation
- design
- advantage

## 5.4 Section 3.3 Overlap-Aware Adaptive Memory

### Motivation

Recency is not enough when the input is sparse or discontinuous.

### Module design

- maintain recent memory and global candidate memory
- compute overlap score, novelty score, and uncertainty score
- retrieve best historical support frame or keyframe set
- decide whether to promote the current frame to memory

### Technical advantage

- allows non-local support under interval captures
- avoids forcing all useful context to come from immediate neighbors

### Draft subsection opening

Streaming reconstruction typically relies on temporally local memory, which is effective when adjacent frames have strong overlap but becomes fragile under low-FPS or interval-sampled input. We therefore replace purely recency-based memory access with overlap-aware adaptive memory. For each incoming frame, the model estimates its overlap with stored keyframes together with geometric novelty and predictive uncertainty, and uses these signals to retrieve the most informative historical support. The same signals also guide whether the current frame should be promoted into long-term memory. This mechanism lets the model attach sparse observations to geometrically relevant history rather than to temporally adjacent but weakly overlapping context.

Role:

- motivation
- design
- advantage

## 5.5 Section 3.4 Uncertainty-Gated Online Relocalization

### Motivation

A streaming system should not keep drifting blindly after an unstable update.

### Module design

- estimate risk from uncertainty and geometry inconsistency
- trigger recovery when risk exceeds threshold
- align current frame to retrieved memory
- resume normal streaming updates

### Technical advantage

- explicit recovery without heavy global optimization
- directly targets catastrophic failure rather than only average error

### Draft subsection opening

Even with improved camera modeling and adaptive memory, streaming updates can still become unreliable after abrupt viewpoint changes or severe overlap loss. Existing feed-forward systems often lack an explicit mechanism for recognizing this failure mode. We address this limitation with an uncertainty-gated online relocalization module. The model continuously estimates a streaming risk score from predictive uncertainty and geometric inconsistency; when the risk becomes high, it retrieves reliable memory anchors and performs a lightweight relocalization step before continuing the stream. This design reduces catastrophic drift while preserving the efficiency of online inference.

Role:

- motivation
- design
- advantage

## 5.6 Section 3.5 Training Objectives and Implementation Details

Keep this section compact.

Include:

- geometry supervision
- pose supervision or self-consistency losses
- uncertainty calibration loss if used
- memory/relocalization auxiliary objectives if needed
- implementation choices:
  - backbone
  - image size
  - memory size
  - calibration sources
  - training mixture across projection and FPS settings

Draft opening:

RoAM3R is trained with a combination of geometry, pose, and confidence objectives. In addition to standard reconstruction losses, we include auxiliary supervision for uncertainty calibration and memory-guided recovery when available. To improve robustness beyond the dense perspective-video regime, training data is mixed across projection families and temporal sampling patterns rather than being restricted to continuous perspective sequences. Further implementation details are provided in the appendix.

Role:

- losses
- training regime
- implementation note

## 6. Experiment Section Outline

Recommended section structure:

1. Experimental Setup
2. Main Comparison on Dense Perspective Video
3. Robustness under Temporal Sparsity
4. Robustness under Projection Shift
5. Recovery under Viewpoint Discontinuity
6. Ablation Studies
7. Qualitative Results and Limitations

## 6.1 Experimental Setup Paragraph

We evaluate RoAM3R on both standard streaming reconstruction benchmarks and controlled robustness benchmarks derived from them. Our evaluation spans three axes of difficulty: projection shift, temporal sparsity, and viewpoint discontinuity. We compare against recent streaming and geometry-centric baselines, including methods specialized for long-horizon streaming as well as methods designed for broader camera-model generalization. We report camera trajectory accuracy, geometry quality, failure rate, recovery success rate, and inference efficiency. All baselines are evaluated under matched preprocessing and sampling protocols whenever possible.

Role:

- experiment setup
- evaluation scope

## 6.2 Section 4.2 Main Comparison on Dense Perspective Video

Purpose:

- show you did not break the original LingBot regime

Main message:

- RoAM3R matches or nearly matches strong perspective-video baselines on the standard setting

Suggested table:

- Table 1: Dense perspective streaming benchmark comparison

Columns:

- Method
- ATE `↓`
- RPE `↓`
- Depth error `↓`
- Completeness `↑`
- FPS `↑`
- Memory `↓`

## 6.3 Section 4.3 Robustness under Temporal Sparsity

Purpose:

- justify adaptive memory

Settings:

- original FPS
- 10 FPS
- 5 FPS
- 2 FPS
- 1 FPS
- interval-capture subset

Suggested table:

- Table 2: Temporal subsampling robustness

Suggested figure:

- Figure 4: performance-versus-FPS degradation curves

Main message:

- RoAM3R degrades more gracefully than recency-based streaming baselines

## 6.4 Section 4.4 Robustness under Projection Shift

Purpose:

- justify ray camera interface

Settings:

- perspective
- fisheye
- 360-derived perspective
- wide-FoV transformed sequences

Suggested table:

- Table 3: Projection-shift evaluation

Suggested figure:

- Figure 5: qualitative reconstructions across camera models

Main message:

- RoAM3R generalizes better across camera models than FoV-centric streaming models

## 6.5 Section 4.5 Recovery under Viewpoint Discontinuity

Purpose:

- justify relocalization

Stressors:

- dropped frame segments
- abrupt heading change
- transient occlusion
- repeated textures

Suggested table:

- Table 4: failure rate and recovery success rate

Suggested figure:

- Figure 6: trajectory recovery examples after viewpoint break

Main message:

- RoAM3R reduces catastrophic failure, not only average drift

## 6.6 Section 4.6 Ablation Studies

Core ablation table:

- baseline
- + ray interface
- + adaptive memory
- + relocalization
- full model

Additional ablations:

- mixed-camera training vs perspective-only training
- mixed-FPS training vs dense-only training
- local-only memory vs local-plus-global retrieval
- uncertainty trigger variants

Suggested table:

- Table 5: Core ablations

Suggested mini-figure:

- Figure 7: failure cases resolved by each module

## 6.7 Section 4.7 Qualitative Results and Limitations

Do not hide the failure boundary.

Explicit limitations:

- extremely poor calibration
- ultra-wide or raw dual-fisheye without suitable metadata
- very low overlap and heavy motion blur simultaneously

Reviewer value:

- this makes the paper look controlled rather than over-claimed

## 7. Teaser and Figure Plan

### Teaser figure

Three rows:

1. standard perspective video
2. low-FPS or interval capture
3. fisheye or 360-derived input

Three columns:

1. input frame
2. baseline reconstruction
3. RoAM3R reconstruction

Caption message:

- one model, three input regimes, graceful robustness beyond continuous perspective video

### Pipeline figure

Blocks:

1. input frame + camera metadata
2. ray camera interface
3. streaming geometry backbone
4. adaptive memory retrieval
5. uncertainty-gated relocalization
6. outputs: pose, depth, confidence

## 8. Claim-Evidence Map

### Claim 1

RoAM3R preserves strong performance on dense perspective streaming benchmarks.

Evidence:

- Table 1 comparison

Status:

- needs experiment

### Claim 2

RoAM3R degrades more gracefully under low-FPS and interval-sampled input.

Evidence:

- Table 2
- Figure 4

Status:

- needs experiment

### Claim 3

RoAM3R generalizes better across camera projection families.

Evidence:

- Table 3
- Figure 5

Status:

- needs experiment

### Claim 4

RoAM3R reduces catastrophic failures after viewpoint discontinuities.

Evidence:

- Table 4
- Figure 6

Status:

- needs experiment

## 9. Self-Review Checklist

Before treating this as a real draft, check:

1. Is the main task clearly defined as streaming reconstruction under projection and temporal shift, not generic SLAM?
2. Does each method module correspond to one explicit failure source?
3. Do experiments validate every major claim separately?
4. Is there at least one strong baseline for each comparison axis?
5. Are all result claims either supported by numbers or explicitly marked as placeholders?

## 10. Immediate Writing Next Step

The next useful draft after this file is:

- a full Introduction v1
- a full Method Overview + Section 3.2 draft
- a complete experiment table plan with dataset names and exact metrics

Do not write the full Related Work yet.
Do not finalize prose before the first controlled robustness experiments are run.
