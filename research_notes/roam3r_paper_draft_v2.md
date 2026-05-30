# RoAM3R Paper Draft v2

Date: 2026-05-13
Target venues: CVPR / ICCV / NeurIPS
Status: pre-result technical draft
Related files:

- `research_notes/roam3r_goal_driven_paper.md`
- `research_notes/roam3r_paper_draft_v1.md`

## 1. Positioning and Claim Discipline

This draft is intentionally written with conservative claim language.

The paper should not claim:

- generic SLAM
- universal camera support
- complete robustness to arbitrary input breakage

The paper should claim:

- robust streaming 3D reconstruction
- improved robustness under projection shift and temporal sparsity
- online recovery behavior under viewpoint discontinuity

The safest positioning is:

- a streaming 3D reconstruction paper
- with explicit robustness contributions
- evaluated beyond dense perspective video

## 2. Related Work Positioning

The paper sits at the intersection of three recent lines of work.

### 2.1 Streaming reconstruction

- LingBot-MAP studies feed-forward streaming reconstruction with geometric context and long-range memory.
- SLAM3R studies real-time dense scene reconstruction from monocular RGB video.
- LONG3R studies long-sequence streaming reconstruction with recurrent memory.
- PAS3R studies pose-adaptive state updates for long video sequences.
- FILT3R studies adaptive latent filtering for streaming state updates.
- FrameVGGT, STAC, and RetrieveVGGT study bounded-memory or retrieval-based long-context streaming variants for VGGT-style pipelines.

### 2.2 Camera-model generalization

- CAM3R studies camera-agnostic reconstruction through ray-aware modeling.
- Wid3R studies wide-FoV reconstruction through camera-model conditioning.

### 2.3 Global or memory-based geometry reasoning

- Spann3R studies spatial memory for global-coordinate reconstruction over image collections.

### 2.4 Our slot

The paper should be framed as addressing a gap that is still weakly covered:

- streaming reconstruction under the joint boundary of projection shift and temporal sparsity

That is narrower than "all of 3D reconstruction", but stronger than "a small patch to LingBot".

## 3. Full Abstract v2

Streaming 3D reconstruction has recently become practical through feed-forward models that infer camera motion and scene geometry online from monocular video. However, existing systems remain strongly biased toward continuous perspective video and often degrade under projection shift, temporal sparsity, and abrupt viewpoint changes. We present RoAM3R, a robust camera-agnostic streaming 3D reconstruction framework designed for observations beyond the standard dense perspective-video regime. RoAM3R introduces a unified ray camera interface that replaces FoV-centric camera encoding with projection-aware conditioning, allowing a single model to process perspective, fisheye, and 360-derived inputs through a shared geometry pathway. To handle sparse or discontinuous temporal input, RoAM3R further employs an overlap-aware adaptive memory that retrieves informative historical keyframes according to overlap, novelty, and uncertainty rather than temporal recency alone. Finally, we incorporate an uncertainty-gated online relocalization mechanism that detects unstable updates and reattaches the current stream to reliable historical anchors without resorting to expensive full global optimization. We evaluate RoAM3R under three controlled axes of difficulty: projection shift, temporal subsampling, and viewpoint discontinuity. Our results are designed to show that robust streaming reconstruction requires jointly modeling camera geometry, memory selection, and recovery behavior rather than optimizing any single axis in isolation.

## 4. Introduction v2

### Paragraph 1

Streaming 3D reconstruction is a core capability for embodied perception, mobile mapping, robotics, and real-time scene understanding. Recent feed-forward systems have shown that dense geometry and camera motion can be estimated online from monocular video with increasingly strong efficiency and quality. Yet this progress is concentrated in a narrow regime: continuous perspective video with smooth motion and strong temporal overlap. In real deployment, however, input streams often violate these assumptions through wide-FoV optics, fisheye distortion, 360-derived views, low-FPS capture, or interval-sampled observations. Under these conditions, the performance of current streaming systems often degrades sharply, limiting their practical usability beyond curated perspective-video benchmarks.

### Paragraph 2

Why do existing methods struggle in this setting? First, most streaming pipelines remain centered on perspective-oriented camera representations, making their geometry reasoning poorly matched to non-rectilinear imaging. Second, their memory policies still depend heavily on local temporal continuity, which weakens rapidly when frame rate drops or viewpoint jumps increase. Third, when local streaming updates become unreliable, many feed-forward systems lack an explicit mechanism for detecting failure and recovering from it. These issues suggest that robustness under realistic input shift is not a one-module problem. Instead, it requires a joint treatment of camera representation, memory selection, and online recovery.

### Paragraph 3

Recent work has improved online 3D reconstruction along several important axes, including real-time monocular reconstruction, long-sequence streaming stability, and bounded-memory inference. In parallel, another line of work has begun to address camera-model generalization for wide-FoV and panoramic imagery. However, these directions remain only partially connected. Streaming reconstruction methods still focus primarily on dense perspective video, while camera-agnostic reconstruction methods are not typically designed for long-horizon online streaming with failure recovery. As a result, the practically important setting of camera-agnostic streaming reconstruction under sparse temporal continuity remains weakly addressed.

### Paragraph 4

In this work, we propose RoAM3R, a robust camera-agnostic streaming 3D reconstruction framework for observations beyond continuous perspective video. RoAM3R contains three key components. First, a unified ray camera interface replaces narrow FoV-centric encoding with projection-aware ray conditioning, enabling shared geometry reasoning across perspective, fisheye, and 360-derived inputs. Second, an overlap-aware adaptive memory retrieves informative historical context according to overlap, novelty, and uncertainty rather than temporal recency alone, making the system better suited for low-FPS and interval-sampled observations. Third, an uncertainty-gated online relocalization module detects unstable streaming updates and reattaches the current state to reliable memory anchors without full global optimization.

### Paragraph 5

We evaluate RoAM3R under three controlled axes of difficulty: projection shift, temporal subsampling, and viewpoint discontinuity. These experiments are designed to answer three questions. Can the method preserve strong performance on standard perspective-video benchmarks? Does it degrade more gracefully than prior streaming systems under low-FPS and interval-sampled input? Can it recover more reliably after abrupt viewpoint breaks? By answering these questions, the paper argues that robust streaming reconstruction should be studied as a joint robustness problem rather than as a pure efficiency or long-context problem.

### Contributions

1. We identify streaming 3D reconstruction under joint projection and temporal shift as a practically important and still under-explored setting.
2. We propose RoAM3R, a camera-agnostic streaming reconstruction framework built from a unified ray camera interface, overlap-aware adaptive memory, and uncertainty-gated online relocalization.
3. We design a controlled robustness evaluation protocol spanning perspective, wide-FoV, temporally sparse, and viewpoint-discontinuous inputs.

## 5. Method v1

## 5.1 Overview

We consider streaming 3D reconstruction from a sequence of monocular observations `\{I_t\}_{t=1}^T`, where both the camera projection model and the temporal spacing between frames may vary. At each step `t`, the model receives the current frame, optional camera metadata, and a bounded memory state summarizing previous observations. The objective is to estimate the current camera state and dense scene geometry online while maintaining robustness under wide-FoV imagery, low-FPS video, interval captures, and abrupt viewpoint changes. Figure 2 illustrates the RoAM3R pipeline. RoAM3R contains three modules: a unified ray camera interface for projection-aware geometry reasoning, an overlap-aware adaptive memory for non-local historical support, and an uncertainty-gated relocalization mechanism for online recovery.

## 5.2 Unified Ray Camera Interface

The first limitation of existing streaming pipelines is that camera geometry is often encoded through narrow perspective-oriented parameterizations, such as focal length or FoV summaries. These representations are efficient when the input remains near-pinhole, but they become increasingly mismatched to fisheye or 360-derived imagery, where image coordinates no longer correspond to perspective projection in a simple way. To overcome this limitation, we introduce a unified ray camera interface.

For each frame, we associate each image patch with a compact ray descriptor that summarizes its viewing direction under the underlying camera model. When calibration metadata is available, these ray descriptors are produced analytically from the camera model. When calibration is partial or noisy, the representation can be augmented with a learnable camera-model token that absorbs residual projection-specific variation. The ray descriptors are fused with visual tokens before geometry reasoning, so that the downstream backbone operates on projection-aware features rather than on a representation that implicitly assumes perspective imaging.

This design has two advantages. First, it removes the need to force all inputs into a single FoV-centric geometry interface. Second, it allows a single streaming model to share a common reasoning pipeline across perspective, fisheye, and 360-derived views. In this sense, the ray interface is not merely an input preprocessing trick; it changes the geometric representation that the streaming model reasons over.

## 5.3 Overlap-Aware Adaptive Memory

The second limitation of prior systems is their heavy dependence on temporally local memory. Under dense video, recency is often a reasonable proxy for overlap. Under low-FPS video or interval-sampled captures, however, temporally adjacent frames may carry limited geometric support, while much older frames may be more informative. RoAM3R addresses this mismatch through overlap-aware adaptive memory.

At each step, the model scores candidate memory entries using three signals: predicted overlap with the current frame, geometric novelty, and uncertainty. The overlap score estimates how useful a stored frame is as a geometric anchor for the current observation. The novelty score measures whether the current frame contributes new scene information that should be preserved in memory. The uncertainty score estimates whether the current observation is reliable enough to update the long-term state. These signals jointly determine both which historical frames are retrieved and whether the current frame is promoted into memory.

Compared with purely recency-based memory, this design allows the model to attach sparse observations to geometrically relevant history rather than to weakly overlapping neighbors. It also makes memory usage more interpretable: retained frames are those expected to maximize future geometric support rather than simply those that arrived most recently.

## 5.4 Uncertainty-Gated Online Relocalization

Even with a better camera interface and adaptive memory, online reconstruction can still become unstable after abrupt viewpoint changes, heavy occlusion, or prolonged low-overlap input. A major weakness of purely forward streaming systems is that they may continue updating the state even when the update is no longer trustworthy. To reduce this failure mode, we add an uncertainty-gated online relocalization mechanism.

RoAM3R continuously estimates a streaming risk score from predictive uncertainty and geometric inconsistency. When the risk remains low, the pipeline proceeds with standard streaming updates. When the risk exceeds a threshold, the model retrieves high-value historical anchors from memory and performs a lightweight relocalization step before resuming normal updates. The purpose of this mechanism is not to reproduce full classical pose graph optimization. Instead, it provides a low-cost recovery path that prevents the system from drifting blindly after an unstable update.

This module changes the evaluation target of the system in an important way. Rather than only reducing average trajectory error, the method explicitly targets catastrophic failure rate and recovery success, which are much closer to how streaming systems are judged in practice.

## 5.5 Training and Optimization

RoAM3R is trained with a combination of geometry supervision, pose supervision or self-consistency objectives, and confidence-related auxiliary losses. To encourage robustness beyond the dense perspective-video regime, the training data is mixed across projection families and temporal sampling patterns. In practice, this means that the model is not exposed only to continuous perspective sequences, but also to wide-FoV imagery and temporally subsampled streams. If relocalization is implemented as a discrete trigger, the risk score can be trained using pseudo-labels derived from geometric inconsistency or validation-time failure indicators.

The main training principle is that robustness should be injected both architecturally and through data exposure. A camera-agnostic representation without mixed-camera training is unlikely to generalize well, while mixed-camera data without projection-aware geometry is unlikely to use the extra data effectively.

## 6. Experiments v1

## 6.1 Experimental Questions

The experiments should answer four concrete questions:

1. Does RoAM3R preserve competitive performance on standard dense perspective-video streaming benchmarks?
2. Does RoAM3R degrade more gracefully under temporal sparsity?
3. Does RoAM3R generalize better under projection shift?
4. Does RoAM3R reduce catastrophic failure after viewpoint discontinuities?

The structure of the experiment section should mirror these four questions exactly.

## 6.2 Experimental Setup

We evaluate RoAM3R on both standard streaming reconstruction benchmarks and controlled robustness benchmarks derived from them. The robustness evaluation spans three axes of difficulty: projection shift, temporal sparsity, and viewpoint discontinuity. We compare against recent streaming reconstruction baselines, long-context memory variants, and camera-generalization models when protocols permit fair comparison. Metrics include trajectory accuracy, geometry quality, failure rate, recovery success rate, and efficiency. For all temporal-subsampling experiments, every baseline is evaluated on the same frame subsets.

## 6.3 Dense Perspective Benchmarks

This experiment verifies that the proposed robustness modules do not damage the strong standard regime. The target outcome is not necessarily large gains over LingBot-MAP or PAS3R on dense perspective video, but competitive performance with minimal regression. This section is important because without it, reviewers may dismiss the method as a robustness-specialized model that sacrifices the mainstream setting too heavily.

## 6.4 Temporal Sparsity

To stress temporal robustness, we evaluate each benchmark sequence under progressive subsampling, such as full FPS, `10 FPS`, `5 FPS`, `2 FPS`, and `1 FPS`, and supplement this with interval-capture sequences when available. The key analysis is not only the absolute score at each operating point, but also the degradation curve as temporal continuity weakens. This is the central validation for the adaptive memory design.

## 6.5 Projection Shift

To validate the ray camera interface, we construct matched or near-matched evaluation settings across perspective, wide-FoV, fisheye, and 360-derived inputs. The main result should show that perspective-centered streaming models degrade sharply outside their native projection regime, while RoAM3R remains substantially more stable. Qualitative examples are especially important here, because average metrics alone may hide severe local geometric distortion.

## 6.6 Viewpoint Discontinuity and Recovery

To evaluate relocalization, we introduce controlled viewpoint breaks through frame dropping, abrupt turns, repeated textures, and transient occlusions. This section should report both average error and discrete failure outcomes. The core claim is that RoAM3R reduces catastrophic breakdowns and recovers more often after instability. If possible, recovery latency should also be reported, since very slow recovery weakens the practical value of a streaming system.

## 6.7 Ablations

The ablation section should be structured around causal claims rather than around implementation convenience. The core table should include:

1. baseline streaming model
2. `+` ray camera interface
3. `+` adaptive memory
4. `+` relocalization
5. full RoAM3R

Additional ablations should isolate:

- mixed-camera training vs perspective-only training
- mixed-temporal training vs dense-only training
- local memory only vs local-plus-retrieval memory
- uncertainty trigger variants

## 7. Related Work Outline

Do not write the final related-work section yet, but the eventual subsection plan should be:

1. Feed-forward and streaming 3D reconstruction
2. Long-horizon and bounded-memory streaming geometry
3. Camera-agnostic and wide-FoV 3D reconstruction
4. Memory and recovery mechanisms for online geometry systems

## 8. Discussion Paragraph for the End of the Paper

Our results suggest that the next frontier for streaming reconstruction is not only longer context or lower memory usage, but robustness to the actual shifts encountered by real capture systems. In particular, camera projection shift and temporal sparsity expose a mismatch between the assumptions built into current feed-forward pipelines and the data encountered in deployment. RoAM3R provides one step toward closing this gap by jointly modeling camera geometry, adaptive memory, and online recovery within a unified streaming framework.

## 9. Reference Seeds

These are the primary papers that should shape the final positioning.

- LingBot-MAP: https://arxiv.org/abs/2604.14141
- SLAM3R: https://arxiv.org/abs/2412.09401
- LONG3R: https://arxiv.org/abs/2507.18255
- PAS3R: https://arxiv.org/abs/2603.21436
- FILT3R: https://arxiv.org/abs/2603.18493
- CAM3R: https://arxiv.org/abs/2603.22631
- Wid3R: https://arxiv.org/abs/2602.05321
- Spann3R: https://arxiv.org/abs/2408.16061
- FrameVGGT: https://arxiv.org/abs/2603.07690
- STAC: https://arxiv.org/abs/2603.20284
- RetrieveVGGT: https://arxiv.org/abs/2605.09644

## 10. Writing Risks

The paper will become weak if any of the following happen:

1. It is written as "LingBot with three patches".
2. The introduction overclaims "general SLAM".
3. The experiments fail to separate projection shift from temporal shift.
4. Recovery is described as important but evaluated only through average ATE.
5. Camera-agnostic claims are made without meaningful wide-FoV evaluation.

## 11. Immediate Next Writing Step

The next useful drafting move after this file is:

- write a full `Related Work v1`
- convert `Method v1` into subsection-level pseudo-LaTeX
- create figure captions and table captions before running final experiments
