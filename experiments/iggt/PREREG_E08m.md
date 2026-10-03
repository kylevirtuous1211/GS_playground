# E08m pre-registration: do VGGT's cameras and IGGT's instance features degrade with more input frames?

Written 2026-10-03, before any run below; the commit that adds this file precedes them, and the runner refuses to start without it.

## Question

VGGT-1B was trained on 2 to 24 frames per scene ("we randomly sample 2–24 frames", VGGT Sec. 3.4) and is run on up to hundreds; IGGT's backbone and instance head were trained on 1 to 12.
Fed more frames than they were trained on, do their outputs get worse?
Answered by holding a fixed set of anchor frames and adding context frames around them, so that only the number of frames each model sees changes, never the frames it is scored on.

## Part 1: VGGT-1B cameras against COLMAP

- Data: all seven official Mip-NeRF 360 scenes (bicycle, bonsai, counter, garden, kitchen, room, stump; `~/datasets/mipnerf360/360_v2`, `images_4`), with their COLMAP poses (`sparse/0`) as ground truth. None of the 17 datasets VGGT lists as training data is Mip-NeRF 360.
- Model: VGGT-1B as SuRFLo ships it (equal to `facebook/VGGT-1B`, E08l), run as SuRFLo runs it: SuRFLo's `no_stretch` 518-wide preprocessing, bf16 autocast.
- Anchors: 8 frames per scene at evenly spaced positions of the name-sorted frame list (the first frame is always an anchor and always first in the batch, so VGGT's reference frame never changes).
- Context: N - 8 further frames drawn uniformly at random without replacement from the remaining frames, for N in {8, 16, 24, 32, 48, 64, 96, 128}; three draws (seeds 0, 1, 2) per N; frames in name order within the batch. Where a scene has fewer frames than N (stump, 125), its largest N uses all of them.
- Metrics on the 28 anchor pairs, as VGGT's own camera evaluation: relative rotation error and relative translation-direction error (with the sign ambiguity VGGT's evaluation allows) per pair, and AUC@30 of their maximum; also the median focal-length error of the anchors against COLMAP's focal mapped through the same preprocessing; peak GPU memory and forward time.
- Reading: per scene and N, the median over the three draws; per N, the median and range over the seven scenes.
  **VGGT degrades beyond its training range** if both: the median over scenes of AUC@30 at each scene's largest N is lower than at N = 24, and in at least 5 of the 7 scenes the AUC@30 at the largest N is below that at N = 24 by more than that scene's N = 24 range over draws. Otherwise: no decline measured up to 128 frames.

## Part 2: IGGT's instance features

IGGT's code breaks above 12 frames (its point and instance heads mis-handle the frame-chunking path). The fix runs those two heads on all frames at once, which for 12 frames or fewer is the code path IGGT already takes; both heads act on each frame separately, so chunking never changed what they compute.
IGGT runs as its demo does: images stretch-resized to 504x336, bf16 autocast, weights loaded strictly.
Anchors: 4 frames per scene, evenly spaced as above; context drawn as above for N in {4, 8, 12, 16, 24, 32}, seeds 0, 1, 2.
Instances are formed as in IGGT's demo, on the anchor frames only: L2-normalised 8-dim features averaged over each pixel's 20 nearest 3D neighbours (IGGT's own predicted points), clustered with HDBSCAN (the demo's sizes scaled by the pixel subsample), every pixel assigned to its nearest clustered sample.

- **2a, stability** on the seven Mip-NeRF 360 scenes (no instance ground truth there): the anchors' mean per-pixel cosine similarity to their features at N = 4 (anchors only), and the adjusted Rand index of their clusters against those at N = 4. Descriptive: change is not the same as getting worse.
- **2b, quality** on ScanNet++ v2 `nvs_sem_val` scenes, which carry instance ground truth: the three E08l scenes (825d228aec, 6115eddb86, 13c3e046d7) and the first two others, by id, with at least 200 unblurred training frames (09c1414f1b, 1ada7a0617). Frames are the official training frames minus those flagged `is_bad`; their 1752x1168 DSLR images have IGGT's 3:2 aspect, so the resize does not stretch them.
  Ground truth per anchor frame: the scene's `scans/mesh_aligned_0.05.ply` ray-cast at 504x336 with the DSLR intrinsics scaled to it and the DSLR pose mapped into the mesh frame (E08l's mapping, checked against COLMAP); each pixel takes the instance of the hit face's first vertex, from `segments.json` and `segments_anno.json`; instances whose label maps (`map_benchmark.csv`, `instance_map_to`) to a class in `top100_instance.txt` count, the rest are unlabelled.
  Metric, T-mIoU style (multi-view consistent): predicted clusters are matched one-to-one to ground-truth instances by the Hungarian algorithm on IoU computed jointly over the 4 anchor frames, over every pixel where the mesh is hit; the score is the mean matched IoU over ground-truth instances with at least 200 pixels across the anchors (unmatched instances score 0).
  Named difference: every `nvs_sem_val` scene is in IGGT's released training archive, so absolute values are optimistic; the question is the trend with N, which the leak does not obviously favour.
- Reading for 2b: per scene and N, the median over draws; **IGGT degrades beyond its training range** if the median over scenes at N = 32 is lower than at N = 12, and in at least 4 of the 5 scenes T-mIoU at N = 32 is below that at N = 12 by more than that scene's N = 12 range over draws. Otherwise: no decline measured up to 32 frames.

## Validity checks, before reading any metric

1. The IGGT fix: on one 8-frame input, the patched model's instance features are bitwise equal to the unpatched model's (the path below 13 frames is untouched); on one 16-frame input, permuting frames 2 to 16 permutes the instance features accordingly (VGGT's global attention is permutation-equivariant apart from the first frame, and the heads act per frame), to within bf16 noise (cosine similarity of every frame's features >= 0.999).
2. The camera metric: fed the COLMAP poses as its own prediction it scores AUC@30 = 1 and zero errors; fed a known rotation perturbation it reports that angle.
3. The ground-truth masks: on every anchor frame the mesh is hit on at least 90% of pixels, and one overlay per scene is saved and looked at before any T-mIoU is read.
