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

## Amendment 1, 2026-10-03, after a code review and before any E08m run

An adversarial review of the code (22 agents, each finding put to a refuter) found three defects, fixed before anything ran:
- The clustering did not match the text above ("as in IGGT's demo"): it left out the demo's HDBSCAN `cluster_selection_epsilon` of 0.06, re-normalised the 3D-averaged features, and counted each pixel among its own 20 neighbours. It now passes epsilon 0.06 (a feature distance, so not scaled with the subsample), does not re-normalise, and excludes the pixel itself.
- VGGT's first forward pass in a process carried CUDA set-up in its time; one untimed pass now comes first.
- The part-1 summary lacked the range over scenes and dropped stump from the N = 128 row; it now gives median and range per N, with stump's 125-frame batch standing in at 128, labelled.

What the demo's clustering does, seen on garden at 12 frames (outputs/iggt/garden, the qualitative render made before this experiment; garden is one of the seven stability scenes, not a T-mIoU scene): with epsilon 0.06 the table, the pot and the ground fall into one cluster holding 46% of the pixels at the pixel stride used here and 47% at half that stride, so the merge comes from the demo's procedure, not from the subsample; without epsilon the same features gave 290 clusters. HDBSCAN at full resolution is not feasible here (109 s at a quarter of the pixels, per run).
T-mIoU therefore measures IGGT's features together with its demo clustering, and can move when a merge flips for reasons unrelated to the features.

Added, before any run: a clustering-free quality metric on the same ScanNet++ anchors, **centroid mIoU**: each ground-truth instance (>= 200 pixels over the anchors) has as centroid the mean of IGGT's features over its pixels; every pixel of those instances is assigned to the nearest centroid by cosine; the score is the mean IoU over the instances.
It is read with the same rule as T-mIoU (N = 32 against N = 12, at least 4 of 5 scenes beyond their N = 12 range over draws), and both verdicts are reported; if they disagree, that is the result.

## Amendment 2, 2026-10-03, after part 1 ran out of memory, before any reading

Part 1 ran out of GPU memory at N = 128 on its first scene (bicycle) while another user process held 18 GB of the 48 GB card; VGGT's full forward had peaked at 24.9 GiB at N = 96.
VGGT's camera prediction depends only on its aggregator and camera head; the depth, point and track heads the full forward also runs never touch it.
Part 1 therefore runs the aggregator (bf16 autocast) and the camera head (autocast off), exactly as VGGT's forward runs them; on an 8-frame bicycle batch this gives a pose encoding bitwise equal to the full forward's.
The camera metrics are unchanged by construction; the reported peak memory and time are now those of this camera-only pass.
The 19 rows already written (bicycle, N <= 96) are discarded and recomputed this way, so every row shares one definition; no reading had been made from them.

## Amendment 3, 2026-10-03, after part 1 ran out of memory again, before any reading

The camera-only pass of Amendment 2 peaked as high as the full forward (24.9 GiB at N = 96): the memory is VGGT's aggregator holding the tokens of all 24 layers, and N = 128 again ran out beside the other process's 18 GB.
The camera head reads only the last layer.
Part 1 now runs the aggregator operation for operation (the same calls to its own blocks), returning only the last layer.
Checked before use: its pose encoding is bitwise equal to the full forward's at N = 8 and at N = 96 (the largest that fits either way), and its peak at N = 96 is 9.9 GiB instead of 24.9.
The reported memory and time are those of this pass. The rows written so far (bicycle, N <= 96) are discarded and recomputed; no reading had been made from them.
