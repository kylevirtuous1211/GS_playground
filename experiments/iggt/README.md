# IGGT

Li et al., "IGGT: Instance-Grounded Geometry Transformer", ICLR 2026, arXiv 2510.22706.
Code [`lifuguan/IGGT_official`](https://github.com/lifuguan/IGGT_official), pinned in `third_party/PINS.tsv`; weights HF `lifuguan/IGGT_official` (`iggt_checkpoint.pth`, revision 12aa6fa), derived from the CC BY-NC VGGT-1B, research only.

## Semantics on SuRFLo (E08n)

<img src="../../docs/media/iggt_semantics_sofa.webp" width="100%" alt="The same flight: SuRFLo's Gaussians, coloured by IGGT instance group, and coloured by CLIP class">

<details>
<summary>Class colours</summary>
<img src="../../docs/media/iggt_semantics_sofa_legend.png" width="520" alt="Class legend">
</details>

We ran SuRFLo on IGGT's backbone and let IGGT's instance head read the very tokens SuRFLo had computed, so one backbone pass gives both the surface and per-pixel instance features (against IGGT's own forward pass, per-pixel cosine similarity median 1.000000, minimum 0.99966).
The features are carried onto SuRFLo's Gaussians through its cameras and grouped, and each group holding at least 0.2% of the Gaussians is named by CLIP from ScanNet++'s 100 most common classes plus a few outdoor words.

This one is **exploratory**, judged by eye only.
On the lounge, instance groups mostly separate the sofa, cushions, pillar, hanging plants and TV, but neighbouring stools merge into one group and the floor and walls split into several; on a ScanNet++ room the groups are fragmented.
Class names are right for some distinct objects (sofa, floor, plant, TV) and wrong for much of the rest: on the lounge the most common name is "blind rail" (the far window wall), white walls often come out as "whiteboard" and the stools as "table".
On garden the ground is named "table", even with the table it surrounds inpainted out of its crops; the cause is untested.

The clip flies the same path as [SuRFLo's](../surflo/README.md); two IGGT bugs had to be fixed first (see [Traps](#traps)).

## Measured with pre-registered metrics (E08m, E08l)


- **More frames:** neither VGGT's cameras (Mip-NeRF 360, AUC@30, median over seven scenes, 0.986 at 24 frames, its training maximum, and 0.987 at 128) nor IGGT's instances (up to 32 frames, ScanNet++ ground truth on scenes from its training data) degrade past their training range ([`results/iggt/`](../../results/iggt/)).
- **SuRFLo on IGGT's backbone:** no longer the same SuRFLo, since every guided metric moved beyond seed noise on every scene: slightly worse on garden, better on the ScanNet++ scenes IGGT was trained on ([`e08l_iggt_backbone.json`](../../results/surflo/e08l_iggt_backbone.json)).

## What it is

VGGT-1B's architecture fine-tuned (backbone learning rate 1e-6) with an instance head that reads the aggregator's layers 4, 11, 17 and 23 (the layers SuRFLo reads) and the point head's features, and predicts an 8-dim instance feature per pixel.
Instances come from clustering those features (its demo: 3D kNN smoothing, then HDBSCAN). The open-vocabulary step in the paper (OpenSeg/LSeg/CLIP mask pooling) is not released.

## Running it

```bash
hf download lifuguan/IGGT_official iggt_checkpoint.pth --revision 12aa6fa --local-dir data/models/iggt
python -m gs_playground.iggt.instances infer --folder <images> --n-images 12 --out <npz>    # surflo env
python -m gs_playground.iggt.instances render --npz <npz> --out <dir>                       # puffin env
bash experiments/iggt/run_E08m_view_count.sh                                                # E08m
```

It runs in the SuRFLo env, with no install of its own requirements: `third_party/patches/iggt_inference-shims.patch` replaces the four imports its model only touches trivially (transformers, basicsr, detectron2, apex).

## Traps

- Its demo loads the weights with `strict=False`; ours load strictly.
- Above 12 frames its point and instance heads break in their frame-chunking path; `iggt_many-frames.patch` runs them on all frames at once, which is the path it already takes for 12 or fewer (E08m checked this bitwise, and that the heads act per frame).
- The instance head needed an even patch grid (504x336); `iggt_any-grid.patch` restores HAT's padding, so it now runs on SuRFLo's 518-wide grids too (bitwise unchanged at 504x336).
- On SuRFLo, one encoding serves both: run SuRFLo with `vggt_weights=<IGGT>` and `save_vggt_tokens=true`, then `python -m gs_playground.iggt.on_surflo features` reads those tokens (E08n; against IGGT's own forward, per-pixel cosine median 1.000000, minimum 0.99966).
- Its demo clustering is fragile: without HDBSCAN's `cluster_selection_epsilon` it shatters garden into 290 pieces; with the demo's 0.06 it merges table, pot and ground.
  On ScanNet++ the features separate ground-truth instances far better (centroid mIoU 0.71-1.00) than the clustering recovers them (T-mIoU 0.24-0.74) (E08m).
- Its backbone is not a drop-in for VGGT-1B under SuRFLo: SuRFLo's output changes, better on ScanNet++ (IGGT's training data), slightly worse on garden (E08l).
- All ScanNet++ `nvs_sem_val` scenes are in its released training archive.
