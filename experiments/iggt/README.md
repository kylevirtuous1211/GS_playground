# IGGT

Li et al., "IGGT: Instance-Grounded Geometry Transformer", ICLR 2026, arXiv 2510.22706.
Code `lifuguan/IGGT_official`, pinned in `third_party/PINS.tsv`; weights HF `lifuguan/IGGT_official` (`iggt_checkpoint.pth`, revision 12aa6fa), derived from the CC BY-NC VGGT-1B, research only.

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
- Its demo clustering is fragile: without HDBSCAN's `cluster_selection_epsilon` it shatters garden into 290 pieces; with the demo's 0.06 it merges table, pot and ground. On ScanNet++ the features separate ground-truth instances far better (centroid mIoU 0.71-1.00) than the clustering recovers them (T-mIoU 0.24-0.74) (E08m).
- Its backbone is not a drop-in for VGGT-1B under SuRFLo: SuRFLo's output changes, better on ScanNet++ (IGGT's training data), slightly worse on garden (E08l).
- All ScanNet++ `nvs_sem_val` scenes are in its released training archive.
