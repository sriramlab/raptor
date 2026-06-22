# Raptor Example Scripts

These scripts are small templates for common Raptor embedding workflows. They are intended to be copied or edited for local paths, projectors, manifests, and output folders.

## Scripts

* `embed_medmnist_npz.sh`: embeds a MedMNIST-style `.npz` blob with DINO/DINOv2.
* `embed_oct_paths_dinov3.sh`: embeds OCT folders, `.npz` files, or `.zip` files listed as absolute paths.
* `cache_oct_volumes.sh`: optionally caches normalized, downsampled OCT volumes before embedding.

## Notes

Create a projector whose first dimension matches the encoder token dimension:

```bash
python create_projector.py --seed 0 --d 1024 --k 100 --saveas data/proj_normal_d1024_k100_run1
python create_projector.py --seed 0 --d 4096 --k 100 --saveas data/proj_normal_d4096_k100_run1
```

Use `--resize_mode global_pad` when the relative geometry of all A/C/S views should be preserved. Use `--resize_mode stretch` to reproduce the original square-resize behavior.
