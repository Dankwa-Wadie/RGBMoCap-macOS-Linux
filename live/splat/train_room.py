"""Train a room Gaussian splat on Apple Silicon (msplat, Metal) and keep a renderable checkpoint.

    .venv-splat/bin/python live/splat/train_room.py COLMAP_DIR OUT_PREFIX [ITERATIONS]

COLMAP_DIR holds images/ and sparse/0 (live/splat/colmap_solve.py). Coordinates stay in
COLMAP's frame (keep_crs) so live/splat/align.py's matrix applies. Writes OUT_PREFIX.ply
and OUT_PREFIX.ckpt; prints "TRAIN step/total" progress.
"""
import sys, time
import msplat

data, prefix = sys.argv[1], sys.argv[2]
iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
config = msplat.TrainingConfig(); config.keep_crs = True; config.iterations = iterations
dataset = msplat.load_dataset(data)
trainer = msplat.GaussianTrainer(dataset, config)
start = time.monotonic()
trainer.train(lambda stats: print(f'TRAIN {trainer.iteration}/{iterations} splats {trainer.splat_count} '
                                  f'{time.monotonic() - start:.0f}s', flush=True), 500)
trainer.save_checkpoint(prefix + '.ckpt'); trainer.export_ply(prefix + '.ply')
print('TRAINED', trainer.splat_count, 'splats in', round(time.monotonic() - start), 's', flush=True)
