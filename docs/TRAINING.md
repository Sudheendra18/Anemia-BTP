# Stage 3 — Baseline Model Training

```bash
python scripts/train_baseline.py
```

Reads everything from `configs/config.yaml` → `training:`. Runs Stages 1–2's
output through patient-level stratified k-fold cross-validation on a
transfer-learning CNN, and writes per-fold + aggregated results.

## Before you run it

- **Run Stages 1 and 2 first** (`python scripts/run_pipeline.py`) — this
  stage reads `cp_anemic_metadata_with_roi.csv` (or the plain metadata CSV,
  see below), it doesn't touch images or the Excel sheet directly.
- **First run downloads pretrained ImageNet weights** from
  `download.pytorch.org` (a few hundred MB for ResNet-50) — needs normal
  internet access once, then they're cached locally.
- **GPU is used automatically if present** — see the dedicated section
  below. If you only have CPU, start with `freeze_backbone: true` (only
  trains the new classification head, much faster) to get a first baseline
  number.
- **Crashes don't cost you the whole run** — see "Recovering from a crash"
  below. This matters most for full fine-tuning (`freeze_backbone: false`)
  on CPU, which is memory-heavy and the most likely thing to crash.

## Recovering from a crash

If training dies partway through (out of memory is the most common cause
for full fine-tuning on CPU), just run the exact same command again:

```bash
python scripts/train_baseline.py
```

It will:
- **Skip every fold that already finished completely** — instant, no retraining.
- **Resume the fold that was interrupted from the epoch after the last one it completed** — not from scratch. Model weights, optimizer state, and training history all pick up exactly where they left off.
- **Never lose the best model found so far** — the best checkpoint (and its metrics/plots) are written to disk the moment a new best epoch is found, not deferred until the fold finishes. A crash can cost at most one epoch of progress.

If you deliberately want to retrain everything from scratch (e.g. after
changing `architecture`, `image_size`, or `freeze_backbone` in the config —
changes like these make old checkpoints incompatible, so they get detected
and ignored automatically, but this flag skips that check overhead entirely
and starts clean):

```bash
python scripts/train_baseline.py --fresh
```

If a crash produces an actual error message (not just the process quietly
disappearing), it's worth sharing that message too — the checkpointing
protects your progress either way, but the message may point at a real
fix (e.g. lowering `batch_size` further, or switching `architecture` to
`efficientnet_b0`, which has far fewer parameters than `resnet50` and is
noticeably lighter to fine-tune on CPU).

## GPU usage

The device is picked automatically — no config flag needed. On a machine
with an NVIDIA GPU (and a normal `pip install torch` with CUDA support),
you'll see this at the top of the log:

```
Using device: cuda (NVIDIA GeForce RTX 3060, 12.0 GB)
```

If it instead prints `Using device: cpu`, PyTorch isn't seeing your GPU —
usually a driver/CUDA-version mismatch, or a CPU-only torch build got
installed. Check with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If that prints `False`, reinstall torch using the exact command for your
CUDA version from https://pytorch.org/get-started/locally/ rather than a
plain `pip install torch`. A free Colab GPU runtime also works fine for
this dataset size if you don't have local GPU access.

When a CUDA GPU is detected, the harness automatically:
- Enables **mixed precision** (`training.mixed_precision: true` by default) — trains in fp16 where safe, roughly halves memory use and speeds up training on most NVIDIA GPUs with no meaningful accuracy cost at this scale. Turn it off in the config only if you see NaN losses.
- Turns on **cuDNN autotuning** (`cudnn.benchmark`) — safe here since every batch uses the same fixed `image_size`.
- Uses **pinned memory + non-blocking transfers** in the data loaders, so copying batches to the GPU overlaps with compute instead of blocking it.

None of this needs touching — it's conditional on `device.type == "cuda"`
throughout, so the exact same config and command run correctly (just
slower) on a CPU-only machine, which is what this was smoke-tested on.

## What's configurable (`configs/config.yaml` → `training:`)

| Key | What it does |
| --- | --- |
| `image_source` | `"with_roi"` trains on the Stage 2 crops, `"raw"` trains on the uncropped images. Switch this once you've visually confirmed which crop quality you trust — no code changes needed. |
| `architecture` | `"resnet50"` or `"efficientnet_b0"`. |
| `freeze_backbone` | `true` = only train the new classifier head (fast, good first sanity check). `false` = full fine-tuning (better ceiling, slower, needs more data/epochs to not overfit). Default is now `false` (full fine-tuning), with `batch_size: 8` and `num_workers: 0` — deliberately conservative to reduce CPU memory pressure. Raise `batch_size` if you have RAM/VRAM to spare and want faster epochs. |
| `k_folds` | Number of cross-validation folds. |
| `primary_metric` | Drives early stopping and best-checkpoint selection per fold. Default `f2` — see below for why. |
| `use_class_weights` | Reweights the loss to counteract the 424 Anemic vs 286 Non-anemic split. Leave on unless you have a specific reason not to. |
| `mixed_precision` | fp16 training on CUDA GPUs for speed/memory. Ignored on CPU/MPS. See GPU section above. |
| `lr_scheduler` | `"none"`, `"cosine"` (default — smooth decay to ~0 over `num_epochs`), or `"plateau"` (halves LR when `primary_metric` stalls). See "Getting the best result" below. |

## Why F2 as the primary metric

This is a screening tool — a **false negative** (calling an anemic patient
non-anemic) is worse than a false positive (flagging a healthy patient for a
confirmatory blood test). F-beta with beta=2 weights recall (sensitivity)
higher than precision, so early stopping and checkpoint selection favor
models that catch more true anemic cases, even at some cost to precision.
If your supervisor wants a different tradeoff, `sensitivity` or `f1` are
also available as `primary_metric`.

## Why patient-level (group) k-fold

Splitting is `StratifiedGroupKFold` on `patient_id`, not a plain
`StratifiedKFold` on rows. If the same patient ever contributed more than
one image, a naive split could put two images of the *same person* into
both train and validation — the model could "cheat" by recognizing that
patient's specific coloring rather than learning anemia signal in general,
inflating the reported metrics. Right now `patient_id == image_id` in this
dataset (see `docs/DATA_CONTRACT.md`), so this currently behaves exactly
like a plain stratified split — but the code is already correct if that
assumption changes, with nothing to rewrite.

Each fold run in `run_kfold_cv()` asserts zero patient overlap between its
train and validation sets before training even starts, so this isn't just
trusted — it's checked every run.

## Reading the output

```
outputs/training/
├── baseline_results_table.csv     # one row per fold + mean/std summary row
└── fold_1/
    ├── best_model.pt              # checkpoint from the best epoch (by primary_metric)
    ├── metrics.json               # that checkpoint's full metric set
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── training_history.png       # loss + primary metric curves, for spotting over/underfitting
    └── resume_state.pt            # only present while this fold is still in progress — removed once it finishes
```

`baseline_results_table.csv` is the "baseline results table" deliverable
from the Work Package A spec — this is what goes in the reproducibility
writeup. The `mean`/`std` rows are what you'd actually report as the
model's performance, not any single fold.

## Sanity checks worth doing on your first real run

1. **Training loss should trend down.** If it doesn't move at all, something's
   probably wrong upstream (check the ROI crops aren't blank/corrupted, or try
   a higher learning rate).
2. **Compare `with_roi` vs `raw` as a first experiment.** If ROI-cropped
   images perform noticeably worse than raw images, that's a signal the crop
   thresholds need retuning (`docs/ROI_TUNING.md`) before trusting them
   for the real benchmarking runs.
3. **Watch the val loss vs train loss gap** in `training_history.png` — a
   widening gap while train loss keeps dropping is classic overfitting,
   common on a dataset this size with full fine-tuning. `freeze_backbone:
   true`, more augmentation, or fewer epochs are the usual first fixes.

## Getting the best result (multi-hour budget)

If you have hours available rather than needing one quick run, two things
in the config are worth using instead of guessing at a single
architecture/learning-rate combination:

**Learning rate schedule** (`training.lr_scheduler`) — `"cosine"` is now the
default: LR smoothly decays from `learning_rate` to ~0 across `num_epochs`,
which usually beats a flat LR once you can afford enough epochs for it to
matter. `"plateau"` instead halves LR whenever `primary_metric` stalls for 3
epochs — a reasonable alternative if you're not sure how many epochs you'll
actually need. Both are resume-safe: the scheduler's internal state is
saved and restored exactly like the model and optimizer, so a crash mid-fold
doesn't reset your LR schedule back to the start.

**Systematic architecture/LR comparison**, instead of one full run:

```bash
python scripts/run_experiments.py
```

This runs every entry under `experiments.runs` in the config back to back —
by default, ResNet-50 and EfficientNet-B0 each at two learning rates — and
ranks them by `training.primary_metric` in a leaderboard. Why this is worth
the extra compute: EfficientNet-B0 has far fewer parameters than ResNet-50,
and with only 710 images, the lighter model may generalize better rather
than just being a fallback for when ResNet-50 is too slow.

Each experiment gets its own isolated `outputs/experiments/<name>/`
directory and reuses the exact same crash-resilient training code as a
normal run — same per-fold checkpointing, same resume behavior. If the
whole thing gets interrupted, rerun the same command: experiments that
finished are skipped instantly, the interrupted one resumes mid-fold, and
anything not yet started begins normally. Nothing extra to manage.

Output:

```
outputs/experiments/
├── leaderboard.csv                     # one row per experiment, sorted best-first by primary_metric
├── resnet50_lr1e-4/
│   ├── effective_config.yaml           # exact config this experiment used — pass to any other script's --config
│   ├── baseline_results_table.csv
│   └── fold_1/ ... fold_5/
└── efficientnet_b0_lr3e-4/
    └── ...
```

Once you have a winner, run threshold tuning against it specifically:

```bash
python scripts/analyze_thresholds.py --config outputs/experiments/<winning-name>/effective_config.yaml
```

(`analyze_thresholds.py` and `train_baseline.py` both accept any config
path — the experiment runner's `effective_config.yaml` is just a regular
config with that experiment's overrides already baked in, so you never have
to hand-reconstruct what combination produced the best result.)

To customize the search — different learning rates, adding `weight_decay`
or `optimizer` overrides, more or fewer combinations — edit
`experiments.runs` in `configs/config.yaml` directly; each entry is just a
name plus a dict of `training:` keys to override.

**Adding experiments to an existing run is safe and additive.** If you've
already run `run_experiments.py` and add new entries to `experiments.runs`,
rerunning only trains the new ones — every already-completed experiment is
recognized by its existing `metrics.json` files and skipped instantly, same
as the crash-resume behavior for a single training run. This is how the two
regularization experiments below were added on top of an already-finished
4-way comparison without redoing any of it.

### Regularization follow-up (weight_decay, augmentation)

If a training log shows the "some overfitting" warning consistently across
folds — a wide, persistent gap between train and val loss at the best
epoch — it's worth a targeted follow-up rather than just accepting it. Two
new keys make this tunable per-experiment:

- `training.weight_decay` — already existed, but is now a natural thing to
  override per experiment (e.g. 5x the default).
- `training.augmentation` — `rotation_degrees`, `color_jitter`, and
  `random_erasing_prob` (0 by default, i.e. off). `RandomErasing` blanks a
  random rectangular patch of the image each draw — a standard, well-tested
  regularizer that doesn't have the domain-plausibility concerns a change
  like vertical-flipping a conjunctiva photo would.

Two experiments in the default config use these, both built on
`efficientnet_b0_lr1e-4` specifically — it was the best-balanced result
across the first four (highest accuracy, highest AUC, best specificity,
lowest fold-to-fold variance), so it's the one worth protecting from
overfitting rather than starting the regularization search from scratch:

- `efficientnet_b0_lr1e-4_wd5e-4` — same config, `weight_decay` raised from 0.0001 to 0.0005.
- `efficientnet_b0_lr1e-4_wd5e-4_aug` — same weight_decay increase, plus stronger augmentation (`rotation_degrees: 20`, `color_jitter: 0.3`, `random_erasing_prob: 0.25`).

Compare their rows in `leaderboard.csv` against the original
`efficientnet_b0_lr1e-4` row: a smaller train/val loss gap (visible in each
experiment's `fold_*/training_history.png`) without a meaningful drop in
F2/AUC would confirm the regularization helped; a drop in performance would
mean these particular settings were too aggressive for this dataset size —
either result is a legitimate, reportable finding for the writeup.

## Threshold tuning (no retraining needed)

The training harness's reported metrics use an implicit 0.5 cutoff (argmax
on the softmax output). That cutoff is not necessarily the best operating
point, especially if you're seeing a sensitivity/specificity imbalance —
class-weighting the loss function encourages the model to lean toward the
minority class, but it doesn't move where the decision boundary sits. Those
are two separate levers.

Once you have trained fold checkpoints (`outputs/training/fold_*/best_model.pt`),
you can tune the threshold without retraining anything:

```bash
python scripts/analyze_thresholds.py
python scripts/analyze_thresholds.py --sensitivity-floor 0.85   # default is 0.90
```

This loads each fold's already-trained `best_model.pt`, runs one inference
pass over that fold's held-out validation images (fast — no backprop), and
pools every fold's validation predictions into one out-of-fold (OOF) set —
every image scored by a model that never saw it during training. It then
sweeps thresholds from 0.05 to 0.95 and reports three different recommended
operating points, since "optimal" depends on what you're optimizing for:

| Recommendation | What it does |
| --- | --- |
| `max_f2` | Whatever threshold maximizes F2 (matches what training already optimizes for). |
| `max_youden_j` | Maximizes sensitivity + specificity − 1 — the standard "best overall balance" criterion in diagnostic-test literature. |
| `max_specificity_at_sensitivity_floor` | The most clinically motivated for a screening tool: hold sensitivity at or above `--sensitivity-floor`, get the best specificity available without dropping below it. |

Output:

```
outputs/training/
├── oof_predictions.csv              # every image's out-of-fold probability — patient_id, label, fold, y_prob
├── threshold_sweep.csv              # full metric set at every threshold from 0.05–0.95
├── threshold_sweep.png              # sensitivity / specificity / F2 vs threshold, recommended point marked
└── confusion_matrix_recommended_threshold.png
```

If a fold's checkpoint is missing (e.g. training was interrupted and hasn't
reached that fold yet), that fold is skipped with a warning rather than
failing the whole analysis — you'll just get OOF predictions for the folds
that do have a checkpoint, clearly logged as a partial result.

**To actually use a different threshold**, anywhere you write real
inference/deployment code later, route the prediction through
`metrics.apply_threshold(y_prob, threshold)` instead of `argmax` — that's
the one place in the codebase that should encode what the decision rule is.

## Not built yet

- Decision-threshold tuning (currently a fixed 0.5 cutoff on the softmax
  probability — since F2 already favors recall via the loss weighting and
  metric choice, lowering the threshold further is a natural next
  experiment once you have a baseline number to compare against).
- The cross-dataset generalization run (CP-AnemiC ↔ Indian dataset) — this
  harness trains and evaluates on one dataset; the cross-dataset script is
  a thin wrapper around the same `build_model` / `evaluate` functions once
  the second dataset has its own metadata CSV in the same schema.
