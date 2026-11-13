# LungHist700: protocol, not architecture

Code and results for a re-evaluation of the public **LungHist700** lung
histopathology dataset, testing whether the 89–98% accuracies reported for it
measure model capacity or evaluation design.

The manuscript is under review and is not included here. Everything needed to
check what was done — the pipeline, every result table, every figure, and the
per-sample out-of-fold predictions behind the calibration and confusion
analyses — is in this repository.

---

## The headline result

One pipeline. Identical ResNet-50 tile-mean features, an identical linear
classifier, identical hyperparameters, identical seeds. The **only** thing that
changes is what may be shared between the training and test partitions.

| Task | Patient-disjoint | Image-level | Inflation |
|---|---|---|---|
| 3-class diagnosis | **0.722** | 0.922 | **+20.0 pts** |
| 3-class grading | **0.492** | 0.786 | **+29.5 pts** |
| 7-class subtype + grade | **0.462** | 0.794 | **+33.2 pts** |

Balanced accuracy, *k* = 8 patient-disjoint folds × 5 seeds. Cohen's *d* > 2 and
Holm-adjusted *p* < 10⁻⁵ in every comparison
(`results/tables/leakage_inflation.csv`).

Exchanging one frozen encoder for another under the correct protocol moves the
same numbers by 14.8, 1.6 and 3.5 points respectively. **The split matters more
than the model, and overwhelmingly so once the task is hard.**

Best patient-disjoint results (Phikon ViT-B/16, tile-mean + global):
3-class **0.875**, grading **0.515**, 7-class **0.535**. The last two have no
published precedent.

---

## Verifying a claim

Every number in the manuscript is generated, never typed. To check one:

| Claim | File |
|---|---|
| Protocol comparison, all tasks/heads | `results/tables/summary_e1_leakage.csv` |
| Paired inflation, effect sizes, *p* values | `results/tables/leakage_inflation.csv` |
| **Per-fold patient/image overlap** (proof the splits are sound) | `results/tables/e1_leakage_folds.csv` |
| Encoder comparison | `results/tables/summary_e2_backbones.csv` |
| Aggregation heads, attention-MIL | `results/tables/summary_e4_heads.csv` |
| Fusion / pooling ablation | `results/tables/e5_fusion_folds.csv` |
| Stain normalisation (Macenko vs raw) | `results/tables/summary_e3_stain.csv` |
| Calibration, ECE, temperature scaling | `results/tables/calibration_full7.csv` |
| Risk–coverage (abstention) | `results/tables/risk_coverage_full7.csv` |
| Cross-magnification transfer | `results/tables/magnification_transfer_*.csv` |
| Patient learning curve | `results/tables/learning_curve_*.csv` |
| Attention faithfulness (deletion/insertion) | `results/tables/attribution_faithfulness_*.csv` |
| LC25000 group-recovery attempt | `results/tables/lc25000_group_recovery.csv` |
| Audit of the eight prior studies | `results/tables/related_work.csv` |
| Dataset audit (counts, duplicates, fold bound) | `results/tables/dataset_audit.txt` |

The `*_preds.csv` files hold per-sample out-of-fold predictions with
probabilities, so the confusion matrices, calibration curves and
risk–coverage curves can be recomputed independently of our code.

`results/RESULTS.md` is the generated report tying these together.

---

## Reproducing from scratch

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# 1. fetch the public data (LungHist700 from figshare, LC25000 parquets)
./code/features/fetch_data.sh          # if absent, see "Data" below

# 2. audit the dataset and build the image index
./code/run_study.sh audit

# 3. encode once per (encoder, stain) pair  — the long step, CPU-only
./code/features/extract_all.sh

# 4. every experiment, in order
./code/run_study.sh e1 e2 e3 e4 e5 transfer curve attribution external report
```

`run_study.sh` takes stage names, so a single stage can be re-run on its own.
Each stage writes its own tables and is restartable: encoding is checkpointed
per block into `_shards/`, and the benchmark driver writes its CSV after every
configuration.

**Hardware.** Everything here ran on a CPU-only laptop (Intel Core Ultra 7
155H, 22 threads, 30 GB RAM). `code/thermal_run.py` bounds the thread pool,
duty-cycles the job and suspends it when the processor package gets too hot;
without it the machine shuts down mid-pass. Full encoding takes several hours
per encoder.

---

## Data

Not included — it is public and large (≈3 GB).

- **LungHist700**: 691 images, 45 patients.
  Diosdado *et al.*, *Scientific Data* 11:1088 (2024),
  <https://doi.org/10.1038/s41597-024-03944-3>; data on figshare,
  <https://doi.org/10.6084/m9.figshare.25459174>.
- **LC25000**: Borkowski *et al.*, arXiv:1912.12142 (2019).

Place LungHist700 under `data/data/` and the LC25000 parquets under
`data/lc25000/`.

---

## Layout

```
code/
  run_study.sh            stage driver: audit e1 e2 e3 e4 e5 transfer curve
                          attribution external report
  thermal_run.py          thermal governor for CPU-only long runs
  reap_orphans.sh         removes workers orphaned by an interrupted run
  preprocess/             dataset audit, image index
  features/               tiling and frozen-encoder extraction
  models/heads.py         linear, MLP, gated attention-MIL heads
  experiments/
    protocols.py          the three splitters, metrics, fold checks
    run_benchmark.py      the task × protocol × representation × head grid
    run_fusion.py         encoder fusion and pooling ablation
    transfer_and_curves.py  magnification transfer, patient learning curve
    attribution.py        attention faithfulness by deletion/insertion
    external_lc25000.py   augmentation-group recovery attempt
  analysis/               aggregation, figures, report and table generators
results/
  tables/                 every CSV cited above
  figures/                every figure, as PDF and PNG
```

---

## Two negative results

Both are reported in the manuscript and both are reproducible here.

**LC25000's augmentation grouping cannot be recovered.** Rotations and flips
leave an image's pixel multiset unchanged, so a per-channel intensity histogram
should be an exact invariant of the augmentation group and should return 250
groups of twenty per class. It returns ≈4,500 groups with a median size of 1
and no group of the documented size
(`results/tables/lc25000_group_recovery.csv`). The released images are
therefore not related by pixel-permuting transforms alone, and we report no
cross-dataset result.

**Macenko stain normalisation lowers accuracy** on this cohort, by 4.7 points
on the three-class task and 6.3 on the seven-class task
(`results/tables/summary_e3_stain.csv`). Every image comes from one laboratory
and one scanner, so there is little stain variation to remove and the transform
discards usable signal. This also serves as a control: the protocol effect is
not an artefact of stain variation, because removing that variation does not
reduce it.

---

## Note on scope

Two planned analyses were not completed, and the manuscript says so: a
natural-image self-supervised encoder (DINOv2) whose encoding pass did not
finish within the compute budget, and stain normalisation beyond a single
normaliser on a single encoder.

---

## License

Code released under the MIT License (see `LICENSE`). The datasets are
distributed by their original authors under their own terms.
