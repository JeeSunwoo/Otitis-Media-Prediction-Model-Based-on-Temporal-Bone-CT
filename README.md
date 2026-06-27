# Otitis Media Prediction from Temporal Bone CT

**A staged left/right‑comparison deep‑learning pipeline for slice‑level otitis media detection — built to *diagnose* severe right‑ear (Rt) class imbalance, not merely to patch it.**

`Python` · `PyTorch` · `EfficientNet‑B0` · `Transformer` · `5‑Fold CV`

> **Official score:** macro **F1 0.8656** (86.6 / 100) · accuracy **87.14%** — exact row matching over 3,320 slices.
> **Honest headline:** the overall score is strong, but **Rt otitis stalls at F1 0.6582**. A failure analysis shows this is **not** a lack of positive samples — it is a *structural mismatch* between a left/right‑asymmetry detector and a target class that is **~97% bilateral**, so the asymmetry signal vanishes exactly where it is needed.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data & the Imbalance Problem](#data--the-imbalance-problem)
- [Method](#method)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Training Setup](#training-setup)
- [Results](#results)
- [Why Rt Otitis Fails (the interesting part)](#why-rt-otitis-fails-the-interesting-part)
- [Limitations & Future Work](#limitations--future-work)
- [Reference](#reference)

---

## Overview

The task is to predict **otitis media** from **temporal bone CT**. Inputs are per‑patient DICOM slices (up to 132 per patient); labels are given per **patient × left/right (R/L) × item (temporal area / otitis media)**, marking lesion presence as `0/1` for each slice.

In essence this is **not** single‑image classification but a **multi‑label sequence classification** problem: for a patient's continuous CT slice sequence, judge otitis positive/negative **per slice, per ear**. The evaluation metric is the **macro F1‑score** (plus left/right positive F1), chosen for robustness to class imbalance.

Three design pillars:

1. **Staged pipeline** — borrow the multi‑stage structure of an otoscope‑image diagnosis paper and adapt it to CT.
2. **Physician‑emulating design** — map a radiologist's reading process (multi‑window → temporal localization → left/right comparison → continuous‑slice inference) one‑to‑one onto model components.
3. **Imbalance head‑on** — apply a full‑stack of countermeasures for Rt otitis, and ultimately *explain* why they cannot close the gap.

---

## Architecture

![Overall architecture](assets/architecture.png)

*Two‑stage pipeline: shared preprocessing → **TemporalModel** (produces a temporal‑span **gate**) → **OtitisModel** (left/right spatial comparison) → **gated fusion** → per‑slice prediction. At inference, 5 folds are ensembled by logit averaging before sigmoid.*

---

## Data & the Imbalance Problem

Before modeling, the train/val distribution was analyzed. Of the four items (Rt/Lt temporal, Rt/Lt otitis), **only Rt otitis is markedly imbalanced** — the central difficulty of the project and the motivation for the entire design.

![Distribution analysis](assets/distribution_analysis.png)

*Left — slice‑level positive rate per item. Right — positive rate along the slice axis (1–132).*

| Evidence | Observation |
|---|---|
| **Slice positive rate** | Rt otitis ≈ **28% (train) / 14% (val)** — lowest of the four items; the others are all ≈ 70%+. The large train↔val gap means train‑overfit collapses on val. |
| **Left/right asymmetry** | Even within otitis, **Lt ≈ 70%+ (common)** vs **Rt 14–28% (scarce)** → the apparent basis for a left/right (diff) comparison design. |
| **Span concentration** | Lesions concentrate in the **central slices (~40–60)** and are ≈ 0 at the ends → justifies a *gate* that judges only inside the temporal span. |
| **Weak Rt signal** | The Rt otitis curve stays ≤ **45%** even centrally, vs ~100% for temporal/Lt → the positive signal is weak at the slice level, not just patient‑level. |

---

## Method

### 1 · Staged pipeline (adapted from otoscope diagnosis → CT)

| Otoscope‑paper stage | CT counterpart | Implementation |
|---|---|---|
| Image‑quality assessment | Noise‑slice filtering | MAD‑based outlier detection on inter‑slice differences (remove "spiking" slices) |
| Tympanic‑membrane segmentation | Temporal‑area span detection | **TemporalModel** detects the positive span over the sequence (no mask GT needed) |
| Laterality determination | L/R half‑split + Lt horizontal‑flip alignment | CSV Rt/Lt labels combined with image half‑split |
| Disease classification | Otitis classification | **OtitisModel** — a left/right‑comparison classifier *gated* by the temporal span |

The key benefit: the otitis classifier is constrained to decide **only inside the detected temporal span (gate)**, directly mitigating the overwhelming surplus of negative slices.

### 2 · Physician‑emulating components

- **Multi‑window** → restore HU and feed a **3‑channel input** of three windows (bone / soft‑tissue / air), like a radiologist toggling windows.
- **ROI localization** → TemporalModel finds the temporal span first; otitis is judged only within that **gate**.
- **Left/right comparison** → split and align L/R, then hard‑code an **asymmetry signal (diff = self − other)** into the input.
- **Continuous‑slice inference** → a **Transformer** reads the whole slice sequence; post‑processing removes spans cut off too short.

### 3 · Imbalance countermeasures (full‑stack)

Applied across the whole pipeline — *data split → loss → training range → architecture → evaluation/threshold*:

- **Stratified K‑Fold** by Rt‑positive status → balanced folds, stable model selection.
- **Loss:** `pos_weight` (√ratio, cap 8) · **Focal BCE** (γ=2) · **per‑side loss** (Rt 1.0 / Lt 0.8).
- **Candidate masking:** train mostly on the temporal span ±5 slices + 30% hard negatives.
- **Architecture:** spatial left/right comparison (`_SpatialSideEncoder`) + antisymmetric **diff branch** (`Rt = logit + α·d`, `Lt = logit − α·d`).
- **Evaluation:** best selected by **L/R positive F1** (not macro); thresholds derived from training‑data ensemble sweep (no val peeking).
- **Generalization:** strong augmentation, Siamese weight sharing, freeze→unfreeze, dropout/weight‑decay, 5‑fold ensemble, early stopping.

---

## Preprocessing

Train and inference **share one preprocessing module** to eliminate train/inference skew.

- **HU restoration** — apply DICOM `RescaleSlope/Intercept` → physically meaningful density values.
- **Multi‑window 3 channels** — `ch0` bone (WL 700 / WW 2000), `ch1` soft tissue (50/350), `ch2` air/wide (−300/1400).
- **L/R split + alignment** — split axial into halves; horizontally flip the **Lt** half so both ears face the same direction.
- **Resize / normalize / augment** — 224×224, ImageNet stats; augmentation (rotation ±13°, translation ±7%, brightness/contrast 0.75–1.25, Gaussian blur) applied identically to L/R/view.

---

## Models

### TemporalModel — temporal‑area span detection
EfficientNet‑B0 GAP features → per‑slice `[Lt feat ‖ Rt feat ‖ diff ‖ slice‑pos]` + positional embedding → Transformer encoder → per‑slice `[Rt_temporal, Lt_temporal]`. Post‑processing (median smooth → gap‑fill → short‑span removal) confirms the span, which is **dilated by ±5 slices** and passed downstream as the **gate**.

### OtitisModel — left/right spatial‑comparison classification
The judgment happens **on top of** a left/right spatial comparison (not post‑hoc):

- **Spatial backbone** — EfficientNet‑B0 *pre‑GAP* map (7×7) → 1×1 conv; location preserved (gradient checkpointing after unfreeze).
- **main path** — `_SpatialSideEncoder(self, self − other)` → shared `_SeqTransformer` (Siamese).
- **diff path** — `_DiffBranch` → antisymmetric scalar `d` (tanh, `|d| < 1`, α learned ≥ 0.2).
- **combine** — `Rt = logit + α·d`, `Lt = logit − α·d` → `[Rt_otitis, Lt_otitis]`.

Outputs are thresholded, **AND‑combined with the temporal gate**, then gap‑filled / short‑span‑cleaned into the final prediction.

---

## Training Setup

| Item | Setting |
|---|---|
| Backbone | EfficientNet‑B0 (ImageNet pretrained) |
| Cross‑validation | 5‑Fold (otitis stratified by Rt‑positive) |
| Batch / input | 2 patients/batch, 224×224, 3 channels (multi‑window) |
| Epochs / Early stopping | up to 50 epochs, patience 12 |
| Backbone freeze | freeze first 8 epochs, then unfreeze |
| Optimizer | AdamW (weight decay 1e‑3) |
| Learning rate | head 2.5e‑5 / backbone 1e‑5 |
| LR schedule | warmup 3 epochs + cosine annealing |
| Loss | Focal BCE (γ=2) + pos_weight (√ratio, cap 8) |
| L/R loss weighting | Rt 1.0 / Lt 0.8 (per‑side loss) |
| Auxiliary loss | diff branch smooth‑L1, λ=0.5 |
| Augmentation | rotation ±13°, translation ±7%, brightness/contrast 0.75–1.25, blur |
| Imbalance handling | stratified fold + candidate mask (±5) + hard‑neg 30% |
| Ensemble / threshold | 5‑fold logit average; per‑side independent sweep on training data |
| Other | AMP, gradient checkpointing, SEED 42 |

---

## Results

![Per-item results scorecard](assets/results_scorecard.png)

**Class-level breakdown** (all 3,320 slices viewed as Normal(0) / Otitis(1)):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal (0) | 0.85 | 0.82 | 0.84 | 1,339 |
| Otitis (1) | 0.88 | 0.90 | 0.89 | 1,981 |
| **macro avg** | 0.87 | 0.86 | **0.8656** | 3,320 |
| accuracy | — | — | **0.8714** | 3,320 |

The temporal items (0.92 / 0.88) and Lt otitis (0.81) are stable; **only Rt otitis lags (0.6582)**. Its confusion matrix is **TP 72 / FP 131 / FN 46 / TN 581** — with FP > TP, **precision collapses to 0.355** (recall 0.610 is fine). The Rt failure is *over‑calling normal as otitis (FP)*, not *missing cases (FN)* — hence its threshold (0.62) is the highest of the four.

---

## Why Rt Otitis Fails (the interesting part)

Decomposing the failure with extra diagnostics shows the cause is **a mismatch between the design's central assumption and the data structure**, not a simple shortage of positives.

<details>
<summary><b>(1) It is not a train → val generalization collapse</b></summary>

After removing threshold/model‑selection leakage and honestly re‑evaluating 5 folds with a fixed threshold, Rt otitis macro F1 ≈ **0.58**, essentially matching the validation **0.658** (val even slightly higher). The generalization gap is essentially absent — discriminative power is *intrinsically* this low, confirmed by AUROC:

| Item | train‑OOF AUROC | val AUROC |
|---|---|---|
| **Rt otitis** | **0.622** | **0.576** |
| Lt otitis | 0.741 | 0.867 |

Rt AUROC is **near‑random (≈0.6)** even on the *training* held‑out, with no large val drop — so this is intrinsic, not a distribution‑shift artifact (Lt generalizes fine, 0.74 → 0.87).
</details>

<details open>
<summary><b>(2) Root cause — "bilaterality ↔ asymmetry detector" structural mismatch</b></summary>

Section 3 took Rt's "left/right asymmetry" as the basis for the diff design. But that asymmetry was **population‑prevalence** asymmetry (Lt more common), **not** the **within‑patient laterality** the model exploits:

| Bilaterality pattern (otitis) | train | val |
|---|---|---|
| Bilateral‑positive | 37 | 3 |
| Rt‑only positive | 1 | 0 |
| Lt‑only positive | 31 | 6 |
| P(Rt+ \| Lt+) | 0.54 | 0.33 |

**Of 38 Rt‑positive patients, 37 (97%) are bilateral; only 1 is Rt‑only.** Since OtitisModel is essentially a left/right‑asymmetry detector (`d = f(R,L) − f(L,R)`):

- **Signal vanishes** — when Rt is positive (97% bilateral), both ears are opacified → `self ≈ other` → `d ≈ 0`. The primary signal disappears *exactly* on the cases that must be hit.
- **Symmetric–symmetric ambiguity** — bilateral (both abnormal) and normal (both normal) are both symmetric; only **absolute appearance** separates them, but the architecture leans on *relative* evidence and dilutes the absolute signal.

By contrast Lt has **31 Lt‑only** patients → abundant within‑patient laterality → the same design works (Lt AUROC 0.74 → 0.87). The diff branch may even hurt Rt slightly: with `d_target = (Rt − Lt label) ≤ 0` almost always, `d` learns negative, and `α·d` *suppresses* the Rt score (no Rt‑only data to learn a positive signal).
</details>

<details>
<summary><b>(3) Prevalence shift is a secondary factor</b></summary>

The train (≈28%) vs val (≈14%) positive‑rate gap mechanically lowers precision (a Bayes effect, not a discriminative one): projecting a fixed operating point, precision drops ~0.21 from train‑prevalence to val‑prevalence. Since AUROC ≈ 0.6 is prevalence‑independent, this only *aggravates* the precision drop — the primary cause remains (2).
</details>

<details>
<summary><b>(4) Why the imbalance countermeasures missed</b></summary>

`pos_weight`, focal loss, candidate masking, per‑side loss, stratified fold all target **prevalence imbalance (excess negatives)**. But the real bottleneck was the **absence of a discriminative feature** under bilaterality — and **no loss weighting can synthesize a feature that isn't there**. The temporal gate confirms this: it improved Lt otitis (FP 56→37, F1 0.885→0.898) but left Rt's FP/FN/F1 unchanged — Rt's false positives occur *inside* the span (normal Rt middle ear mistaken for otitis), matching diagnosis (2).
</details>

---

## Limitations & Future Work

- **Strengthen the absolute‑appearance signal** — reduce dependence on L/R difference; learn each ear's absolute opacification (remove the α lower bound / add a self‑only branch).
- **Middle‑ear ROI crop** — the lesion is local (middle ear / mastoid); cropping raises absolute‑signal density.
- **Revise model selection** — select best by Rt alone (or per‑item macro), so easy Lt doesn't mask Rt failure.
- **Data** — acquire more **Rt‑only (unilateral)** cases; the left/right comparison design needs unilateral examples to be valid.

---

## Reference

- Staged otoscope‑image diagnosis pipeline — *J. Clin. Med.* 14(23):8572 — https://www.mdpi.com/2077-0383/14/23/8572

---

<sub>Author: Ji Seonwoo · Student ID 2021741045 · Temporal Bone CT Otitis Media Prediction</sub>
