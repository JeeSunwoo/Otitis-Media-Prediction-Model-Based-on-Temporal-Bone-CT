# Otitis Media Prediction Model Based on Temporal Bone CT

*A Staged Left/Right Comparison Pipeline for Slice-Level Otitis Media Detection*

**Project:** Otitis Media Prediction Model · **Name:** Ji Seonwoo · **Student ID:** 2021741045

---

## 1. Project Overview and Objective

The objective of this project is to implement a deep-learning model that predicts the occurrence of otitis media from temporal bone CT images. The input data are per-patient DICOM slices (up to 132 slices), and the ground-truth labels are provided as a CSV that, for each slice (1–132), marks lesion presence as 0/1 at the granularity of patient (No) × left/right (R/L) × item (Image number: temporal area / otitis media).

In essence, therefore, this is not single-image classification but a "multi-label sequence classification problem in which, for a continuous CT slice sequence of a single patient, otitis-media positive/negative is judged per slice for each of the left and right ears." As the evaluation metric we use the F1-score, which is robust to class imbalance (macro F1 and left/right positive F1).

**The core design directions are summarized in the following three points.**

- **(Design skeleton)** We borrow the staged pipeline structure of an existing otoscope-image diagnosis paper into the CT domain.
- **(Reading emulation)** We directly reflect into the model architecture the reasoning process a radiologist actually performs when reading a temporal bone CT (multi-window adjustment → temporal-area localization → left/right comparison → continuous-slice inference).
- **(Imbalance focus)** We apply a series of techniques to confront head-on the severe class imbalance of Rt (right) otitis media revealed in the data analysis.

---

## 2. Rationale for Algorithm (Model) Selection

### 2.1 Borrowing the paper's staged pipeline structure

The referenced otoscope-image-based middle-ear-disease diagnosis study does not apply a single classifier directly; instead it uses a pipeline that decomposes diagnosis into several stages. Representatively, it consists of (1) image-quality assessment → (2) tympanic-membrane region segmentation → (3) laterality determination → (4) disease classification. The key is that each stage narrows the search range of the next, so the final disease classifier focuses only on "regions where a lesion is likely to exist."

Because the CT data in this project have no segmentation mask ground truth (mask GT), we adapted and borrowed the above pipeline into the CT domain as follows.

| Original paper (otoscope image) stage | This project (CT) counterpart | Implementation |
|---|---|---|
| Image-quality assessment | Noise-slice filtering | MAD-based outlier detection on inter-slice differences to remove "spiking" slices |
| Tympanic-membrane segmentation | Detecting the temporal-area slice span | TemporalModel detecting the positive span over the sequence instead of a mask |
| Laterality determination | Left/right half-split + Lt horizontal-flip alignment | Directly combining the CSV's Rt/Lt labels with the image half-split |
| Disease classification | Otitis classification (OtitisModel) | A left/right-comparison classifier using the temporal span as a gate |

*Table 1. Correspondence mapping that converts the reference paper's staged diagnostic pipeline into the CT setting of this project.*

The key benefit of this staged structure is that the otitis classifier is constrained to judge only "within the slice span (gate) where the temporal area is detected," rather than over the whole image. This directly mitigates the situation—described later in the Rt otitis imbalance problem—where negative (normal) slices are overwhelmingly numerous.

> **Reference:** <https://www.mdpi.com/2077-0383/14/23/8572>

### 2.2 Design emulating a physician's actual reading process

On top of the staged structure, we mapped one-to-one—as model components—the reasoning process a radiologist actually performs when reading a temporal bone CT. This is not merely a choice for performance, but a design intent to induce the model to judge on "medically valid grounds."

- **Multi-window adjustment →** A physician views a single image while switching between windows such as a bone window and a soft-tissue window. → Instead of a single PNG, we restore HU and use a 3-channel input made of three different windows.
- **Region-of-interest localization →** A physician does not judge the disease right away, but first finds the slice range where the middle-ear/temporal-bone structures are visible. → TemporalModel first detects the temporal span, and otitis is judged only within that span (gate).
- **Left/right symmetry comparison →** When judging whether one ear is abnormal, a physician necessarily compares it against the opposite (presumed-normal) ear. → After splitting and aligning left/right, we hard-code into the model input an asymmetry signal (diff) obtained by subtracting the opposite-side features from one side's features (no bypass path).
- **Continuous-slice inference →** A physician does not conclude from a single slice, but checks whether a lesion continues across adjacent upper/lower slices. → With a Transformer we view the context of the entire slice sequence together, and in post-processing we remove spans cut off too short.

### 2.3 Backbone and sequence-encoder selection

**Backbone (EfficientNet-B0):** Because the dataset is on the order of a few dozen patients, a backbone with many parameters carries a high overfitting risk. We selected EfficientNet-B0 as the base backbone because it can leverage ImageNet pretrained weights while remaining lightweight and suitable for 224px input. In particular, in OtitisModel we extract the pre-GAP spatial feature map (7×7) as-is and project it with a 1×1 conv (SpatialBackbone), enabling left and right to be compared "in space" rather than as "vectors."

**Sequence encoder (Transformer):** To learn inter-slice positional relationships and long-range context (the continuity of a lesion span), we use positional embeddings + a Transformer Encoder. Compared with an RNN, it is suitable in that it views the entire slice sequence in parallel and can directly model relationships between distant slices.

---

## 3. Data Analysis — Class Imbalance of Rt Otitis Media

Prior to model design, we analyzed the train/val data distribution. Of the four items (Rt/Lt temporal, Rt/Lt otitis), only Rt (right) otitis media is markedly imbalanced.

<img width="2160" height="2528" alt="architecture" src="https://github.com/user-attachments/assets/d1a43079-4d7a-42d9-8433-bd618e9adfff" />
<img width="1624" height="410" alt="distribution_analysis" src="https://github.com/user-attachments/assets/5dc647d8-3606-453f-96d5-c739d187bc45" />


*Figure 1. Train/Val distribution analysis of the CT slice dataset.*

**The grounds for the Rt otitis imbalance observed in Figure 1 are as follows.**

- **(Graph (2)) Slice-level positive rate:** Among valid slices, the positive ratio for Rt otitis is train ≈ 28%, val ≈ 14%—the lowest among the four items (the remaining Rt/Lt temporal and Lt otitis are all about 70% or higher). Not only are positive slices absolutely scarce, but the gap between train (28%) and val (14%) is large, so overfitting to the train distribution easily collapses on val.
- **(Graph (2)) Left/right asymmetry:** Even for the same otitis media, Lt (about 70%+) is common while Rt (14–28%) is scarce. This strong left/right asymmetry becomes the direct basis for the left/right comparison (diff) design that "judges one side relative to the opposite ear."
- **(Graph (3)) Span concentration of lesions:** Looking at the positive ratio per slice position, the lesions of all items concentrate in the central slice span (roughly slices 40–60) and are nearly 0 at the extreme ends. That is, lesions exist only in part of the span where the temporal structures are visible, so a staged gate design that first finds that span and judges only within it is justified.
- **(Graph (3)) Consistently weak signal of Rt:** The Rt otitis curve stays at most around 45% even within the same central span—consistently low at every slice position—unlike the temporal area and left otitis (nearly 100%). This means that beyond simply having few positive patients, the positive signal is weak even at the slice level, which is why imbalance countermeasures such as loss weighting, focal loss, and candidate masking are needed.

In summary, Rt otitis media has (a) a very low slice positive rate [(2)], (b) a large train/val distribution gap [(2)], (c) a weak signal across the entire lesion axis [(3)], and (d) a strong asymmetry with the left side [(2)]. Therefore, if ordinary loss/training settings are used as-is, the model falls into the trap of achieving high accuracy even when predicting "all negative." All imbalance countermeasures in the next section start from this analysis.

---

## 4. Methods to Address the Rt Otitis Imbalance

To address the Rt otitis imbalance, we applied techniques in a multi-layered manner across the entire pipeline of "data split → loss function → restricting the training range → model architecture → evaluation/threshold."

### 4.1 Balanced fold construction (Stratified K-Fold)

In 5-fold cross-validation, if patients are split by simple random division, the number of Rt-otitis-positive patients varies greatly per fold (e.g., one fold has 2, another has 5). As a result, the per-fold Rt F1 variance becomes large (a wobble of roughly 0.33–0.50), making it hard to judge "whether this best is skill or luck."

To solve this, at the otitis stage we constructed stratified folds based on whether each patient holds Rt-otitis positives. After separately shuffling the positive-patient set and the negative-patient set, we distributed them to folds in round-robin fashion so that every fold has a similar Rt-positive ratio (reproducible with a fixed SEED).

**Expected effect:** Since the absolute number of Rt-positive patients does not change, the goal is not a "performance jump" but "stabilization"—reducing inter-fold variance to raise the reliability of model selection and threshold derivation. When folds are balanced, ensembling and threshold derivation are not swayed by the luck of a particular fold.

### 4.2 Loss-function-level countermeasures

- **pos_weight:** To increase the loss contribution of scarce positive slices, we apply a per-class pos_weight. For otitis we use the square root of the negative/positive ratio with an upper bound (cap=8), so the weight does not explode under extreme imbalance.
- **Focal BCE Loss:** To reduce the loss of the many easily-classified negative slices and focus on the few hard positives, we use Focal Loss (γ=2). It suits this project's characteristics of imbalance + hard positives.
- **Per-side Loss (left/right separation):** When summing left/right losses, we assign weight 1.0 to the harder Rt and 0.8 to the comparatively easier Lt. This prevents Rt learning from being buried by the loss being dominated by the easy Lt.

### 4.3 Restricting the range of training slices (Candidate Masking)

Otitis is meaningful only in slices where the temporal area is visible. Therefore, during otitis training, we primarily use only the slices within a candidate region obtained by expanding the temporal-area positive span by ±5 slices, and randomly include only 30% of the clearly-negative slices outside it as hard negatives. This prevents the overwhelmingly numerous "out-of-interest negative slices" from eroding training, while retaining the minimum negatives needed to learn boundary false positives.

### 4.4 Model-architecture-level countermeasure — left/right comparison

Rt otitis is scarce, but there is a strong asymmetry in that the same patient's Lt is almost always positive. To actively exploit this structure, OtitisModel compares left and right directly in space.

- **Spatial left/right comparison (_SpatialSideEncoder):** It feeds the self feature map and the (self − other) difference map together into a conv, producing per-side features by viewing absolute evidence and left/right relative evidence simultaneously.
- **Antisymmetric diff branch (_DiffBranch):** It places an antisymmetric branch of the form d = f(R,L) − f(L,R) so that structurally d(R,L) = −d(L,R) is guaranteed, bounded to (−1, 1) with tanh. The final logits are combined as Rt = logit + α·d, Lt = logit − α·d, where α is learned but fixed with a lower bound of 0.2 (so diff does not turn off).
- **Auxiliary loss (aux):** The diff branch is directly supervised on the left/right label difference (Rt − Lt) with smooth-L1 (auxiliary loss λ=0.5), so the asymmetry signal is learned to reflect the actual diagnostic difference.

### 4.5 Evaluation/threshold-level countermeasures

- **best-selection criterion:** We take the model-selection criterion to be the left/right positive F1 rather than macro F1, preventing best from being updated incorrectly when easy Lt performance masks the hard Rt failure.
- **Independent left/right thresholds:** We determine Rt and Lt thresholds by sweeping them independently, and rather than choosing by looking at val, we derive them via 5-fold ensemble (logit-average) re-prediction on the training data, avoiding threshold overfitting.

### 4.6 Efforts for generalization (overfitting prevention)

The abnormal (positive) data of Rt otitis are extremely small in absolute amount for both patients and slices (slice positive rate train 28% · val 14%). When positive samples are this few, the model easily overfits by "memorizing" the few positive cases, so only training performance rises while it collapses on validation/real data. In particular, the large gap between train (28%) and val (14%) positive rates directly shows the risk that a model memorized to the train distribution fails to generalize on val. Therefore, aiming to "learn the general features of positives" rather than to "memorize positives," we put in place the following devices.

- **Strong data augmentation (memorization prevention):** Rotation ±13°, translation ±7%, brightness/contrast 0.75–1.25, and Gaussian blur are applied identically to left/right/view. By making the same positive slice look slightly different each time, pixel-level memorization is prevented.
- **Model-capacity / regularization control:** We use a lightweight backbone suited to small data (EfficientNet-B0) and suppress model capacity and weight magnitude with dropout (e.g., 0.4 in the otitis sequence encoder) and weight decay (1e-3). The fewer the parameters, the harder it is to memorize few samples.
- **Weight sharing (Siamese):** By sharing the left/right encoder weights (Siamese), we halve the parameters and impose a structural constraint that left and right are judged by the same rule. This acts as a form of structural regularization.
- **freeze→unfreeze staged training:** For the first 8 epochs we freeze the pretrained backbone to stabilize only the head, then unfreeze. Shaking the backbone from scratch with little data damages the ImageNet prior knowledge and worsens generalization.
- **5-fold ensemble:** We ensemble the 5 fold models by logit averaging. Since averaging cancels the variance of a single model, even if one fold overfits to a particular positive case, the final prediction becomes more stable and general.
- **Early stopping (patience 12):** If the validation performance (left/right positive F1) does not improve for a number of epochs, training stops. This blocks the regime of entering overfitting while only the training loss keeps decreasing.
- **Validation-non-referencing threshold:** Since the threshold is derived from the training-data ensemble rather than from val, even fine overfitting in the form of post-hoc tuning by looking at the validation score is prevented at the source.

---

## 5. Data Preprocessing

Preprocessing was implemented in a single module so that training and inference share identical code, fundamentally blocking performance degradation due to preprocessing mismatch. The key steps are as follows.

- **HU restoration:** Applying the DICOM RescaleSlope/Intercept, pixel values are restored to HU (Hounsfield Units). Unlike a plain PNG conversion, this yields physically meaningful density values.
- **Multi-window 3 channels:** The same HU image is converted into three different windows, normalized to 0–255, and stacked into 3 channels. ch0 = bone (WL 700 / WW 2000), ch1 = soft tissue (50/350), ch2 = air/wide (−300/1400). This packs the process of a physician viewing through multiple windows into a single input.
- **Left/right split and alignment:** The axial image is split into left/right halves, and the left (Lt) half is horizontally flipped (hflip) to align its orientation with the right. Left and right then face the same direction, making left/right comparison possible.
- **resize / normalization / augmentation:** Each half is resized to 224×224 and normalized with ImageNet statistics (mean/std). During training, augmentation applied identically to left/right/view (rotation ±13°, translation ±7%, brightness/contrast 0.75–1.25, Gaussian blur) is added to suppress overfitting (memorization).

---

## 6. Model Architecture and Operating Principles

<img width="2160" height="2528" alt="architecture" src="https://github.com/user-attachments/assets/e70c4e9c-ed02-49a0-a281-f653e2482551" />


*Figure 2. Overall two-stage architecture: shared preprocessing → TemporalModel (temporal-span gate) → OtitisModel (left/right spatial comparison) → gated fusion → final per-slice prediction.*

The final system consists of two staged models. Both receive left/right-split inputs and output, per slice sequence, the probability for each of left and right.

### 6.1 TemporalModel — temporal-area span detection

It uses the GAP feature vector of EfficientNet-B0. At each slice, the left/right features, their difference (diff), and slice-position meta-information are passed through a trunk; then positional embeddings are added and a Transformer views the sequence context. The output is the per-slice [Rt_temporal, Lt_temporal] probability. This output goes through post-processing (median smoothing → filling small gaps → removing short spans) to be confirmed as the temporal span, then is expanded by ±5 slices (dilation) and passed as the gate for the otitis stage.

### 6.2 OtitisModel — left/right spatial-comparison-based otitis classification

Its characteristic is that the judgment itself is designed to occur on top of a left/right spatial comparison (not post-hoc correction).

- **Spatial backbone:** With SpatialBackbone (the pre-GAP spatial map of EfficientNet-B0 → 1×1 conv projection, keeping 7×7), the left/right spatial maps are obtained. Since GAP is not applied, the positional information of local lesions is preserved. After backbone unfreeze, gradient checkpointing is used to control VRAM.
- **main path:** _SpatialSideEncoder produces per-side features from self and the (self−other) difference map, and a shared _SeqTransformer views the sequence context (left/right weight sharing = Siamese).
- **diff path:** _DiffBranch produces the antisymmetric scalar d, combined as Rt = logit + α·d, Lt = logit − α·d (α learned, lower bound 0.2).

The output [Rt_otitis, Lt_otitis] probabilities, after passing the threshold, are AND-combined with TemporalModel's gate, so false positives outside the temporal span are removed. After subsequent gap-filling and short-span-removal post-processing, this becomes the final prediction.

### 6.3 Inference pipeline summary

At inference, the 5-fold models are ensembled by logit averaging and then sigmoid is applied (not sigmoid averaging). MAD-based noise slices are excluded in advance, and after prediction they are filled by copying the values of adjacent normal slices. The threshold uses the value derived during training as-is (auto), securing the legitimacy of the submission.

---

## 7. Training Method and Main Parameters

Training proceeds in a staged manner in the order temporal → otitis, and each stage is performed with 5-fold cross-validation. The main hyperparameters are as follows.

| Item | Setting |
|---|---|
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Cross-validation | 5-Fold (otitis stratified by Rt-positive) |
| Batch / input | 2 patients/batch, 224×224, 3 channels (multi-window) |
| Epochs / Early stopping | up to 50 epochs, patience 12 |
| Backbone freeze | freeze for the first 8 epochs, then unfreeze |
| Optimizer | AdamW (weight decay 1e-3) |
| Learning rate | head 2.5e-5 / backbone 1e-5 |
| LR schedule | warmup 3 epochs + cosine annealing |
| Loss function | Focal BCE (γ=2) + pos_weight (√ratio, cap 8) |
| Left/right loss weighting | Rt 1.0 / Lt 0.8 (per-side loss) |
| Auxiliary loss | diff branch smooth-L1, λ=0.5 |
| Augmentation | rotation ±13°, translation ±7%, brightness/contrast 0.75–1.25, blur |
| Imbalance handling | stratified fold + candidate mask (±5) + hard-neg 30% |
| Ensemble / threshold | 5-fold logit average; per-side independent derivation via training-data sweep |
| Others | AMP (mixed precision), gradient checkpointing, SEED 42 |

*Table 2. Main training settings and hyperparameters.*

Model selection saves as best only when the left/right positive F1 measured on the validation fold improves; after training ends, the 5 folds are ensembled and the threshold is swept over the entire training data and stored in the checkpoint. The threshold derived this way does not look into the validation data, so there is no concern of threshold overfitting at submission.

---

## 8. Validation Results and Analysis

### 8.1 Official performance metrics

On the official scoring based on exact row matching (No + R/L + Image number), over all 3,320 slices the official result is macro F1 = 0.8656 (FINAL SCORE 86.6 / 100), accuracy 87.14%. This is an improvement of about 0.13 points in macro F1 over the simple baseline (iloc0: accuracy 76.17%, macro F1 0.7333). The macro F1 used by the official scoring is defined by viewing all slices as the two classes Normal(0) / Otitis(1) and averaging the two class F1s.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal(0) | 0.85 | 0.82 | 0.84 | 1,339 |
| Otitis(1) | 0.88 | 0.90 | 0.89 | 1,981 |
| macro avg | 0.87 | 0.86 | **0.8656** | 3,320 |
| accuracy | — | — | **0.8714** | 3,320 |

*Table 3. Official scoring result at the class (Normal/Otitis) level.*

<img width="2160" height="1196" alt="results_scorecard" src="https://github.com/user-attachments/assets/b0967160-06f0-4e4e-8f81-1f13cc681775" />


*Figure 3. Official scoring results: headline metrics (macro F1, final score, accuracy) and per-item macro F1.*

### 8.2 Per-item results

Separating the same prediction into the four items (Rt·Lt × temporal·otitis), the per-item macro F1 (the macro F1 obtained by viewing that item as Normal/Otitis 2-class) is summarized as follows.

| Item | Item macro F1 | Accuracy | Threshold | Slices |
|---|---|---|---|---|
| Rt temporal | 0.9160 | 93.37% | 0.32 | 830 |
| Lt temporal | 0.8798 | 91.33% | 0.32 | 830 |
| **Rt otitis** | **0.6582** | **78.67%** | 0.62 | 830 |
| Lt otitis | 0.8128 | 85.18% | 0.40 | 830 |

*Table 4. Per-item macro F1 (Threshold is the value derived via a 5-fold ensemble sweep on the training data and stored in the checkpoint).*

Among the four items, the two temporal items (0.92 / 0.88) and Lt otitis (0.81) are all stable, but only Rt otitis is markedly low at 0.6582. The Rt otitis confusion matrix is TP 72 / FP 131 / FN 46 / TN 581; with FP (131) exceeding TP (72), the collapse of precision (0.355) is the direct cause of the score drop (recall 0.610 is comparatively decent). In other words, the failure on Rt is a problem of "calling normal cases otitis (FP)" rather than "missing cases (FN)."

That the Rt otitis threshold (0.62) was set the highest among the four items results from a conservative setting—to suppress these false positives—being automatically reflected in the threshold stage; but as discussed below, the threshold alone has clear limits.

### 8.3 Result analysis — pinpointing the cause of Rt otitis failure

Decomposing the cause of the Rt otitis failure with additional diagnostics, it turned out that the cause is not a simple "lack of positive samples" but a mismatch between the design's central assumption and the data structure. We present the grounds step by step.

**(1) It is not a train → val generalization collapse.** First we tested the hypothesis that "training went well but it collapsed on validation." Removing threshold leakage (the optimistic bias of choosing the threshold together on the scoring fold) and model-selection leakage, and honestly re-evaluating the 5 folds (applying a fixed threshold), the Rt otitis macro F1 is about 0.58—essentially matching the validation set's 0.658 (validation is even slightly higher). That is, the generalization gap is essentially absent, and Rt's low performance means the model's actual discriminative power is at that level to begin with, not an accident of the validation stage. Re-confirming this with a threshold-independent metric (AUROC) gives the following.

| Item | train-OOF AUROC | val AUROC |
|---|---|---|
| **Rt otitis** | **0.622** | **0.576** |
| Lt otitis | 0.741 | 0.867 |

*Table 5. AUROC of left/right otitis (training held-out vs validation).*

Rt AUROC is 0.622 even on the training held-out—almost at the random (0.5) level—with no large difference from validation (0.576). This means not a discriminative-power drop due to a distribution shift, but that the discriminative power itself is intrinsically insufficient (whereas Lt generalizes normally, 0.74→0.87).

**(2) Root cause — the structural mismatch of "bilaterality ↔ a left/right-asymmetry detector."** In Section 3 we took Rt otitis's "left/right asymmetry" as the direct basis for the left/right comparison (diff) design. However, further analysis showed that this asymmetry was merely a population-prevalence-level asymmetry (Lt being more common), which differs from the within-patient laterality (one side diseased, the other normal) that the left/right comparison model actually exploits.

| Bilaterality pattern (otitis) | train | val |
|---|---|---|
| Bilateral-positive patients | 37 | 3 |
| Rt-only positive | 1 | 0 |
| Lt-only positive | 31 | 6 |
| P(Rt+ \| Lt+) | 0.54 | 0.33 |

*Table 6. Per-patient bilaterality pattern.*

The key is that of the 38 Rt-positive patients, 37 (97%) are bilateral, and only 1 patient is Rt-only positive. OtitisModel is essentially a left/right-asymmetry detector that views the self feature map and the (self − other) difference map together and uses the antisymmetric branch d = f(R,L) − f(L,R) as evidence of the left/right difference. However:

- **Vanishing of the asymmetry signal:** Since almost all cases where Rt is positive (97%) are bilateral, in those slices left and right are both opacified → self ≈ other → both the asymmetry signal and d become ≈ 0. The model's primary discriminative signal vanishes precisely in the very cases where Rt must be hit.
- **Indistinguishability of symmetric–symmetric:** Bilateral (both abnormal) and normal (both normal) are both left/right-symmetric, so asymmetry alone cannot distinguish the two. The only signal separating them is absolute appearance (whether that ear is opacified), but the architecture weights relative (left/right difference) evidence, and the absolute signal is diluted by being combined with the diff path.

By contrast, for Lt there are 31 Lt-only-positive patients, so within-patient laterality is abundant and the same design works as intended (Lt AUROC 0.74→0.87). That is, the left/right comparison design is suitable for problems where asymmetry exists (Lt·temporal) but is structurally unsuitable for Rt otitis, where bilaterality is dominant. Moreover, the antisymmetric branch may not only have failed to help Rt but possibly worked slightly to its disadvantage. The auxiliary-loss target d_target = (Rt label − Lt label) is almost always ≤ 0 by data composition (Lt-only → −1, bilateral → 0), so d is learned in the negative (−) direction, and in the combination Rt = logit + α·d, α·d acts to somewhat suppress the Rt score (with only 1 Rt-only case, there is also no data to learn a positive signal).

**(3) The distribution gap (prevalence shift) is a secondary factor.** The positive-rate gap between train (slice positive rate ≈ 28%) and val (≈ 14%)—a 0.51× ratio—also contributes to the Rt precision drop. However, this is not a discriminative-power drop but a mechanical (Bayes) effect on precision. Assuming the same operating point (TPR 0.340, FPR 0.241) and projecting precision while changing only the prevalence, precision is 0.432 at prevalence 0.350 (train) and 0.226 at prevalence 0.171 (val), shaving off about 0.21. A substantial part of the validation-set Rt precision (0.355) is explained by this effect, and the difference in bilaterality composition (P(Rt+|Lt+) 0.54→0.33) acts in the same direction. Still, since the AUROC ≈ 0.6 of (1) is independent of prevalence, the distribution gap is a secondary factor that aggravated the precision drop, and the primary cause is the structural mismatch of (2).

**(4) Why the imbalance countermeasures missed the target.** The pos_weight, focal loss, candidate masking, per-side loss, and stratified fold applied in Section 4 are all techniques aimed at prevalence imbalance (excess negatives). However, the actual bottleneck revealed in (1)–(3) was not prevalence but the absence of a discriminative feature (the vanishing of the asymmetry signal under bilaterality). Since a non-existent feature cannot be created no matter how much the loss weighting is tuned, it is a natural consequence that Rt did not improve despite the multi-layered imbalance handling. This is also confirmed in the before/after comparison of the temporal gate. While Lt otitis improved with the gate (FP dropped 56→37, F1 improved 0.885→0.898), for Rt otitis FP, FN, and F1 were all unchanged. That is, Rt's false positives occur not outside but inside the temporal span (mistaking a normal Rt middle ear for otitis), which exactly matches the diagnosis of (2) that the model cannot read absolute appearance.

### 8.4 Synthesis, limitations, and improvement directions

Overall, the staged gate structure and the left/right comparison design worked as intended on items where asymmetry exists (temporal 0.88–0.92, Lt otitis 0.81), achieving an official macro F1 of 0.8656. The Rt otitis imbalance, however, was not broken through head-on; through diagnosis we pinpointed that the cause lies in the mismatch between the design's central assumption and the data structure rather than in the lack of positive samples itself. In summary, Rt otitis is about 97% bilateral, so the left/right asymmetry signal vanishes in the target class, and a model designed to rely on asymmetry could not have structural discriminative power for that class (AUROC ≈ 0.6). The Section 4 countermeasures aimed at prevalence imbalance could not directly target this failure mode. The improvement directions based on this diagnosis are as follows.

- **Strengthening the absolute-appearance signal:** Reduce dependence on left/right difference (relative evidence) and learn each ear's absolute opacification directly. Concretely, remove the α lower bound of the diff combination (or add a self-only branch), and expand the proportion of absolute features that remain alive even in bilateral cases.
- **Introducing a middle-ear ROI:** Although the lesion is local (middle ear/mastoid), feeding the entire half dilutes the signal. Raise the absolute-signal density with a middle-ear ROI crop (synergy with absolute-appearance learning).
- **Revising the model-selection criterion:** Select the best epoch by Rt alone (or per-item macro) rather than by the left/right positive F1 average, so easy Lt does not mask Rt failure.
- **Data side:** Secure additional Rt-positive, especially Rt-only (unilateral), cases. For the left/right comparison design to be valid, unilateral cases are needed.

---

<sub>Ji Seonwoo · Student ID 2021741045 · Temporal Bone CT Otitis Media Prediction</sub>
