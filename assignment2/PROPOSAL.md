# Assignment 2 — Dataset Proposal

**Course:** Deep Learning and Its Applications (CO3133) — Semester-261
**Group:** G-03 — Nguyen Van An (2352013), Vo Le Hai Dang (2352257), Huynh Vuong Khang (2350011)
**Task track:** Image classification
**Dataset:** Food-101
**Date:** 06 Oct 2026

---

**1. Dataset name, source, version, and license**

Food-101 (Bossard, Guillaumin & Van Gool, ETH Zürich, ECCV 2014). 101 food categories,
101,000 images collected from Foodspotting.com. Source: official ETH Zürich page
(https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) and the HuggingFace
mirror (https://huggingface.co/datasets/ethz/food101), which we use for download.
Version: original 2014 release (static). License (verbatim): *"The Food-101 data set
consists of images from Foodspotting which are not property of ETHZ. Any use beyond
scientific fair use must be negotiated with the respective picture owners."*
Our use is a university course project — scientific fair use.

**2. Task, input, and output definitions**

Fine-grained multi-class image classification. Input: a single RGB photograph
(max side 512 px, resized per protocol). Output: one of 101 dish classes
(e.g., apple_pie … pho), produced as class logits. Prediction unit: one image.

**3. Number of samples and annotation types**

101,000 images; one crowd-sourced dish label per image. Official split:
75,750 train (750 per class, labels auto-assigned, intentionally uncleaned by the
authors) and 25,250 test (250 per class, human-verified labels).

**4. Preliminary distribution analysis**

Perfectly class-balanced by construction (750 train / 250 test per class, verified
from the official dataset card). Known issues to be analyzed in full EDA: documented
label noise in the training set, variable aspect ratios and image sizes, and possible
near-duplicates from the Foodspotting source.

**5. Data-splitting plan**

Use the official train/test split unchanged as the final evaluation set. Carve a
stratified validation set from the 75,750 training images: 10% (7,575 images), seed 42,
stratified per class — the same protocol as our Assignment 1 pipeline
(sklearn train_test_split, documented seed).

**6. Split unit used to prevent leakage**

One photograph. The official test split was human-verified and released separately;
we never re-split it. Leakage checks before training: hash-based duplicate and
near-duplicate detection between train and validation (perceptual or checksum
hashing), verifying the split-unit promise empirically.

**7. Evaluation metric(s)**

Top-1 accuracy and macro-F1 (per handbook §20.1 for classification). Secondary:
per-class accuracy to expose fine-grained confusions, plus training and inference
time for the compute-cost analysis.

**8. Baseline plan**

A self-implemented small CNN (ported from our Assignment 1 convolutional baseline:
conv/pool blocks + MLP head), trained from scratch at 128×128 on our local GPU.
Purpose: a from-scratch reference point against the fine-tuned pretrained model.

**9. Planned pretrained model(s)**

ResNet-18 with ImageNet weights (torchvision), fine-tuned with a documented
protocol: standard resize and flip augmentation, mixed precision (AMP), AdamW,
identity-fixed seed and split shared with the baseline. Mandatory controlled
experiment: frozen backbone (linear probe) vs. full fine-tune (handbook §20.4).
Optional second backbone (ResNet-50) for a backbone A-vs-B contrast if compute allows.

**10. Compute estimate**

Main experiments will run on the group's dedicated remote server (to be connected
at the start of the compute phase; this proposal will be updated with exact hardware
when available). Interim and fallback compute: local NVIDIA RTX 3050 Laptop 4GB /
Ryzen 5 5625U / 14GB RAM — sufficient for the from-scratch baseline (~1–2h) and for
ResNet-18 fine-tuning with AMP at reduced settings — plus free Colab/Kaggle T4 GPUs
for larger-resolution runs. Storage: ~10GB dataset, kept out of the repository.

**11. Subset selection rules, if any**

None planned — the full 75,750 training images meet the §19.1 thresholds
(≥5 classes, ≥5,000 training samples). Contingency: if compute proves insufficient
for 224px fine-tuning, cap per-class training images by stratified sampling
(fixed seed), never below 5,000 training samples total; any such subset would be
documented with its seed and selection rule.
