# Assignment 3 — Dataset Proposal

**Course:** Deep Learning and Its Applications (CO3133) — Semester-261
**Group:** G-03 — Nguyen Van An (2352013), Vo Le Hai Dang (2352257), Huynh Vuong Khang (2350011)
**Task track:** Image-text classification
**Dataset:** MM-IMDb
**Date:** 18 Nov 2026

---

**1. Dataset name, source, version, and license**

**MM-IMDb** (Arevalo, Solorio, Montes-y-Gómez & González, *Gated Multimodal Units for Information Fusion*, ICLR 2017 Workshop). A multimodal dataset containing **25,959 movies**, each associated with a movie poster (image), plot summary (text), and one or more of **23 genre labels**. The dataset was constructed using movie IDs from MovieLens 20M, with movie information collected from IMDb. Source: original research paper (https://arxiv.org/abs/1702.01992) and official implementation repository (https://github.com/johnarevalo/gmu-mmimdb). We use the Hugging Face mirror (https://huggingface.co/datasets/pranavmr/MM-IMDb) for downloading the paired image–text data.

**Version**: Original 2017 dataset.

**License**:
- Official repository: MIT-licensed.
- Hugging Face mirror: Apache License 2.0 (`apache-2.0`).
Our intended use is a non-commercial university course project for educational and research purposes.

**2. Task, modalities, input, and output definitions**

Multi-label image-text classification. Input: a paired movie poster (RGB image, resized to 224×224) and its corresponding plot summary (tokenized and truncated/padded to 256 tokens). Output: 23 independent sigmoid probabilities over movie genres, trained with binary cross-entropy on logits. Prediction unit: one poster–plot pair.

**3. Number of paired samples and annotation types**

25,959 image-text pairs; 23 film genre labels; one or many labels per pair. Official split: stratified training, dev (validation) and test split of 60% - 10% - 30% (15,552 - 2,608 - 7,799). The official split mentioned in the original research paper (Arevalo et al., 2017, p. 7) was not implemented in the Hugging Face mirror.

Although MovieLens 20M is a curated benchmark dataset, its official documentation does not guarantee complete data correctness. The MM-IMDb authors do not report systematic manual verification or cleaning of the retrieved IMDb genre annotations, or the semantic alignment for each poster-plot pair. Consequently, the genre labels are treated as source-provided annotations with potential errors and inconsistencies.

**4. Preliminary pair-quality and distribution analysis**

- Genre distribution: an average of 2.48 genres per movie. Substantial class imbalance: Drama is the most frequent genre (13,967 movies), Film-Noir the least (338) (Arevalo et al., 2017, p. 8).
- Text and image modalities: 92.5 words on average, left-skewed text-length distribution, varying image sizes (Arevalo et al., 2017, p. 7).
- Pair validity definition: a valid pair is a poster and a plot summary joined on the **same IMDb movie ID**. Records with missing or corrupt images, empty or duplicate text, or unmatched IDs are rejected.
- Verification plan: report the missing-modality rate, hash images and normalized text to detect exact and near duplicates, and manually spot-check 100 random pairs. Remaining issues, including invalid labels and mismatched pairs, are analysed in the full EDA.

**5. Data-splitting plan**

Implement the official stratified train/dev/test split from the original research paper with `scikit-multilearn`'s `iterative_train_test_split` (https://github.com/scikit-multilearn/scikit-multilearn/blob/master/skmultilearn/model_selection/iterative_stratification.py#L175). We use a two-stage procedure: a 70:30 split to obtain the test set, then a 60:10 split on the remaining 70% to obtain training and dev/validation sets. By default, the method preserves the frequencies of each single label and of the co-occurrences of each genre pair. Issues with the reproducibility of this method have been reported (https://github.com/scikit-multilearn/scikit-multilearn/issues/144), so splits are verified over multiple runs and the resulting indices are frozen and stored with the repository, combined with further inspection to avoid data leakage.

**6. Split unit used to prevent leakage**

The leakage-prevention unit is the movie identity (entity). Records associated with the same movie, including alternative posters or plot descriptions, are assigned to the same subset. Leakage checks before each splitting stage:
- Image modality: hash-based duplicate and near-duplicate detection between splits (perceptual or checksum hashing), verifying the split-unit promise empirically.
- Text modality: normalized text hashing, TF-IDF, or MinHash.

**7. Evaluation metric(s)**

Macro-F1 as the primary metric (per handbook §29.3 for classification), accounting for class imbalance across 23 genres. Secondary: micro-F1, sample-wise F1 to evaluate predicted genre sets per movie, exact-match accuracy to expose fine-grained confusions, and training/inference time for the compute-cost analysis. Per-class sigmoid thresholds are tuned on the dev split before test evaluation.

**8. Baseline plan**

All four models share the same frozen split, per-class threshold tuning on dev, checkpoint selection by validation macro-F1, and `BCEWithLogitsLoss`.

| | Image-only | Text-only | Simple fusion | Improved fusion |
|---|---|---|---|---|
| Encoder | ResNet-18, ImageNet weights, fine-tuned | DistilBERT-base-uncased, fine-tuned | frozen encoders, cached features | frozen encoders + GMU |
| Input | 224×224, ImageNet mean/std, random crop 0.8–1.0, horizontal flip | plot summary, ≤256 tokens | cached 512-d + 768-d vectors | same as simple fusion |
| Representation | 512-d pooled | 768-d `[CLS]` | concatenation, 1,280-d | GMU output, 512-d |
| Head | dropout 0.3 → Linear(512→23) | dropout 0.3 → Linear(768→23) | 1,280→512→ReLU→dropout→23 | 512→23 |
| Training | AdamW 3e-4, 10 epochs, batch 256, AMP | AdamW 2e-5, 5 epochs, batch 64, 10% warmup | Adam 1e-3, 30 epochs, batch 256 | same as simple fusion |
| Role | baseline A | baseline B | M2 deliverable | M3 deliverable |

- **Image-only baseline:** full fine-tune of ResNet-18; no layer freezing. Captures colour and composition cues but no plot semantics.
- **Text-only baseline:** DistilBERT on plot summaries; 256 tokens covers the 92.5-word average. Expected to be the stronger unimodal, since genre is largely described by plot events.
- **Simple fusion baseline:** precompute both feature sets once and train only the head. Two variants are reported: feature concatenation (primary) and probability averaging (secondary).
- **Improved multimodal model:** gated multimodal unit (GMU), the fusion from the MM-IMDb paper:

  `h = tanh(W_i x_i) ⊙ σ(W_g [x_i; x_t]) + tanh(W_t x_t) ⊙ (1 − σ(W_g [x_i; x_t]))`

  with modality dropout during training (drop image or text with p = 0.2), enabling missing-modality evaluation at test time.

Planned comparisons: unimodal vs multimodal, concatenation vs GMU, modality dropout on/off, and missing-modality tests (image dropped, text dropped). These cover the handbook's unimodal/fusion comparison and ablation requirements.

**9. Planned pretrained model(s)**

- **Image encoder:** ResNet-18 with ImageNet weights (11.7M parameters), fine-tuned end-to-end for the image baseline and frozen for the fusion models.
- **Text encoder:** DistilBERT-base-uncased (66M parameters), fine-tuned for the text baseline and frozen for the fusion models, maximum 256 tokens.
- **Optional modern reference:** CLIP ViT-B/32 (frozen), clearly labelled as an extension: zero-shot genre prompts, linear probe, and a poster→plot retrieval figure. It is not part of the required four models.
- **Scale check (A100 only, optional):** ResNet-50 and RoBERTa-base to test whether gains come from fusion or from capacity.

**10. Compute estimate**

Main experiments run on a dedicated **NVIDIA A100 80 GB** server (DGX), accessed over SSH under tmux with one GPU per run; the local NVIDIA RTX 3050 Laptop 4 GB is kept as a fallback. The dataset is ~1.9 GB and stays outside the repository; cached image (512-d) and text (768-d) features total under 100 MB.

| Configuration | Per epoch (A100) | Per seed |
|---|---:|---:|
| ResNet-18 fine-tune, 224, batch 256 | 30–60 s | 5–10 min |
| DistilBERT, 256 tokens, batch 64 | 20–40 s | 2–4 min |
| Simple fusion / GMU (cached features) | <2 s | <1 min |

The M2 draft requires under 30 A100 GPU-minutes with one seed. M3 (four models × 3 seeds, four ablations, missing-modality evaluation) is estimated at 3–5 A100 GPU-hours. Checkpoints and features stay under 2 GB. On the RTX 3050 fallback, the image model takes roughly 1–1.5 h per seed and the text model 15–20 min; fusion runs are still minutes.

**11. Subset selection rules, if any**

None planned: the 15,552 training pairs meet the §28.2 thresholds (≥5 classes, about 5,000 paired samples). Contingency: if cleaning drops the pair count below 5,000, we will request instructor approval before proceeding.

**12. Optional extensions (not required)**

Clearly separated from the core above: three-seed runs with mean ± standard deviation, GMU vs FiLM fusion comparison, low-data learning curve (10/25/50/100% of training pairs), CLIP zero-shot and linear-probe reference, and a support/conflict case study showing movies where the modalities agree and disagree.
