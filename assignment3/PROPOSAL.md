# Assignment 3 — Dataset Proposal

**Course:** Deep Learning and Its Applications (CO3133) — Semester-261
**Group:** G-03 — Nguyen Van An (2352013), Vo Le Hai Dang (2352257), Huynh Vuong Khang (2350011)
**Task track:** Image-text classification
**Dataset:** MM-IMDb
**Date:** 18 Oct 2026

---

**1. Dataset name, source, version, and license**

**MM-IMDb** (Arevalo, Solorio, Montes-y-Gómez & González, *Gated Multimodal Units for Information Fusion*, ICLR 2017 Workshop). A multimodal dataset containing **25,959 movies**, each associated with a movie poster (image), plot summary (text), and one or more of **23 genre labels**. The dataset was constructed using movie IDs from MovieLens 20M, with movie information collected from IMDb. Source: original research paper (https://arxiv.org/abs/1702.01992) and official implementation repository (https://github.com/johnarevalo/gmu-mmimdb). We use the Hugging Face mirror (https://huggingface.co/datasets/pranavmr/MM-IMDb) for downloading the paired image–text data.

**Version**: Original 2017 dataset.

**License**: 
- Official repository: MIT-licensed.
- Hugging Face mirror: Apache License 2.0 (`apache-2.0`).
Our intended use is a non-commercial university course project for educational and research purposes.

**2. Task, modalities, input, and output definitions**

Multi-label image-text classification. Input: a paired movie poster (RGB image, resized per protocol) and its corresponding textual plot summary (tokenized and truncated/padded to a fixed sequence length). Output: list of movie genre(s), produced as class logits converted to independent probabilities. Prediction unit: one image-text pair.

**3. Number of paired samples and annotation types**

25,959 image-text pairs; 23 film genre labels; one or many labels per pair. Official split: stratified training, dev (validation) and test split of 60% - 10% - 30% (15,552 - 2,608 - 7,799). The official split mentioned in the original research paper (Arevalo et al., 2017, p. 7) was not implemented in the Hugging Face mirror. 

Although the source dataset MovieLens 20M Although MovieLens 20M is a curated benchmark dataset, its official documentation does not guarantee complete data correctness. The MM-IMDb authors do not report systematic manual verification or cleaning of the retrieved IMDb genre annotations, or the semantic alignment for each poster-plot pair. Consequently, the genre labels are treated as source-provided annotations with potential errors and inconsistencies.

**4. Preliminary pair-quality and distribution analysis**

- Genre distribution: An average of 2.48 genres per movie. Substantial class imbalance, Drama is the most frequent genre (13,967 movies), while Film-Noir is the least (338) (Arevalo et al., 2017, p. 8).
- Text and image modalities: 92.5 words on average, left-skewed distribution of text lengths, varying image sizes (Arevalo et al., 2017, p. 7).
- Additional issues: to be analyzed in full EDA, including missing or corrupted images, empty text, duplicate records, invalid labels, and mismatched pairs.

**5. Data-splitting plan**

Implement the official stratified train/dev/test split mentioned the original research paper with `scikit-multilearn`'s `iterative_train_test_split` (https://github.com/scikit-multilearn/scikit-multilearn/blob/master/skmultilearn/model_selection/iterative_stratification.py#L175). We intend to perform a two-stage procedure, the first 70:30 split to obtain the test set, then another 60:10 split on the remaining 70% to obtain the training and dev/validation sets. By default, the method preserves the frequencies of each singular label and of the co-occurences of each genre pair. Issues on the reproducibility of this method had been reported (https://github.com/scikit-multilearn/scikit-multilearn/issues/144), so it is crucial to verify the splits with multiple runs and/or to store the splitted indices, combined with further inspections to avoid data leakage.

**6. Split unit used to prevent leakage**

The leakage-prevention unit is the movie identity (entity). Records associated with the same movie, including alternative posters or plot descriptions, will be assigned to the same subset. Leakage checks before each splitting stage: 
- Image modality: hash-based duplicate and near-duplicate detection between two sets (perceptual or checksum hashing), verifying the split-unit promise empirically.
- Text modality: normalized text hashing, TF-IDF or MinHash.

**7. Evaluation metric(s)**

Macro-F1 as the primary metric (per handbook §29.3 for classification), accounting for class imbalance across 23 genres. Secondary: sample-wise F1 to evaluate predicted genre sets per movie and exact-match accuracy to expose fine-grained confusions, plus training and inference time for the compute-cost analysis.

**8. Baseline plan**

**TODO**
- Image modality baseline:
- Text modality baseline:
- Simple fusion baseline:
- Improved multimodal model / fusion design: 

**9. Planned pretrained model(s)**

**TODO**

**10. Compute estimate**

**TODO**
Main experiments will run on the group's dedicated remote server (to be connected
at the start of the compute phase; this proposal will be updated with exact hardware
when available). Interim and fallback compute: local NVIDIA RTX 3050 Laptop 4GB /
Ryzen 5 5625U / 14GB RAM — sufficient for the from-scratch baseline (~1–2h) and for
ResNet-18 fine-tuning with AMP at reduced settings — plus free Colab/Kaggle T4 GPUs
for larger-resolution runs. Storage: ~1.93GB dataset, kept out of the repository.

**11. Subset selection rules, if any**

**TODO**
None planned — the 15,552 training images meet the §28.2 thresholds
(≥5 classes, ≥5,000 training samples). Contingency: 