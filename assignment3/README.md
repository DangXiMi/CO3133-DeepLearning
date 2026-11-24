# Assignment 3 — Multimodal Deep Learning

**Topic:** MM-IMDb image-text genre classification

## Dataset

- **MM-IMDb**, 25,959 movie poster and plot-summary pairs.
- **23 genre labels**, multi-label classification.
- Official source: Arevalo et al., 2017; Hugging Face mirror `pranavmr/MM-IMDb`.
- Data is downloaded into `data/mm-imdb/`.

## Split

- Iterative multi-label stratification with `scikit-multilearn`.
- 60% train, 10% dev, 30% test.
- Exact duplicate records found in the EDA are removed deterministically with priority **test > dev > train**, or the first copy when both copies lie in the same split.
- The retained and dropped indices are recorded in `results/dedup_dropped_indices.csv`.
- Split indices are frozen in `data/mm-imdb/splits.npz`.

## M2 models

| Role | Model | Input |
|---|---|---|
| Unimodal A | ResNet-18, ImageNet weights, fine-tuned | 224×224 poster |
| Unimodal B | DistilBERT-base-uncased, fine-tuned | plot summary, ≤256 tokens |
| Simple fusion | frozen image and text encoders, cached 512-d + 768-d features, concatenation head | poster + plot |

For M3, Unimodal A is upgraded to **ResNet-50** and Unimodal B to **BERT-base-uncased**. GMU and the joint fine-tune also use this upgraded encoder pair.

Metrics: macro-F1, micro-F1, sample-wise F1, exact match.

## M2 results (deduplicated test split, seed 42)

Final split sizes after the EDA duplicate cleanup: **15,553 train / 2,595 dev / 7,805 test**.

| Model | Macro-F1 | Micro-F1 | Sample F1 | Exact match |
|---|---:|---:|---:|---:|
| Image-only ResNet-18 | 0.316 | 0.431 | 0.427 | 0.023 |
| Text-only DistilBERT | 0.568 | 0.644 | 0.646 | 0.172 |
| Simple fusion | **0.582** | 0.645 | 0.638 | 0.146 |

The simple fusion model slightly exceeds the text-only baseline on macro-F1. It is a small multimodal improvement, not a claim that multimodal always wins.

## M3 results (deduplicated test split, seed 42)

| Model | Macro-F1 | Micro-F1 | Sample F1 | Exact match |
|---|---:|---:|---:|---:|
| Image-only ResNet-50 | 0.331 | 0.439 | 0.442 | 0.037 |
| Text-only BERT | 0.568 | 0.643 | 0.646 | 0.182 |
| Simple fusion | 0.582 | 0.642 | 0.640 | 0.153 |
| GMU, no modality dropout | 0.588 | 0.650 | 0.646 | 0.146 |
| GMU + modality dropout (p = 0.2) | 0.586 | 0.649 | 0.647 | 0.156 |

### Ablation and robustness

- GMU without modality dropout is slightly better than GMU with dropout on macro-F1: **0.588 vs 0.586**.
- Missing-modality evaluation:
  - GMU without dropout: image dropped **0.523**, text dropped **0.285**.
  - GMU with dropout: image dropped **0.540**, text dropped **0.287**.
- Exact-match support/conflict cases for GMU without dropout against text-only:
  - GMU correct and text-only wrong: **570 movies**
  - GMU wrong and text-only correct: **847 movies**
  - Both correct: **573 movies**
  - Both wrong: **5,815 movies**

The fusion result is an improvement over unimodal on macro-F1, but it is not a claim that multimodal helps on every movie.

### Actual joint multimodal fine-tuning

ResNet-50 and BERT are both unfrozen and trained end-to-end with the GMU classifier. Encoder learning rates are 1e-5, GMU learning rate 1e-4, batch 64, 3 epochs, mixed precision.

| Model | Macro-F1 | Micro-F1 | Sample F1 | Exact match |
|---|---:|---:|---:|---:|
| End-to-end multimodal GMU fine-tune | **0.597** | 0.670 | 0.672 | 0.177 |

This is better than the frozen-encoder GMU variant by **+0.008 macro-F1**. Text-only BERT has a comparable exact-match score of **0.182**, so the joint model does not dominate every metric.

Missing-modality behavior for the fine-tuned model:

- Image dropped: macro-F1 **0.581**.
- Text dropped: macro-F1 **0.191**.

The full-model gain is real, but the text modality remains critical when the model was trained jointly.

## How to run

```bash
pip install -r assignment3/requirements.txt
jupyter notebook assignment3/notebooks/main.ipynb        # M2: unimodal baselines + simple fusion
jupyter notebook assignment3/notebooks/m3.ipynb          # M3: GMU, ablation, missing modality, error analysis
jupyter notebook assignment3/notebooks/finetune_gmu.ipynb  # actual joint multimodal fine-tuning
```

The notebooks:

1. download MM-IMDb and build the frozen split,
2. train the image-only ResNet-18/50 and text-only DistilBERT/BERT baselines,
3. extract frozen features and train the simple fusion head,
4. train the GMU variants and run full and missing-modality evaluation,
5. evaluate all models on the test split and save error and support/conflict analyses,
6. write checkpoints and `results/m2_results.{json,csv}` and `results/m3_results.{json,csv}`.

## Outputs

Small checkpoints and result artifacts are committed. Large encoder checkpoints are regenerated from the notebooks rather than pushed because of GitHub's per-file size limit.

```text
assignment3/checkpoints/image_resnet18.pt
assignment3/checkpoints/text_distilbert.pt
assignment3/checkpoints/image_resnet50.pt
assignment3/checkpoints/text_bert.pt
assignment3/checkpoints/fusion_mlp.pt
assignment3/checkpoints/gmu_no_dropout.pt
assignment3/checkpoints/gmu_dropout.pt
assignment3/checkpoints/finetune_gmu.pt
assignment3/results/m2_results.json
assignment3/results/m2_results.csv
assignment3/results/m3_results.json
assignment3/results/m3_results.csv
assignment3/results/m3_error_analysis.json
assignment3/results/m3_predictions.npz
assignment3/results/m3_finetune_gmu.json
assignment3/results/m3_finetune_gmu.csv
```

## Remaining final deliverables

- final report,
- slides,
- YouTube video,
- final page polish and LMS submission.

## Links

- [Assignment 3 page](index.html)
- [Dataset proposal](PROPOSAL.md)
- [AI usage disclosure](AI_USAGE.md)
