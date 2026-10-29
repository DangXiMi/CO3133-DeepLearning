# Assignment 2 — Deep Learning on Large-Scale Data for Specialized Computer Vision or Language Tasks

**Topic:** Food-101 image classification: simple baseline vs pretrained ResNet-50

## Dataset

- **Food-101** (Bossard et al., 2014), 101 classes, 101,000 images.
- Official split: 75,750 train / 25,250 test.
- A stratified 10% validation split is carved from the training set with a fixed split seed of 42.
- Data is downloaded from the HuggingFace mirror `ethz/food101` into `data/food-101/`.

## Models

| Role | Model | Input size | Params |
|---|---|---:|---:|
| Simple baseline | Assignment 1 CNN, retrained from scratch | 128×128 | 8,829,349 |
| Pretrained/modern model | ImageNet ResNet-50, full fine-tuning | 224×224 | 23,714,981 |
| Controlled ablation | ImageNet ResNet-50, frozen backbone | 224×224 | 23,714,981 (206,949 trainable) |

`resnet50_frozen` is the same ResNet-50 as `resnet50_full`, with the backbone frozen. Only the classification head is trained.

## Experimental design

- **Seeds:** 42, 67, 69 for training.
- **Split seed:** fixed at 42 so every model and seed sees the same train/validation/test split.
- **Model selection:** best validation macro-F1; the official test set is never used for model selection.
- **Epochs:** 30.
- **Batch size:** 128.
- **Controlled experiment:** frozen backbone vs full fine-tuning.
  - Hypothesis: full fine-tuning beats a frozen backbone on Food-101.
  - Changed factor: backbone frozen vs fully trainable.
  - Fixed factors: data split, augmentation, input size, epochs, optimizer, batch size, seeds.
  - Metrics: test accuracy, macro-F1, training time, inference time.
- **Curves, examples, and timing** reuse `utils/experiment_utils.py` from Assignment 1.

## Results

Mean ± standard deviation over seeds 42, 67, and 69.

| Model | Test accuracy | Test macro-F1 | Train time | Inference |
|---|---:|---:|---:|---:|
| Assignment 1 CNN | 40.18% ± 2.38 | 39.74% ± 2.48 | 1342 s | 0.416 ms |
| ResNet-50 frozen | 69.29% ± 0.06 | 68.96% ± 0.08 | 1252 s | 0.730 ms |
| ResNet-50 full | **84.20% ± 0.10** | **84.19% ± 0.15** | 1698 s | 0.623 ms |

Full fine-tuning improves test accuracy by **14.91 percentage points** and macro-F1 by **15.23 points** over the frozen backbone.

## How to run

```bash
pip install -r assignment2/requirements.txt
jupyter notebook assignment2/notebooks/main.ipynb
```

The notebook:

1. downloads Food-101 and builds the train/val/test loaders,
2. trains the simple baseline, the fine-tuned ResNet-50, and the frozen ResNet-50,
3. repeats each run with seeds 42, 67, and 69,
4. selects the best checkpoint per run by validation macro-F1,
5. evaluates every run on the official test set,
6. computes accuracy, macro-F1, per-class precision/recall/F1, top confusions, training time, and inference time,
7. saves curves, example predictions, confusion matrices, per-run histories, and the aggregated results.

Useful settings at the top of the notebook:

```python
SEEDS = [42, 67, 69]
SPLIT_SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
WORKERS = 8
LIMIT = None
MODELS = ["baseline", "resnet50_full", "resnet50_frozen"]
```

## Outputs

```text
assignment2/results/results.csv
assignment2/results/summary.csv
assignment2/results/results.json
assignment2/results/history/<model>_seed<seed>.json
assignment2/results/curves/<model>_seed<seed>.png
assignment2/results/examples/<model>_seed<seed>.png
assignment2/results/confusion/<model>_seed<seed>.npy
assignment2/results/confusion/<model>_seed<seed>_top_confusions.json
assignment2/results/confusion/<model>_seed<seed>_top_confusions.png
assignment2/checkpoints/<model>_seed<seed>.pt
```

## Repository layout

```text
assignment2/
  notebooks/
    main.ipynb      multi-seed training, evaluation, and ablation notebook
    eda.ipynb       dataset EDA
    check_duplicate.ipynb
  src/
    training_loop.py  shared train/validation loop
    models/
      baseline.py   simple baseline (reuses Assignment 1 CNN)
      resnet.py     pretrained ResNet-50 builder
  checkpoints/      best checkpoints per model and seed
  results/          metrics, histories, curves, examples, confusions
utils/
  dataloader.py     Food-101 download, split, transforms, loaders
  experiment_utils.py  reused curves, examples, and timing helpers
```

## Links

- [Assignment 2 page](index.html)
- [Dataset proposal](PROPOSAL.md)
- [Results JSON](results/results.json)
- [AI usage disclosure](AI_USAGE.md)
