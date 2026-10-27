# Assignment 2 — Deep Learning on Large-Scale Data for Specialized Computer Vision or Language Tasks

**Topic:** Food-101 image classification: simple baseline vs pretrained ResNet-18

## Dataset

- **Food-101** (Bossard et al., 2014), 101 classes, 101,000 images.
- Official split: 75,750 train / 25,250 test.
- A stratified 10% validation split is carved from the training set.
- Data is downloaded from the HuggingFace mirror `ethz/food101` into `data/food-101/`.

## Models

| Role | Model | Input size | Params |
|---|---|---:|---:|
| Simple baseline | Assignment 1 CNN, retrained from scratch | 128×128 | 8,829,349 |
| Pretrained/modern model | ImageNet ResNet-18, fine-tuned | 224×224 | 11,228,325 |

`resnet18_frozen` is the same ResNet-18 with the backbone frozen, kept for the mandatory freeze-vs-full ablation.

## How to run

```bash
jupyter notebook assignment2/notebooks/main.ipynb
```

The notebook:

1. downloads Food-101 and builds the train/val/test loaders,
2. trains the simple baseline and the fine-tuned ResNet-18,
3. selects the best checkpoint by validation macro-F1,
4. evaluates both models on the official test set,
5. writes `assignment2/results/results.json` and the checkpoints.

Useful settings at the top of the notebook:

```python
SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
WORKERS = 8
LIMIT = None                     # set 1000 for a smoke run
MODELS = ["baseline", "resnet18"]  # add "resnet18_frozen" for the ablation
```

## Preliminary results (M2 draft, 10 epochs, seed 42)

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Assignment 1 CNN (baseline) | 33.05% | 32.28% |
| ResNet-18 (pretrained, fine-tuned) | 72.25% | 71.95% |

## Outputs

```text
assignment2/results/results.json
assignment2/checkpoints/baseline.pt
assignment2/checkpoints/resnet18.pt
```

## Repository layout

```text
assignment2/
  notebooks/
    main.ipynb      training and evaluation notebook
  src/
    training_loop.py  shared train/validation loop
    models/
    baseline.py     simple baseline (reuses Assignment 1 CNN)
    resnet.py       pretrained ResNet-18 builder
  checkpoints/      best checkpoints from the reported run
  results/          metrics and histories
utils/
  dataloader.py     Food-101 download, split, transforms, loaders
  experiment_utils.py
```

## Links

- [Assignment 2 page](index.html)
- [Dataset proposal](PROPOSAL.md)
- [Results JSON](results/results.json)
- [AI usage disclosure](AI_USAGE.md)
