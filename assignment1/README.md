# Assignment 1 — Foundations of Deep Learning Pipelines and Architectures

**Topic:** From Linear Models to Modern Sequence Models: A Comparative Study for Image Classification

## Dataset

- **Primary results:** Fashion-MNIST (70,000 28×28 grayscale images, 10 classes)
- **Development/debugging only:** MNIST
- **Optional extension:** CIFAR-10

The dataset is downloaded automatically by `torchvision.datasets.FashionMNIST` into `data/` on the first run.

## Split and preprocessing

- Official 60,000 train / 10,000 test split.
- The 60,000 training images are split into **48,000 train / 12,000 validation** with `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- Every model uses the same split, seed 42, preprocessing, batch size, optimizer, and epoch budget.
- Images are scaled to [0, 1] and z-scored with the training-set mean 0.2863 and standard deviation 0.3532; the same transform is applied to validation and test.
- No augmentation is used in the main comparison.

## How to run

The entry point is the executed notebook:

```bash
jupyter notebook assignment1/notebooks/main.ipynb
```

Run all cells from top to bottom. The notebook:

1. loads Fashion-MNIST and builds the shared DataLoaders,
2. trains Linear, MLP, CNN, GRU, and Transformer for 10 epochs,
3. selects the best checkpoint per model by validation macro-F1,
4. evaluates each checkpoint on the 10,000-image test set,
5. writes checkpoints, metrics, histories, curves, confusion matrices, and prediction examples.

Training protocol: Adam, learning rate `1e-3`, batch size 64, 10 epochs, cross-entropy loss, gradient clipping at max norm 1.0, seed 42.

## Results summary

Test metrics on the official 10,000-image test set; checkpoints selected by validation macro-F1.

| Model | Params | Test accuracy | Test macro-F1 | Train (s) | Inference (ms/sample) |
|---|---:|---:|---:|---:|---:|
| Linear | 7,850 | 84.10% | 84.15% | 9.6 | 0.005 |
| MLP | 235,146 | 87.89% | 87.89% | 12.8 | 0.006 |
| CNN | 455,946 | 91.79% | 91.79% | 20.5 | 0.010 |
| GRU | 61,962 | 88.93% | 88.85% | 52.5 | 0.025 |
| Transformer | 69,450 | 88.71% | 88.69% | 32.5 | 0.012 |

Results may fluctuate slightly across reruns because of GPU non-determinism and mixed precision (AMP) where enabled.

## Outputs

```text
assignment1/checkpoints/linear.pt
assignment1/checkpoints/mlp.pt
assignment1/checkpoints/cnn.pt
assignment1/checkpoints/gru.pt
assignment1/checkpoints/transformer.pt

assignment1/results/results.csv
assignment1/results/results.json
assignment1/results/config.json
assignment1/results/history/<model>.json
assignment1/results/curves/<model>.png
assignment1/results/confusion/<model>.png
assignment1/results/examples/<model>.png
```

## Run notes

- Seed: 42, set for NumPy and PyTorch.
- Reruns are not bit-identical: GPU non-determinism and mixed precision (AMP) where enabled can cause small metric fluctuations.
- Checkpoint selection: highest validation macro-F1.
- Reported test numbers come from the selected checkpoint, not the final epoch.
- Training time includes per-epoch validation and checkpointing; inference time is a full forward pass over the test set after warmup.

## Repository layout

```text
assignment1/
  notebooks/
    main.ipynb         five-model training and evaluation notebook
    eda.ipynb          dataset EDA
  src/
    training_loop.py   shared model-agnostic training/validation functions
    models/
      linear.py
      mlp.py
      cnn.py
      rnn.py           GRU, LSTM, and plain RNN variants
      transformer.py   row-token Transformer
  checkpoints/         best checkpoints from the reported run
  results/             metrics, histories, curves, confusion matrices, examples
  materials/           report, slides, and video links for the final submission
utils/
  dataloader.py        unified torchvision loading and stratified split
  experiment_utils.py  timing and figure helpers for the A1 notebook
```

## Links

- [Assignment 1 page](index.html)
- [Report](https://github.com/DangXiMi/CO3133-DeepLearning/blob/main/assignment1/materials/report.pdf)
- [Slides](https://github.com/DangXiMi/CO3133-DeepLearning/blob/main/assignment1/materials/slides.pdf)
- [Video](https://www.youtube.com/watch?v=TODO_A1_VIDEO) <!-- TODO: replace with the real YouTube URL -->
- [Results JSON](results/results.json)
- [Results CSV](results/results.csv)
- [Run configuration](results/config.json)
- [AI usage disclosure](AI_USAGE.md)
