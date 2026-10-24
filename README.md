# CO3133 - Deep Learning and Its Applications - Course Project

## Course Information

**Course:** CO3133 - Deep Learning and Its Applications<br>
**Course code:** CO3133<br>
**Semester:** 261<br>
**Instructor:** Le Thanh Sach<br>
**Group:** G-03

---

## Group Members

| Full name | Student ID | Role | GitHub |
|---|---:|---|---|
| Nguyen Van An | 2352013 | Decided later | [BanAnA9205](https://github.com/BanAnA9205) |
| Vo Le Hai Dang | 2352257 | Decided later | [DangXiMi](https://github.com/DangXiMi) |
| Huynh Vuong Khang | 2350011 | Decided later | [hvhvk](https://github.com/hvhvk) |

**Group code repository:**  
[https://github.com/DangXiMi/CO3133-DeepLearning](https://github.com/DangXiMi/CO3133-DeepLearning)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DangXiMi/CO3133-DeepLearning.git
cd CO3133-DeepLearning
```

### 2. Create and activate a virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

For Assignment 1, use the pinned environment that produced the committed results:

```bash
pip install -r assignment1/requirements.txt
```

The repository-root `requirements.txt` is the broader course environment. The Assignment 1 environment used:

- Python 3.14.7
- PyTorch 2.14.0
- torchvision 0.29.0
- CUDA 13.0
- NumPy 2.5.3
- pandas 3.0.6
- Matplotlib 3.11.2
- scikit-learn 1.8.0

Hardware used for the reported runs: **NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB**.

---

## Assignment 1: Foundations of Deep Learning Pipelines and Architectures

**Topic:** From Linear Models to Modern Sequence Models: A Comparative Study for Image Classification

### Mandatory models

- Linear / softmax classifier
- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN), self-designed and trained from scratch
- GRU (LSTM is also implemented in `models/rnn.py` as an optional comparison)
- Transformer with row-token projection and sinusoidal positional encoding

### Dataset

- **Primary results:** Fashion-MNIST (70,000 28×28 grayscale images, 10 classes)
- **Development/debugging only:** MNIST
- **Optional extension:** CIFAR-10

The dataset is downloaded automatically by `torchvision.datasets.FashionMNIST` into `data/` on the first run.

### Split and preprocessing

- Official 60,000 train / 10,000 test split.
- The 60,000 training images are split into **48,000 train / 12,000 validation** with `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- Every model uses the same split, seed 42, preprocessing, batch size, optimizer, and epoch budget.
- Images are scaled to [0, 1] and z-scored with the **training-set** mean 0.2863 and standard deviation 0.3532; the same transform is applied to validation and test.
- No augmentation is used in the main comparison. The flip/near-black experiments in `assignment1/eda.ipynb` are EDA-only.

### How to run the code

The reproducible entry point is the executed notebook:

```bash
jupyter notebook assignment1/main.ipynb
```

Run all cells from top to bottom. The notebook:

1. loads Fashion-MNIST and builds the shared DataLoaders,
2. trains Linear, MLP, CNN, GRU, and Transformer for 10 epochs,
3. selects the best checkpoint per model by validation macro-F1,
4. evaluates each checkpoint on the 10,000-image test set,
5. writes checkpoints, metrics, histories, curves, confusion matrices, and prediction examples.

Training protocol: Adam, learning rate `1e-3`, batch size 64, 10 epochs, cross-entropy loss, gradient clipping at max norm 1.0, seed 42.

### Outputs

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

### Results summary

Test metrics on the official 10,000-image test set; checkpoints selected by validation macro-F1.

| Model | Params | Test accuracy | Test macro-F1 | Train (s) | Inference (ms/sample) |
|---|---:|---:|---:|---:|---:|
| Linear | 7,850 | 84.10% | 84.15% | 9.3 | 0.005 |
| MLP | 235,146 | 87.89% | 87.89% | 12.0 | 0.007 |
| CNN | 455,946 | 91.90% | 91.86% | 20.9 | 0.011 |
| GRU | 61,962 | 88.93% | 88.85% | 14.5 | 0.007 |
| Transformer | 69,450 | 88.71% | 88.69% | 34.2 | 0.013 |

The reported run is tagged `a1-m2-final` in this repository.

The full discussion, curves, confusion matrices, error analysis, and per-class results are on the Assignment 1 page:

- [Assignment 1 page](assignment1/index.html)
- [Results JSON](assignment1/results/results.json)
- [Results CSV](assignment1/results/results.csv)
- [Run configuration](assignment1/results/config.json)
- [AI usage disclosure](assignment1/AI_USAGE.md)

### Reproducibility notes

- Seed: 42, set for Python, NumPy, PyTorch CPU, and CUDA.
- Checkpoint selection: highest validation macro-F1.
- Reported test numbers come from the selected checkpoint, not the final epoch.
- Training time includes per-epoch validation and checkpointing; inference time is a full forward pass over the test set after warmup, excluding loss/metric computation.
- Report, slides, and video links will be added to the Assignment 1 page when those deliverables are finalized.

### Repository layout

```text
assignment1/
  main.ipynb           five-model training and evaluation notebook
  eda.ipynb            dataset EDA
  training_loop.py     shared model-agnostic training/validation functions
  models/
    linear.py
    mlp.py
    cnn.py
    rnn.py             GRU, LSTM, and plain RNN variants
    transformer.py     row-token Transformer
  checkpoints/         best checkpoints from the reported run
  results/             metrics, histories, curves, confusion matrices, examples
utils/
  dataloader.py        unified torchvision loading and stratified split
```

---

## Optional extensions

The code also supports CIFAR-10 through `load_split("cifar10")`, an LSTM variant through `LSTMClassifier`, and an EDA-only augmentation experiment. These are not part of the main five-model comparison reported above.
