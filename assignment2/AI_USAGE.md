# AI Usage Disclosure — Assignment 2

## Course Project: CO3133 - Deep Learning and Its Applications — Group G-03

**Entry 1 — Dataset Proposal (assignment2/PROPOSAL.md)**
- Tool: opencode CLI (GLM-5.3)
- Used by: Nguyen Van An
- Stage: A2 M1 development — dataset proposal (06 Oct 2026)
- Purpose: Draft the Assignment 2 Dataset Proposal for the image-classification track.
- Prompt summary: "Let's find some good dataset to do" and "Let's do food-101, team has authorized me to choose the dataset"
- AI contribution: compiled dataset candidates against the handbook thresholds, verified dataset metadata online (class counts, split sizes, license text), and drafted all 11 proposal fields of handbook §18 for the user's chosen dataset.
- Student verification: the user selected the task track and the dataset, and the license and split facts were quoted verbatim from the official dataset card before adoption.
- Affected files: assignment2/PROPOSAL.md
- Responsible member: Nguyen Van An
- Sources: CO3133 handbook §17–§19; Food-101 dataset card (huggingface.co/datasets/ethz/food101); official ETH Zürich page

**Entry 2 — M2 Draft: reuse of Assignment 1 pipeline for Assignment 2 (assignment2/main.ipynb and supporting modules)**
- Tool: opencode CLI (GLM-5.3)
- Used by: Nguyen Van An
- Stage: A2 M2 development — draft pipeline and preliminary training (26–27 Oct 2026)
- Purpose: Reuse the Assignment 1 pipeline for Assignment 2 wherever possible, adding only the Food-101 data path and the pretrained ResNet adapter required by A2.
- Prompt summary: Requested that A2 follow the Assignment 1 structure and reuse the A1 pipeline wherever possible.
- AI contribution: reused the A1 CNN, the A1 training/experiment conventions, and the A1 experiment utilities for curves, example predictions, and inference timing. The A2-specific additions were the Food-101 loader in `utils/dataloader.py` and a torchvision ResNet-50 adapter in `assignment2/src/models/resnet.py`; these are dataset/model plumbing, not new architecture.
- Student verification: the user reviewed the reused components, ran the notebook end-to-end on the A100, checked the preliminary accuracy/macro-F1 numbers, and confirmed the baseline/pretrained distinction.
- Affected files: assignment2/main.ipynb; assignment2/training_loop.py; assignment2/models/baseline.py; assignment2/models/resnet.py; utils/dataloader.py; utils/experiment_utils.py
- Responsible member: Nguyen Van An
- Sources: Assignment 1 code in this repository; torchvision ResNet documentation; CO3133 handbook §20

**Entry 3 — M3 Final experiments: multi-seed runs and freeze-vs-full ablation**
- Tool: opencode CLI (GLM-5.3)
- Used by: Nguyen Van An
- Stage: A2 M3 final experiments (28 Oct 2026)
- Purpose: Assist the student in extending the completed A2 M2 pipeline to the A2 M3 experiment setup.
- Prompt summary: Having completed the A2 M2 base pipeline, the student requested assistance extending it to A2 M3: multi-seed training with fixed seeds and a freeze-vs-full ablation.
- AI contribution: assisted with routine notebook plumbing — organizing the seed loop and formatting the per-class metrics, top-confusion extraction, and ablation summary. The experimental design, model choice, seed selection, ablation setup, and acceptance criteria were decided and reviewed by the student.
- Student verification: Nguyen Van An directed the work, reviewed every change, ran the full 30-epoch, 3-seed experiment, and checked the reported accuracy, macro-F1, training time, and inference time against `results/summary.csv` and `results/results.json`.
- Affected files: assignment2/notebooks/main.ipynb; assignment2/src/training_loop.py; assignment2/src/models/resnet.py; assignment2/src/models/baseline.py; assignment2/README.md; assignment2/index.html
- Responsible member: Nguyen Van An
- Sources: CO3133 handbook §20–§23; torchvision ResNet-50 documentation
