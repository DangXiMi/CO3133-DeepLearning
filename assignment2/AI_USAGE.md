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
- Purpose: Reuse the already-finished Assignment 1 code for Assignment 2 instead of writing a new pipeline from scratch.
- Prompt summary: "the structure should look like assignment 1, as in we should reuse as much as possible" and "only write bare bones then add in [TODO]"
- AI contribution: adapted the existing Assignment 1 CNN and training-loop conventions to Food-101, added a Food-101 DataLoader to utils/dataloader.py, and added a torchvision ResNet-18 builder with a freeze flag for the later ablation; no new architecture was invented.
- Student verification: the user reviewed the reused components, ran the notebook end-to-end on the A100, checked the preliminary accuracy/macro-F1 numbers, and confirmed the baseline/pretrained distinction.
- Affected files: assignment2/main.ipynb; assignment2/training_loop.py; assignment2/models/baseline.py; assignment2/models/resnet.py; utils/dataloader.py
- Responsible member: Nguyen Van An
- Sources: Assignment 1 code in this repository; torchvision ResNet documentation; CO3133 handbook §20
