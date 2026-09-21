# AI Usage Disclosure Log

## Course Project: CO3133 - Deep Learning and Its Applications

### Assignment 1

**Entry 1 — Landing page HTML generation**
- Tool: ChatGPT-4
- Used by: Vo Le Hai Dang
- Stage: Setup (Sep 2026)
- Purpose: Generate initial HTML structure for the landing page and assignment pages
- Prompt summary: *To be filled in by member (Vo Le Hai Dang).*
- AI contribution: boilerplate HTML structure
- Student verification: manually reviewed and adjusted against handbook §2.1–2.2
- Affected files: index.html, assignment1–3 pages
- Responsible member: Vo Le Hai Dang
- Sources: CO3133 handbook

**Entry 2 — Data pipeline refactor (utils/dataloader.py)**
- Tool: opencode CLI (GLM-5.3)
- Used by: Nguyen Van An
- Stage: A1 M1 development — data pipeline refactor (21–22 Sep 2026)
- Purpose: Refactor the data loading and DataLoader code out of the EDA notebook
  into a shared module, unifying all dataset reading into one single method
- Prompt summary: "Bring the dataloader from eda to utils, then keep use the
  torchvision datasets and delete get_data" — followed by the request for a
  unified dataloader selecting mnist/fashion_mnist/cifar10 by argument
- AI contribution: moved the notebook's loading and DataLoader construction into
  utils/dataloader.py; generalized it into a single load_split(dataset_name)
  dispatching over torchvision datasets; normalized all datasets to a unified
  format (uint8 (N, C, H, W) images, int64 labels) per the user's decision;
  updated the notebook cells to import the module; removed the superseded
  utils/get_data.py download utility
- Student verification: user directed each step (torchvision as the data source,
  the unified interface, the channels-first format) and reviewed every diff;
  tested by real runs: split shapes/stratification/determinism verified for all
  three datasets, CIFAR loaded from an MD5-verified local tarball (checksum
  matches the official file), and the full EDA notebook executed end-to-end
- Affected files: utils/dataloader.py; assignment1/eda.ipynb; utils/get_data.py (deleted)
- Responsible member: Nguyen Van An
- Sources: torchvision documentation; official CIFAR-10 checksum (MD5 c58f30108f718f92721af3b95e74349a)
