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

**Entry 3 — Duplicate image detection (eda.ipynb)**
- Tool: Gemini
- Used by: Huynh Vuong Khang
- Stage: A1 M1 development — EDA (21 Sep 2026)
- Purpose: Plotting images for each class with selected pixels highlighted.
- Prompt summary: "Use matplotlib.pyplot to plot 5 grayscale images for a class, and highlight user-specified pixels", "List possible values for cmap and pixel value range"
- AI contribution: implemented a baseline for extracting images of a specified class using **torch.nonzero**, plotting the images and highlights with **matplotlib.pyplot.imshow**, suggested alternives for "cmap" arguments other than the "grayscale" color map.
- Student verification: reviewed official PyTorch and Matplotlib documents, rewritten pixel masking, color map and highlighting configuration.   
- Affected files: assignment1/eda.ipynb
- Responsible member: Huynh Vuong Khang
- Sources: [matplotlib.pyplot.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html#matplotlib.pyplot.imshow), [matplotlib.colors.ListedColormap](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html), [torch.nonzero](https://docs.pytorch.org/docs/2.14/generated/torch.nonzero.html).

**Entry 4 — Image augmentation (eda.ipynb)**
- Tool: Gemini
- Used by: Huynh Vuong Khang
- Stage: A1 M1 development — EDA (21 Sep 2026)
- Purpose: Searching PyTorch tools for randomly flipping images of class 5, 7 and 9 (shoe-like items).
- Prompt summary: "Without giving code, give me pytorch methods to implement random horizontal image flip for class 5, 7 and 9"
- AI contribution: listed **torch.isin** for masking images of class 5, 7 and 9, **torch.rand** to implement probability, **torch.flip** to implement flipping operation.
- Student verification: reviewed official PyTorch documents for usage and return values, manually implemented a script based on given tools.   
- Affected files: assignment1/eda.ipynb
- Responsible member: Huynh Vuong Khang
- Sources: [torch.isin](https://docs.pytorch.org/docs/2.14/generated/torch.isin.html#torch.isin), [torch.rand](https://docs.pytorch.org/docs/2.14/generated/torch.rand.html#torch.rand), [torch.flip](https://docs.pytorch.org/docs/2.14/generated/torch.flip.html#torch-flip)
