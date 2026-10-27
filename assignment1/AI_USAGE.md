# AI Usage Disclosure — Assignment 1

## Course Project: CO3133 - Deep Learning and Its Applications — Group G-03

**Entry 1 — Shared data pipeline (utils/dataloader.py)**
- Tool/model: opencode CLI (GLM-5.3)
- Used by: Nguyen Van An
- Stage: A1 M1 (21–22 Sep 2026)
- Purpose: Unify dataset loading and split construction.
- Affected files/sections: utils/dataloader.py; assignment1/eda.ipynb;
  utils/get_data.py (removed); report Part 1.
- Prompt summary: Requested a unified torchvision-based loader with a shared
  split interface.
- AI assistance: helped refactor the loader and unify the data interface.
- Student verification: Nguyen Van An reviewed the diff and tested split shapes,
  stratification, determinism, and EDA execution.
- Verification scope: data pipeline code.
- Responsible member: Nguyen Van An
- Sources: torchvision documentation; official CIFAR-10 checksum; handbook §10.

**Entry 2 — Sample image visualization and pixel highlighting (eda.ipynb)**
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

**Entry 3 — Image augmentation (eda.ipynb)**
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

**Entry 4 — Model modules and M2 experiment support**
- Tool/model: opencode CLI (GLM-5.3)
- Used by: Nguyen Van An
- Stage: A1 M1–M2 (22 Sep–25 Oct 2026)
- Purpose: Clean up the model modules and support the five-model experiment.
- Affected files/sections: assignment1/models/{linear,mlp,cnn,rnn,transformer}.py;
  assignment1/main.ipynb; assignment1/index.html; README.md.
- Prompt summary: Requested documentation cleanup, barebone RNN/Transformer
  scaffolds, and targeted help with the M2 notebook and page updates.
- AI assistance: helped with docstrings, module cleanup, initial RNN
  cell-loop and row-token Transformer scaffolds, and drafting the M2 notebook
  workflow and page text.
- Student verification: Nguyen Van An specified the architecture and module
  boundaries, reviewed and edited the code, ran the notebook, and checked
  reported numbers against results/results.json.
- Verification scope: model code, notebook, and documentation.
- Responsible member: Nguyen Van An
- Sources: handbook §§11.1.3–11.1.5, 3.3, 4.2, 12.1; PyTorch documentation;
  Vaswani et al. (2017).
