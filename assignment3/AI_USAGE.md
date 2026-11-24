# AI Usage Disclosure — Assignment 3

## Course Project: CO3133 - Deep Learning and Its Applications — Group G-03

**Entry 1 — Dataset Proposal (assignment3/PROPOSAL.md)**
- Tool: ChatGPT (GPT-6 Sol)
- Used by: Huynh Vuong Khang
- Stage: A3 M1 development — dataset proposal (18 Oct 2026)
- Purpose: Draft the Assignment 3 Dataset Proposal for the image-text classification track.
- Prompt summary: "Find some multimodal datasets, where one modality is image and can reuse ResNet-50" and "FInd available sources, versions, and licenses for MM-IMDb" and "Recommend ways to apply stratified splitting for multi-label dataset"
- AI contribution: compiled dataset candidates against the handbook thresholds, verified dataset metadata online (class counts, split sizes, license text), search for `scikit-multilearn`'s `iterative_train_test_split`, and drafted the first 7 proposal fields of handbook §18 for the user's chosen dataset.
- Student verification: The user selected the task track and the dataset, and inspected the proposed method for stratified splitting. The license and splitting plans were referenced from the main sources of the dataset.
- Affected files: assignment3/PROPOSAL.md
- Responsible member: Huynh Vuong Khang
- Sources: CO3133 handbook §26-§29; MM-IMDb [original research paper](https://arxiv.org/abs/1702.01992), [official implementation repository](https://github.com/johnarevalo/gmu-mmimdb), [Hugging Face mirror](https://huggingface.co/datasets/pranavmr/MM-IMDb); `scikit-multilearn.iterative_train_test_split` (official implementation repository)[https://github.com/scikit-multilearn/scikit-multilearn/blob/master/skmultilearn/model_selection/iterative_stratification.py#L175] and [Issue no. 144 on reproducibility](https://github.com/scikit-multilearn/scikit-multilearn/issues/144)

**Entry 2 — M2 Draft: unimodal baselines and simple fusion**
- Tool: opencode CLI (GLM-5.3)
- Used by: Nguyen Van An
- Stage: A3 M2 development — unimodal models and simple fusion (22–23 Nov 2026)
- Purpose: Implement the M2 deliverables: unimodal image and text baselines plus a simple fusion baseline with quantitative results.
- Prompt summary: Extend the A3 proposal to the M2 deliverables: image-only ResNet-18, text-only DistilBERT, and a simple fusion baseline.
- AI contribution: assisted with the mechanical implementation of data loading, iterative multi-label splitting, model/training helpers, threshold tuning, and notebook scaffolding.
- Student verification: the group reviewed the code, ran the notebook end-to-end, and checked the reported metrics against `results/m2_results.json` and `results/m2_results.csv`.
- Affected files: `utils/mmimdb.py`; `assignment3/src/models.py`; `assignment3/src/training_loop.py`; `assignment3/notebooks/main.ipynb`; `assignment3/README.md`; `assignment3/index.html`
- Responsible member: Nguyen Van An
- Sources: MM-IMDb dataset card and original paper; torchvision and Transformers documentation; scikit-multilearn documentation; CO3133 handbook §29–§31

**Entry 3 — M3 Final: GMU ablation and error analysis**
- Tool: opencode CLI (GLM-5.3)
- Used by: Nguyen Van An
- Stage: A3 M3 development — improved fusion and evaluation (23 Nov 2026)
- Purpose: Extend the M2 baselines to the M3 requirements: improved multimodal fusion, controlled ablation, missing-modality evaluation, and support/conflict analysis.
- Prompt summary: Implement the planned GMU and modality-dropout experiments, with ResNet-50 for the image encoder and BERT-base for the text encoder.
- AI contribution: assisted with the mechanical implementation of the GMU forward pass, training/evaluation helpers, missing-modality tests, and per-class and support/conflict analysis code. The model choices and acceptance criteria were decided by the group.
- Student verification: the group reviewed the implementation, ran a 256-example end-to-end smoke test, executed the full M3 experiment, and verified the reported numbers against `results/m3_results.json`, `results/m3_results.csv`, and `results/m3_error_analysis.json`.
- Affected files: `assignment3/src/models.py`; `assignment3/src/training_loop.py`; `assignment3/notebooks/m3.ipynb`; `assignment3/README.md`; `assignment3/index.html`
- Responsible member: Nguyen Van An
- Sources: Arevalo et al. (2017), *Gated Multimodal Units for Information Fusion*; torchvision and Transformers documentation; CO3133 handbook §29–§31

**Entry 4 — Joint multimodal fine-tuning experiment**
- Tool: opencode CLI (GLM-5.3)
- Used by: Nguyen Van An
- Stage: A3 M3 development — end-to-end multimodal fine-tuning (23 Nov 2026)
- Purpose: Add an actual joint fine-tuning run with ResNet-50 and BERT unfrozen, to compare directly with the frozen-encoder GMU ablation.
- Prompt summary: "if there is a fine-tuning strategy then we should do fine-tuning" and "still just do it, do the actual fine-tuning"
- AI contribution: assisted with the mechanical implementation of the end-to-end multimodal wrapper, paired data loader, parameter-group optimization, and missing-modality evaluation. The group chose the fine-tuning strategy and reviewed the implementation.
- Student verification: the group reviewed the model and training code, ran a 256-example smoke test, executed the full 3-epoch joint fine-tune, and verified the reported metrics against `results/m3_finetune_gmu.json` and `results/m3_finetune_gmu.csv`.
- Affected files: `utils/mmimdb.py`; `assignment3/src/models.py`; `assignment3/src/training_loop.py`; `assignment3/notebooks/finetune_gmu.ipynb`
- Responsible member: Nguyen Van An
- Sources: torchvision ResNet-50 and Transformers BERT documentation; CO3133 handbook §29.2

**Entry 5 — Duplicate cleanup and clean reruns**
- Tool: opencode CLI (GLM-5.3)
- Used by: Nguyen Van An
- Stage: A3 M3 development — duplicate cleanup and verification (24 Nov 2026)
- Purpose: Apply the EDA duplicate finding to the actual split and rerun M2, M3, and the joint fine-tune cleanly.
- Prompt summary: the EDA found 6 exact-duplicate pairs; check split leakage, deduplicate, rerun, and verify carefully before pushing.
- AI contribution: inspected the exact duplicate records and split memberships, implemented a deterministic test > dev > train dedup policy with an audit CSV, and reran the M2, M3, and fine-tuning notebooks. Also fixed model-builder checks so M2 explicitly uses ResNet-18 and DistilBERT while M3 uses ResNet-50 and BERT.
- Student verification: the group reviewed the duplicate audit and split indices. All 6 exact-duplicate pairs now have only one member retained in the final split, and the final split is **15,553 train / 2,595 dev / 7,805 test**. The final results and notebook outputs were checked against `results/m2_results.json`, `results/m3_results.json`, and `results/m3_finetune_gmu.json`.
- Affected files: `utils/mmimdb.py`; `assignment3/src/models.py`; `assignment3/notebooks/main.ipynb`; `assignment3/notebooks/m3.ipynb`; `assignment3/notebooks/finetune_gmu.ipynb`; `assignment3/results/dedup_dropped_indices.csv`
- Responsible member: Nguyen Van An
- Sources: `assignment3/results/exact_duplicate_records.csv`; assignment 3 proposal and handbook §28.1 on leakage prevention