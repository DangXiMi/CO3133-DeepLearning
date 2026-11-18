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