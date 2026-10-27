# CO3133 - Deep Learning and Its Applications - Course Project

## Course Information

**Course:** Deep Learning and Its Applications  
**Course code:** CO3133  
**Semester:** 261  
**Instructor:** Le Thanh Sach  
**Group:** G-03

## Group Members

| Full name | Student ID | Role | GitHub |
|---|---:|---|---|
| Nguyen Van An | 2352013 | Data pipeline, models, experiments | [BanAnA9205](https://github.com/BanAnA9205) |
| Vo Le Hai Dang | 2352257 | Landing page, repository coordination | [DangXiMi](https://github.com/DangXiMi) |
| Huynh Vuong Khang | 2350011 | EDA and visualization | [hvhvk](https://github.com/hvhvk) |

**Group code repository:**  
[https://github.com/DangXiMi/CO3133-DeepLearning](https://github.com/DangXiMi/CO3133-DeepLearning)

## Assignment index

Each assignment has its own README with setup, dataset, run instructions, and results:

- [Assignment 1 — Foundations of Deep Learning Pipelines and Architectures](assignment1/README.md)
- [Assignment 2 — Deep Learning on Large-Scale Data and Specialized Tasks](assignment2/README.md)
- [Assignment 3 — Multimodal Deep Learning](assignment3/index.html)

## Quick start

```bash
git clone https://github.com/DangXiMi/CO3133-DeepLearning.git
cd CO3133-DeepLearning
```

Create and activate a virtual environment, then install the requirements for the assignment you want to run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r assignment1/requirements.txt   # Assignment 1
```

Assignment-specific commands are documented in each assignment’s README.

## Repository layout

```text
assignment1/   Assignment 1 code, results, checkpoints, materials, README
assignment2/   Assignment 2 code, results, checkpoints, materials, README
assignment3/   Assignment 3 page and materials (work in progress)
utils/         shared data loading and experiment helpers
data/          downloaded datasets (not committed)
```

## AI usage disclosure

See [AI_USAGE.md](AI_USAGE.md) and each assignment’s `AI_USAGE.md` for details.
