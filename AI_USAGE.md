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

**Entry 2 — Dataset download utility (utils/get_data.py)**
- Tool: opencode CLI (GLM-5.3)
- Used by: Nguyen Van An
- Stage: A1 M1 development — data pipeline setup (21 Sep 2026)
- Purpose: Implement the dataset download helper per the user's skeleton
- Prompt summary: "Check the code I write in utils/get_data.py. I have made an
  example docstring in get_dataset, which will be the standard arguments. Can
  you expand it to get_all_datasets too, and check if the download links are
  correct? If yes, then help me complete the get_dataset function given the
  hints in the docstring."
- AI contribution: implemented the download logic following the user-specified
  signature, docstring standard, and code layout; found the Fashion-MNIST URL
  invalid (404) and replaced it with the four IDX files keras itself
  distributes; per user request, located a faster CIFAR-10 mirror and verified
  it byte-identical to the official file via MD5
- Student verification: user iteratively reviewed and corrected the output
  (docstring wording, folder layout, link semantics, return values); final
  version tested by real download runs with MD5s checked against torchvision's
  published checksums; file structure and shapes verified
- Affected files: utils/get_data.py
- Responsible member: Nguyen Van An
- Sources: keras/src/datasets/fashion_mnist.py; torchvision/datasets/mnist.py
  (published MD5s); official CIFAR-10 checksum
