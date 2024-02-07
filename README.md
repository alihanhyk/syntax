# Adaptive Experiment Design with Synthetic Controls
Code author: Alihan Hüyük ([ah2075@cam.ac.uk](mailto:ah2075@cam.ac.uk))

This repository is for reproducing the main experimental results in the AISTATS'24 paper: ["Adaptive experiment design with synthetic controls"](https://arxiv.org/abs/2401.17205). The method proposed in the paper, *Syntax*, is implemented as a combination of `infer_synthetic` and `recruit_adaptive` in `src/algs.py`.

### Usage

First, clone the repository and install the required python packages by running:
```shell
python -m pip install -r requirements.txt
```
Optionally, install a LaTeX distribution, which is only required for figure generation.

Then, the main experimental results can be reproduced by running:
```shell
python src/main.py
python src/eval.py > res/tab-main.txt       # prints Table 2
python src/eval-misx.py > res/tab-misx.txt  # prints Table 3
python src/eval-misz.py > res/tab-misz.txt  # prints Table 4
python src/plot.py                          # generates Figures 2 and 3

python src/main-sens.py
python src/plot-sens.py                     # generates Figure 4
```

### Citing

If you use this software, please cite the original paper:
```
@inproceedings{huyuk2024adaptive,
  author={Alihan H\"uy\"uk and Zhaozhi Qian and Mihaela van der Schaar},
  title={Adaptive experiment design with synthetic controls},
  booktitle={Proceedings of the 27th International Conference on Artificial Intelligence and Statistics},
  year={2024}
}
```