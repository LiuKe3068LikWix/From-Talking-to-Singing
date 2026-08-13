# From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection

Official inference code for our ICML 2026 paper. This repository provides the
**T-AVFD** model, the complete original checkpoint, and evaluation code for the
**Singing Head DeepFake (SHDF)** dataset.

Project page: https://liuke3068likwix.github.io/SingingHead-DeepFake/

## Installation

```bash
conda create -n tavfd python=3.10 -y
conda activate tavfd
pip install -r requirements.txt
```

## Checkpoint

Download the complete original ICML checkpoint from
[Google Drive](https://drive.google.com/drive/folders/1frssYQ54WNDkjJ-Nde_-Kv_recUCr2Rr?usp=sharing)
and place it at:

```text
checkpoints/TAVFD.pt
```

## SHDF features

Download the processed SHDF NPZ package from
[Google Drive](https://drive.google.com/drive/folders/1frssYQ54WNDkjJ-Nde_-Kv_recUCr2Rr?usp=sharing)
and extract it under the `data` directory as follows:

```text
data/SHDF_features/
  0_real/*.npz
  1_fake/*.npz
```

Each NPZ contains:

* `visual`: `[T, 1024]`
* `audio`: `[T, 1024]`
* `global`: `[1, 768]` or `[768]`
* `local`: optional compatibility field

## Evaluation

```bash
python eval.py --data_root data/SHDF_features --device cuda:0
```

The script loads `checkpoints/TAVFD.pt` by default and evaluates the NPZ feature
files under the `0_real` and `1_fake` subdirectories. It reports AP and AUC and
saves:

```text
outputs/SHDF_scores.csv
outputs/SHDF_scores.json
```

Labels are `0 = real` and `1 = fake`. Higher scores indicate a greater
likelihood of being fake.

The interface for other processed datasets is also retained:

```bash
python eval.py --metadata /path/to/test.csv --data_root /path/to/npz_root --output outputs/other_scores.csv --device cuda:0
```

The metadata CSV must contain `path,label` columns and use the same NPZ format.

## Citation

```bibtex
@inproceedings{liu2026from,
  title        = {From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection},
  author       = {Liu, Ke and Wei, Jiwei and Zhang, Wenyu and Zhou, Shuchang and Chai, Ruikun and Dai, Yutao and Zhang, Chaoning and Yang, Yang},
  booktitle    = {International Conference on Machine Learning},
  year         = {2026},
  organization = {PMLR}
}
```
