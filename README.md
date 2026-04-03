# RelativeFlow

Official code for the RelativeFlow paper on CT and MR image denoising.

Medical image denoising suffers from the noisy reference problem, where only heterogeneous noisy references are available for supervision. RelativeFlow addresses this issue by learning relative noisier-to-noisy mappings that progressively compose a unified flow toward high-quality targets.

## Highlights

- RelativeFlow decomposes the ill-posed absolute noise-to-clean mapping into relative noisier-to-noisy mappings and composes them into a unified denoising flow.
- RelativeFlow realizes this formulation with consistent transport (CoT) and simulation-based velocity field (SVF), enabling denoising from heterogeneous noisy references in both CT and MR.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Train with `train.py`

`--modality` selects the restoration setting (`ct` or `mr`). `--io-format` specifies how target images are read (`dicom` or `nifti`). `--train-file` lists clean or high-quality target paths, one per line; the legacy `input_path,target_path` format is also supported. During training, degraded inputs are generated online from the target images.

```bash
python train.py --modality ct --io-format dicom --train-file <ct_target_list.txt>
python train.py --modality mr --io-format nifti --train-file <mr_target_list.txt>
```

### Predict with `predict.py`

`--test-file` uses paired paths in the form `input_path,target_path`. `--ckpt` is the checkpoint to load. `--io-format` specifies how the input and output images are read and saved.

```bash
python predict.py --modality ct --io-format dicom --test-file <paired_path_CT_testA.txt> --ckpt <checkpoint>
python predict.py --modality mr --io-format nifti --test-file <paired_path_MR_testA.txt> --ckpt <checkpoint>
```

### Evaluate with `evaluate.py`

`--test-file` uses the same paired list as prediction. `--predictions-dir` points to the folder containing restored outputs and can be omitted to use the default prediction directory.

```bash
python evaluate.py --modality ct --io-format dicom --test-file <paired_path_CT_testA.txt> --predictions-dir <dir>
python evaluate.py --modality mr --io-format nifti --test-file <paired_path_MR_testA.txt> --predictions-dir <dir>
```

## Repository Structure

```text
RelativeFlow/
├─ train.py
├─ predict.py
├─ evaluate.py
├─ models/
│  ├─ unet.py
│  ├─ nn.py
│  └─ fp16_util.py
├─ utils/
│  ├─ data.py
│  ├─ simulation.py
│  └─ metrics.py
├─ requirements.txt
├─ LICENSE
└─ .gitignore
```
