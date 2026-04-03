# RelativeFlow

RelativeFlow is the normalized open-source release of the paper-facing v2 route for CT and MR image restoration.

This repository focuses on the core RelativeFlow pipeline described in the paper: simulation-based degradation, SVF supervision, and iterative restoration. The current release keeps the project structure compact and does not include the quality encoder branch.

## Highlights

- Unified `train.py`, `predict.py`, and `evaluate.py` entry points for CT and MR.
- Paper-aligned training and sampling defaults for the released implementation.
- Online degradation from clean or high-quality targets during training.
- Explicit `--io-format` control for image reading and writing instead of binding file format to modality.
- Simplified repository layout for easier open-source use and maintenance.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run inference

```bash
python predict.py --modality ct --io-format dicom --test-file <paired_path_CT_testA.txt> --ckpt <checkpoint>
python predict.py --modality mr --io-format nifti --test-file <paired_path_MR_testA.txt> --ckpt <checkpoint>
```

### Evaluate predictions

```bash
python evaluate.py --modality ct --io-format dicom --test-file <paired_path_CT_testA.txt> --predictions-dir <dir>
python evaluate.py --modality mr --io-format nifti --test-file <paired_path_MR_testA.txt> --predictions-dir <dir>
```

### Train a model

```bash
python train.py --modality ct --io-format dicom --train-file <ct_target_list.txt>
python train.py --modality mr --io-format nifti --train-file <mr_target_list.txt>
```

## Repository Scope

- This release follows the paper-facing RelativeFlow v2 route.
- The quality encoder branch is intentionally excluded from this repository.
- `modality` controls the restoration setting and degradation simulator.
- `io-format` controls how images are loaded and saved.

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

## Data Lists

Training reads only the clean or high-quality target path for each sample. Each line in the training list can be either:

```text
target_path
```

or the legacy-compatible form:

```text
input_path,target_path
```

Prediction and evaluation use paired lists in the form:

```text
input_path,target_path
```

During training, the degraded input is generated online from the target image rather than loaded from disk.

## Paper-Aligned Defaults

- Loss: `L2`
- Epochs: `30`
- Initial `Δt_min = 0.2`
- Initial `Δt_max = 0.2`
- Decay factor `α = 0.9` with epoch-wise delta-range update
- Velocity denominator: `exp(delta_t) - 1`
- Sampling steps: `0.2,0.1,0.05`
- Sampling update: `x = x + delta_t * u`

## Detailed Commands

### Training

```bash
python train.py --modality ct --io-format dicom --train-file <ct_target_list.txt>
python train.py --modality mr --io-format nifti --train-file <mr_target_list.txt>
```

Checkpoints are saved by default to `weights/ct` and `weights/mr`.

### Prediction

```bash
python predict.py --modality ct --io-format dicom --test-file <paired_path_CT_testA.txt> --ckpt <checkpoint>
python predict.py --modality mr --io-format nifti --test-file <paired_path_MR_testA.txt> --ckpt <checkpoint>
```

Prediction outputs are written by default to `outputs/ct/testA` and `outputs/mr/testA`.

### Evaluation

```bash
python evaluate.py --modality ct --io-format dicom --test-file <paired_path_CT_testA.txt> --predictions-dir <dir>
python evaluate.py --modality mr --io-format nifti --test-file <paired_path_MR_testA.txt> --predictions-dir <dir>
```

If `--predictions-dir` is omitted, the scripts use the default prediction output directory.

## License

This project is released under the Apache-2.0 License.
