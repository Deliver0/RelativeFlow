# RelativeFlow

This repository contains the normalized open-source implementation of the RelativeFlow v2 route for CT and MR restoration.

The released code follows the paper-facing route and does not include the quality encoder branch.

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

## Install

```bash
pip install -r requirements.txt
```

## Data List Format

Training uses a text file in which each line contains either:

```text
target_path
```

or the legacy compatible form:

```text
input_path,target_path
```

Prediction and evaluation use a text file in which each line contains:

```text
input_path,target_path
```

For training, only the clean or high-quality target path is used for loading. The degraded input is generated online during training.

## Training

Default settings are aligned with the paper-oriented release:

- `L2` loss
- `30` epochs
- initial `Δt_min = 0.2`
- initial `Δt_max = 0.2`
- `α = 0.9` with epoch-wise update of the training delta range
- CT and MR velocity denominator both use `exp(delta_t) - 1`
- checkpoints are saved to `weights/ct` and `weights/mr`

```bash
python train.py --modality ct --io-format dicom --train-file <ct_target_list.txt>
python train.py --modality mr --io-format nifti --train-file <mr_target_list.txt>
```

## Prediction

Default iterative sampling uses:

- delta sequence: `0.2,0.1,0.05`
- update rule: `x = x + delta_t * u`
- output directories: `outputs/ct/testA` and `outputs/mr/testA`

```bash
python predict.py --modality ct --io-format dicom --test-file <paired_path_CT_testA.txt> --ckpt <checkpoint>
python predict.py --modality mr --io-format nifti --test-file <paired_path_MR_testA.txt> --ckpt <checkpoint>
```

## Evaluation

```bash
python evaluate.py --modality ct --io-format dicom --test-file <paired_path_CT_testA.txt> --predictions-dir <dir>
python evaluate.py --modality mr --io-format nifti --test-file <paired_path_MR_testA.txt> --predictions-dir <dir>
```

If `--predictions-dir` is omitted, the scripts use the default prediction output directory.

## Notes

- Image I/O format is selected explicitly through `--io-format` rather than being inferred from modality.
- The repository is organized as root-level entry scripts plus `models/` and `utils/`, following a standard GitHub project layout.

## License

This project is released under the Apache-2.0 License.
