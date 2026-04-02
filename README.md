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

Training and evaluation scripts use a text file in which each line contains:

```text
input_path,target_path
```

For training, the target path should point to the clean or high-quality image.

## Training

Default settings are aligned with the paper-oriented release:

- `L2` loss
- `30` epochs
- CT and MR velocity denominator both use `exp(delta_t) - 1`
- checkpoints are saved to `weights/ct` and `weights/mr`

```bash
python train.py --modality ct --train-file <paired_path_CT_train.txt>
python train.py --modality mr --train-file <paired_path_MR_train.txt>
```

## Prediction

Default iterative sampling uses:

- delta sequence: `0.2,0.1,0.05`
- update factor: `expm1`
- output directories: `outputs/ct/testA` and `outputs/mr/testA`

```bash
python predict.py --modality ct --test-file <paired_path_CT_testA.txt> --ckpt <checkpoint>
python predict.py --modality mr --test-file <paired_path_MR_testA.txt> --ckpt <checkpoint>
```

## Evaluation

```bash
python evaluate.py --modality ct --test-file <paired_path_CT_testA.txt> --predictions-dir <dir>
python evaluate.py --modality mr --test-file <paired_path_MR_testA.txt> --predictions-dir <dir>
```

If `--predictions-dir` is omitted, the scripts use the default prediction output directory.

## Notes

- CT data is handled through DICOM I/O.
- MR data is handled through NIfTI I/O.
- The repository is organized as root-level entry scripts plus `models/` and `utils/`, following a standard GitHub project layout.

## License

This project is released under the Apache-2.0 License.
