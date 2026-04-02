__all__ = [
    "CTDataset",
    "MRDataset",
    "batch_stack",
    "build_output_path",
    "load_ct_dicom",
    "load_mr_nifti",
    "read_paired_paths",
    "save_ct_dicom",
    "save_mr_nifti",
    "DegradationSimulatorCT",
    "DegradationSimulatorMR",
    "MetricComputer",
]


def __getattr__(name: str):
    if name in {"CTDataset", "MRDataset", "batch_stack", "build_output_path", "load_ct_dicom", "load_mr_nifti", "read_paired_paths", "save_ct_dicom", "save_mr_nifti"}:
        from . import data

        return getattr(data, name)
    if name in {"DegradationSimulatorCT", "DegradationSimulatorMR"}:
        from . import simulation

        return getattr(simulation, name)
    if name == "MetricComputer":
        from . import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
