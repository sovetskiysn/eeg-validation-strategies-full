"""MNE-BIDS-Pipeline settings and labelled analysis epochs."""

from __future__ import annotations

import fcntl
import logging
import shutil
import subprocess
import tempfile
import types
import typing
from collections import deque
from collections.abc import Sequence as ABCSequence
from pathlib import Path
from pprint import pformat

import mne
import mne_bids
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from mne_bids_pipeline import _config as mbp_config
from omegaconf import DictConfig, OmegaConf

from utils import PROJECT_ROOT


log = logging.getLogger(__name__)


# The native BIDS vocabulary is written by dataset_standardization. This mapping
# is the research question: it collapses conditions into the shared, protocol-
# defined target classes. Preparation recipes select BIDS tasks, but never
# redefine a condition's class.
DATASET_MAPPING = {
    "sam40": {
        "low_attention": ["relax"],
        "high_attention": ["stroop", "arithmetic", "mirror"],
    },
    "distinguishing": {
        "low_attention": ["unfocused", "drowsy"],
        "high_attention": ["focused"],
    },
}


class _Mismatch(TypeError):
    """A value cannot be represented by one branch of a Pipeline annotation."""


def _coerce(value, annot):
    """Convert Hydra's lists to the concrete container type Pipeline expects."""
    origin = typing.get_origin(annot)

    if annot is typing.Any:
        return value
    if origin is typing.Annotated:
        return _coerce(value, typing.get_args(annot)[0])
    if origin in (typing.Union, types.UnionType):
        for member in typing.get_args(annot):
            try:
                return _coerce(value, member)
            except _Mismatch:
                continue
        raise _Mismatch(annot)
    if annot is type(None):
        if value is None:
            return None
        raise _Mismatch(annot)
    if origin is typing.Literal:
        if value in typing.get_args(annot):
            return value
        raise _Mismatch(annot)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise _Mismatch(annot)
        args = typing.get_args(annot)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(item, args[0]) for item in value)
        if len(args) != len(value):
            raise _Mismatch(annot)
        return tuple(_coerce(item, item_annot) for item, item_annot in zip(value, args))
    if origin in (list, ABCSequence):
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise _Mismatch(annot)
        (item_annot,) = typing.get_args(annot) or (typing.Any,)
        return [_coerce(item, item_annot) for item in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise _Mismatch(annot)
        _, value_annot = typing.get_args(annot) or (typing.Any, typing.Any)
        return {key: _coerce(item, value_annot) for key, item in value.items()}
    if isinstance(annot, type):
        if annot is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, annot) and not (annot is not bool and isinstance(value, bool)):
            return value
        raise _Mismatch(annot)
    return value


def prepare_epochs(preparation_config: DictConfig) -> mne.Epochs:
    """Prepare one dataset and return its labelled analysis epochs."""
    # =============================================================================
    # Step 1: validate the selected dataset conditions
    # =============================================================================
    scenario_name = str(preparation_config.name)
    dataset_name = Path(str(preparation_config.dataset_dir)).name
    save_html_reports = bool(preparation_config.save_preparation_html_reports)
    pipeline_config = preparation_config.mne_bids_pipeline_config
    tasks = pipeline_config.task
    if OmegaConf.is_config(tasks):
        tasks = OmegaConf.to_container(tasks, resolve=True)
    tasks = [tasks] if isinstance(tasks, str) else list(tasks)
    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"No condition mapping for {dataset_name!r}: {sorted(DATASET_MAPPING)}.")
    conditions = {
        condition: label
        for label, group in enumerate(DATASET_MAPPING[dataset_name].values())
        for condition in group
    }

    # =============================================================================
    # Step 2: compose MNE-BIDS-Pipeline config and preparation paths
    # =============================================================================
    params = OmegaConf.to_container(pipeline_config, resolve=True)
    assert isinstance(params, dict)
    annotations = mbp_config.__annotations__
    if unknown := sorted(set(params) - set(annotations)):
        raise ValueError(
            f"Unknown mne_bids_pipeline_config settings {unknown}. "
            "Check the preparation YAML for a typo or a renamed option."
        )
    params = {key: _coerce(value, annotations[key]) for key, value in params.items()}
    derivative_root = Path(params["deriv_root"])
    if not derivative_root.is_absolute():
        derivative_root = PROJECT_ROOT / derivative_root
    derivative_root.mkdir(parents=True, exist_ok=True)
    hydra_config = HydraConfig.get()
    output_root = Path(
        hydra_config.sweep.dir
        if hydra_config.mode == RunMode.MULTIRUN
        else hydra_config.run.dir
    )
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    destination_dir = output_root / "_preparation" / scenario_name
    job_output_dir = Path(HydraConfig.get().runtime.output_dir)
    if not job_output_dir.is_absolute():
        job_output_dir = PROJECT_ROOT / job_output_dir
    pipeline_log_path = (
        job_output_dir.resolve() / "_preparation" / f"{scenario_name}.mne-bids-pipeline.log"
    )

    # =============================================================================
    # Step 3: run MNE-BIDS-Pipeline preprocessing and snapshot its reports
    # =============================================================================
    # Pipeline reports are mutable subject/session files shared by all tasks and
    # runs in one derivatives root. The same lock protects preprocessing and the
    # snapshot, so concurrent jobs can reuse one complete report copy safely.
    with (derivative_root / ".preparation.lock").open("w") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", encoding="utf-8") as stream:
            stream.write("# Generated from the composed Hydra preparation.\n")
            for key, value in sorted(params.items()):
                stream.write(f"{key} = {pformat(value, sort_dicts=True)}\n")
            stream.flush()
            command = [
                "mne_bids_pipeline",
                "--config",
                stream.name,
                "--steps=preprocessing",
                "--logger-level=info",
            ]
            log.info(
                "Preparation %s: started for tasks %s (derivatives: %s).",
                scenario_name,
                ", ".join(map(str, tasks)),
                derivative_root,
            )
            cache_hits = existing_outputs = 0
            output_tail: deque[str] = deque(maxlen=100)
            pipeline_log_path.parent.mkdir(parents=True, exist_ok=True)
            with pipeline_log_path.open("w", encoding="utf-8") as pipeline_log:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert process.stdout is not None
                for line in process.stdout:
                    pipeline_log.write(line)
                    output_tail.append(line)
                    if "Computation unnecessary (cached " in line:
                        cache_hits += 1
                    elif "Computation unnecessary (output files exist)" in line:
                        existing_outputs += 1
                return_code = process.wait()

        if return_code:
            log.error(
                "Preparation %s failed; full Pipeline log: %s\nLast Pipeline output:\n%s",
                scenario_name,
                pipeline_log_path,
                "".join(output_tail),
            )
            raise subprocess.CalledProcessError(return_code, command)
        if cache_hits:
            cache_status = f"reused ({cache_hits} Pipeline cache hits)"
        elif existing_outputs:
            cache_status = f"not reused; {existing_outputs} existing outputs were kept"
        else:
            cache_status = "not reused"

        if save_html_reports:
            report_paths = sorted(
                report_path
                for report_path in derivative_root.rglob("*_report.html")
                if "sub-average" not in report_path.parts
            )
            if not report_paths:
                raise FileNotFoundError(
                    f"{derivative_root} contains no MNE-BIDS-Pipeline HTML reports."
                )
            completed_marker = destination_dir / ".snapshot_complete"
            source_marker = destination_dir / ".source_derivative_root"
            if completed_marker.exists():
                saved_source = source_marker.read_text(encoding="utf-8").strip()
                if saved_source != str(derivative_root):
                    raise RuntimeError(
                        f"{destination_dir} already snapshots {saved_source}, not {derivative_root}."
                    )
            else:
                if destination_dir.exists():
                    if any(destination_dir.iterdir()):
                        raise RuntimeError(
                            f"{destination_dir} exists without .snapshot_complete; "
                            "remove the incomplete snapshot before rerunning."
                        )
                    destination_dir.rmdir()
                destination_dir.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    dir=destination_dir.parent, prefix=f".{scenario_name}.snapshot-"
                ) as temporary_dir:
                    staged_dir = Path(temporary_dir) / scenario_name
                    for report_path in report_paths:
                        destination = staged_dir / report_path.relative_to(derivative_root)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(report_path, destination)
                    (staged_dir / ".source_derivative_root").write_text(
                        f"{derivative_root}\n", encoding="utf-8"
                    )
                    (staged_dir / ".snapshot_complete").touch()
                    staged_dir.replace(destination_dir)
        else:
            destination_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "Preparation %s: complete; cache %s; derivatives: %s; HTML report snapshot: %s; full log: %s.",
            scenario_name,
            cache_status,
            derivative_root,
            destination_dir if save_html_reports else "disabled",
            pipeline_log_path,
        )

    # =============================================================================
    # Step 4: find the prepared recordings
    # =============================================================================
    pipeline = OmegaConf.create(params)
    window_size = float(pipeline.rest_epochs_duration)
    overlap = float(pipeline.rest_epochs_overlap)
    if not 0 <= overlap < window_size:
        raise ValueError("rest_epochs_overlap must lie in [0, rest_epochs_duration).")

    processing = "clean" if pipeline.spatial_filter == "ica" else "filt"
    raw_paths = sorted(
        mne_bids.find_matching_paths(
            root=derivative_root,
            datatypes="eeg",
            tasks=tasks,
            processings=processing,
            suffixes="raw",
            extensions=".fif",
            check=False,
        ),
        key=lambda path: (path.subject or "", path.session or "", path.task or "", path.run or ""),
    )
    raw_paths = [path for path in raw_paths if path.split in (None, "01")]
    if not raw_paths:
        raise FileNotFoundError(
            f"{derivative_root} contains no proc-{processing} raw derivatives after Pipeline ran."
        )
    if missing_tasks := sorted(set(tasks) - {path.task for path in raw_paths}):
        raise FileNotFoundError(
            f"{dataset_name}: no prepared recordings found for requested task(s) {missing_tasks}."
        )

    # =============================================================================
    # Step 5: cut labelled analysis epochs from every recording
    # =============================================================================
    expected_channels = list(pipeline.analyze_channels)
    epochs_by_recording = []
    for raw_path in raw_paths:
        raw = mne.io.read_raw_fif(raw_path.fpath, preload=True, verbose=False)
        if set(raw.ch_names) != set(expected_channels):
            raise ValueError(
                f"{raw_path.fpath} has channels {raw.ch_names}, expected {expected_channels} "
                "from the preparation config."
            )
        raw.pick(expected_channels)
        blocks = [
            annotation
            for annotation in raw.annotations
            if annotation["description"] in conditions and annotation["duration"] >= window_size
        ]
        if not blocks:
            continue
        block_events = [
            mne.make_fixed_length_events(
                raw,
                id=conditions[annotation["description"]],
                start=annotation["onset"],
                stop=annotation["onset"] + annotation["duration"],
                duration=window_size,
                overlap=overlap,
            )
            for annotation in blocks
        ]
        events = np.concatenate(block_events)
        sfreq = raw.info["sfreq"]
        duration = round(window_size * sfreq) / sfreq
        starts = (events[:, 0] - raw.first_samp) / sfreq
        recording_unit = "_".join(
            [dataset_name, f"sub-{raw_path.subject}"]
            + ([] if raw_path.session is None else [f"ses-{raw_path.session}"])
            + ([] if raw_path.run is None else [f"run-{raw_path.run}"])
        )
        epochs_by_recording.append(
            mne.Epochs(
                raw,
                events,
                event_id={
                    class_name: code
                    for code, class_name in enumerate(DATASET_MAPPING[dataset_name])
                    if code in events[:, 2]
                },
                tmin=0.0,
                tmax=duration - 1.0 / sfreq,
                baseline=None,
                preload=True,
                reject_by_annotation=True,
                metadata=pd.DataFrame(
                    {
                        "dataset": dataset_name,
                        "subject": raw_path.subject,
                        "session": raw_path.session,
                        "task": raw_path.task,
                        "run": raw_path.run,
                        "recording_unit": recording_unit,
                        "label": events[:, 2],
                        "condition": np.repeat(
                            [annotation["description"] for annotation in blocks],
                            [len(one) for one in block_events],
                        ),
                        "window_start_s": starts,
                        "window_stop_s": starts + duration,
                    }
                ),
                verbose=False,
            ).set_annotations(None)
        )
    if not epochs_by_recording:
        raise ValueError(f"{dataset_name}: selected tasks produced no labelled analysis windows.")
    epochs = mne.concatenate_epochs(epochs_by_recording, on_mismatch="raise", verbose=False)
    observed_labels = set(epochs.events[:, 2])
    missing_classes = [
        class_name
        for label, class_name in enumerate(DATASET_MAPPING[dataset_name])
        if label not in observed_labels
    ]
    if missing_classes:
        raise ValueError(
            f"{dataset_name}: selected recordings contain no analysis windows for {missing_classes}."
        )
    return epochs
