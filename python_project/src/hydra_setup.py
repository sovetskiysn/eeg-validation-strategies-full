"""Hydra multirun extensions: server hardware telemetry for the sweep's lifetime."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from hydra.experimental.callback import Callback
from omegaconf import DictConfig

_SAMPLE_INTERVAL_S = 30
_CSV_HEADER = (
    "timestamp_utc,gpu_name,gpu_memory_total_mib,gpu_memory_used_mib,"
    "gpu_utilization_pct,gpu_memory_utilization_pct,gpu_temperature_c,"
    "gpu_power_w,ram_total_mib,ram_used_mib,ram_available_mib,load_1m,"
    "disk_available_gib\n"
)


def _sample_line() -> str:
    timestamp = subprocess.run(
        ["date", "-u", "--iso-8601=seconds"], capture_output=True, text=True, check=True
    ).stdout.strip()
    gpu_fields = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,"
            "utilization.memory,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()[0]
    gpu = ",".join(field.strip() for field in gpu_fields.split(","))
    ram_total, ram_used, ram_available = subprocess.run(
        ["free", "-m"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[1].split()[1:4]
    load_1m = Path("/proc/loadavg").read_text().split()[0]
    disk_available = (
        subprocess.run(
            ["df", "-BG", "--output=avail", "."], capture_output=True, text=True, check=True
        )
        .stdout.splitlines()[1]
        .strip()
        .rstrip("G")
    )
    return f"{timestamp},{gpu},{ram_total},{ram_used},{ram_available},{load_1m},{disk_available}\n"


class HardwareTraceCallback(Callback):
    """Sample GPU/RAM/disk/load for the whole multirun sweep, not per job.

    `on_multirun_start`/`on_multirun_end` bracket every job the sweep launches,
    so the poll loop starts and stops exactly once regardless of how many jobs
    run in between -- unlike `on_job_start`/`on_job_end`, which fire per job.
    """

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        sweep_dir = Path(config.hydra.sweep.dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = sweep_dir / "hardware_usage.csv"
        self._log_path.write_text(_CSV_HEADER)

        summary_path = sweep_dir / "hardware_summary.txt"
        with summary_path.open("w") as summary:
            for command in (
                ["lscpu"],
                ["nvidia-smi", "-q"],
                ["free", "-h"],
                ["df", "-h", "."],
            ):
                try:
                    subprocess.run(command, stdout=summary, stderr=subprocess.STDOUT, check=False)
                except OSError as error:  # missing binary must not abort the sweep
                    summary.write(f"# {command[0]} unavailable: {error}\n")

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        self._stop_event.set()
        self._thread.join()

    def _poll(self) -> None:
        while True:
            self._sample()
            if self._stop_event.wait(_SAMPLE_INTERVAL_S):
                return

    def _sample(self) -> None:
        try:
            line = _sample_line()
        except Exception as error:  # telemetry must never abort the sweep
            line = f"# sample failed: {error}\n"
        with self._log_path.open("a") as log:
            log.write(line)
