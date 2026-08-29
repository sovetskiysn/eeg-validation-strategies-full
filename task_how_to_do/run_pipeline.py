"""Hydra -> mne-bids-pipeline.

conf/config.yaml       — настройки запуска (steps, hydra)
conf/dataset/<имя>.yaml — монолитный конфиг pipeline, уходит в config.py целиком

    python run_pipeline.py dataset=nback
    python run_pipeline.py dataset=stroop steps=sensor
    python run_pipeline.py --multirun dataset=nback,stroop
"""
import subprocess
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from mne_bids_pipeline import _config as mbp_config

from _coerce import coerce


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    ann = mbp_config.__annotations__

    params = OmegaConf.to_container(cfg.dataset, resolve=True)

    # Неизвестные ключи pipeline проглатывает молча — ловим здесь
    if unknown := sorted(set(params) - set(ann)):
        raise ValueError(
            f"Ключи, которых нет в mne_bids_pipeline._config: {unknown}. "
            "Опечатка в YAML или переименованная опция."
        )

    # list <-> tuple по аннотациям mne-bids-pipeline
    params = {k: coerce(v, ann[k]) for k, v in params.items()}

    cfg_path = out_dir / "generated_config.py"
    cfg_path.write_text(
        "# Сгенерировано из Hydra. Не редактировать.\n"
        + "".join(f"{k} = {v!r}\n" for k, v in sorted(params.items())),
        encoding="utf-8",
    )

    cmd = ["mne_bids_pipeline", "--config", str(cfg_path), f"--steps={cfg.steps}"]
    print("[hydra] $", " ".join(cmd))
    if (code := subprocess.run(cmd).returncode) != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
