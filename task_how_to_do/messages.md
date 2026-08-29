Разделил структурно вместо блэклиста: датасеты уехали в config-группу, и раннер берёт cfg.dataset целиком — что там лежит, его не касается.

conf/
  config.yaml            # steps + hydra, всё про запуск
  dataset/
    nback.yaml           # монолит pipeline
    stroop.yaml

config.yaml:

yaml
defaults:

- dataset: nback
- _self_

steps: init,preprocessing

hydra:
  job: {chdir: false}
  run:   {dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}}
  sweep: {dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}}

В раннере вместо RUNNER_KEYS теперь одна строка:

python
params = OmegaConf.to_container(cfg.dataset, resolve=True)

Граница проходит по дереву конфига, а не по списку имён. Добавишь в config.yaml ещё служебных полей — dry_run, subjects, что угодно — трогать питон не придётся, они просто не в cfg.dataset.

Побочно ушло дублирование: блок hydra: был скопирован в оба датасетных файла, теперь он один. И запуск стал естественнее — dataset=stroop вместо --config-name stroop, а значит работает sweep.

bash
python run_pipeline.py dataset=nback
python run_pipeline.py dataset=stroop steps=sensor
python run_pipeline.py --multirun dataset=nback,stroop
python run_pipeline.py dataset=stroop dataset.h_freq=40.0

Всё четыре прогнал. Multirun развёл датасеты по своим папкам (bids_root = '/home/claude/bids_stroop' во втором job'е), override дал h_freq = 40.0 в сгенерированном конфиге, guard на +dataset.bogus=1 сработал.

Одна деталь синтаксиса: внутри dataset/*.yaml интерполяция стала относительной — deriv_root: ${.bids_root}/derivatives/... с точкой. Абсолютный ${bids_root} там теперь не разрешится, потому что в итоговом дереве ключ лежит на dataset.bids_root. В сгенерированном конфиге проверил: deriv_root = '/home/claude/bids_nback/derivatives/mne-bids-pipeline' — раскрылось верно.
