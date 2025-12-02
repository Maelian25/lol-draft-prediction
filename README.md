# League of Legends Draft Prediction Bot 🎮

A machine learning project to predict optimal champion picks during the draft phase in League of Legends ranked games.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)
# lol-draft-prediction

League of Legends Draft Prediction — a Python project for collecting,
analyzing, and training models to predict champion picks and bans during
the draft phase. The repository supports multi-region scraping, dataset
analysis, preprocessing for PyTorch, and training two model types
(Transformer and an MLP).

This README documents the repository layout, how to install
dependencies, and how to run the pipeline from the command line
(PowerShell examples included).

---

## Overview

This project gathers high-ELO match data, reconstructs draft states
(picks and bans), analyzes champion statistics and synergies, and
trains models to predict picks during a draft. It was originally
designed to work with Riot's APIs and includes utilities to handle
rate limits, caching, and preprocessing for efficient training.

The codebase now exposes a small CLI entrypoint (`main.py`) and a set
of pipeline modules under `src/pipeline` to keep responsibilities
separated and the main script concise.

## Project layout (important files/folders)

```
main.py                       # CLI entrypoint (orchestrates pipeline)
src/
  pipeline/                   # scraping, analysis, training modules
  ML_models/                   # model definitions (MLP, Transformer)
  ML_training/                 # training utilities and trainer class
  analysis/                    # dataset analysis helpers
  data_scrapping/              # dataset extraction code and Dataset class
  utils/                       # constants, logging, helpers
data/                          # raw/cached datasets, checkpoints
requirements.txt               # Python dependencies
README.md                      # This file
```

## Installation (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) Quick import test to confirm modules are available:

```powershell
python -c "import src.pipeline; print('pipeline ok')"
```

## CLI Usage (main.py)

The `main.py` script is the recommended way to run the full
pipeline. It supports the following options:

- `--no-scrape` : do not call external APIs — only use cached
  datasets. Missing cached elos/regions will be skipped (warning).
- `--no-train` : run scraping + analysis only; skip model training.
- `--max-workers N` : set maximum number of threads for scraping.
- `--rebuild` : force rebuild of preprocessed files before training.
- `--model {transformer,mlp}` : choose which model to train. Default is
  `transformer`.

Examples (PowerShell):

```powershell
# Run full pipeline (scrape -> analysis -> train transformer)
python main.py

# Use cached data only and train the MLP
python main.py --no-scrape --model mlp

# Run everything but skip training
python main.py --no-train

# Limit threads to 1 and force preprocessing rebuild
python main.py --max-workers 1 --rebuild
```

Notes:

- If `--no-scrape` is used and no cached data exists for the requested
  regions, the pipeline will fail with `RuntimeError: No data was
  successfully loaded or scraped.` — either provide cached files in
  `data/datasets/` or run without `--no-scrape` to allow scraping.
- Dataset and checkpoint folders are controlled by constants in
  `src/utils/constants.py` (`DATASETS_FOLDER`, `MLP_CHECKPOINTS`,
  `TRANSFORMER_CHECKPOINTS`).

## What each pipeline module does

- `src/pipeline/scraper.py` : per-region threaded loading. It first
  tries to load cached files (via `src.utils.data_helper.load_scrapped_data`),
  and only calls the `Dataset.extract_match_data` API when needed
  (unless `--no-scrape` is set).
- `src/pipeline/analysis.py` : dataset-level EDA and analytics. It
  computes game duration stats, patch distributions, pick/ban rates,
  synergy matrices and champion embeddings, and builds `matches_states`
  ready for preprocessing.
- `src/pipeline/training.py` : calls `preprocess_and_save` and trains
  the requested model. The Transformer is default; pass `--model mlp`
  to train the MLP variant.

## Logs and outputs

- Logging is configured in `src/utils/logger_config.py`. Default logs
  include `main_file.log` and module-specific logs under the `logs/`
  folder.
- Training runs and TensorBoard events are placed under `runs/`.
  Model checkpoints are saved to the checkpoint folders inside
  `data/` as defined by the constants.

## Data format and draft order

Matches are stored as JSON with keys like `match_id`, `game_version`,
`game_duration`, `picks` and `bans`. Picks and bans include champion
IDs, ordering and side. The code also supports reconstructing draft
states if Timeline data is not available, but Timeline-based states are
preferable.

Draft order (standard ranked) is handled according to Riot's
specifications (two ban phases and two pick phases). See
`src/data_scrapping/dataset.py` for exact extraction details.

## Troubleshooting

- "No data was successfully loaded or scraped." — check that there are
  cached files under `data/datasets/<region>/` or run without
  `--no-scrape`.
- Import errors — ensure the virtual environment is activated and
  `requirements.txt` is installed.

## Development suggestions

- During development use `--max-workers 1` to make scraping and logs
  easier to follow.
- Consider adding `--only-analysis` or `--only-scrape` flags if you
  want to run one stage at a time.