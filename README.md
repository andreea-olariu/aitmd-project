# AITDM – Federated Learning Starter Kit

[![CI](https://github.com/Chiru-CatalinMihail/AITDM_FLOWER_PROJECT_TUTORIAL/actions/workflows/ci.yml/badge.svg)](https://github.com/Chiru-CatalinMihail/AITDM_FLOWER_PROJECT_TUTORIAL/actions/workflows/ci.yml)

Starter code for the **AI for Trustworthy Decision Making** project. It gives you:

1. **A reproducible environment** with one command (Flower, PyTorch, MONAI, ...).
2. **A working federated learning app** (Flower + PyTorch + MONAI) you can run either
   with a single `flwr run` command or "for real" with one terminal per server / client.
3. **A tool to split any dataset in a biased (non-IID) way**, which simulates what
   real federated data looks like (different hospitals, devices, users, annotators).
4. **Trust mechanisms ready to use:** 9 aggregation strategies (robust, fair, adaptive),
   simulated attackers, differential privacy with ε reporting, Grad-CAM and feature importance.
5. **Two data tracks:** medical images (MedMNIST) and Human Activity Recognition (skeleton CSVs).
   Both use the same scripts and the same Flower app.
6. **Reproducible experiments:** every run is seeded, and every number in the tutorials is a
   mean ± std over seeds 1, 2, 3 that `experiments/reproduce_tutorials.sh` re-creates.

Everything is tested with **Flower 1.41**, PyTorch 2.14 and MONAI 1.6 on Python 3.12.

---

## Quick start

```bash
# 0. Get the code
git clone https://github.com/Chiru-CatalinMihail/AITDM_FLOWER_PROJECT_TUTORIAL.git
cd AITDM_FLOWER_PROJECT_TUTORIAL

# 1. Create .venv and install everything (drop --cpu if you have an NVIDIA GPU)
bash scripts/setup_env.sh --cpu
source .venv/bin/activate

# 2. Download a medical dataset (DermaMNIST, skin lesions) and split it into
#    3 biased clients
python data_splitting/split_data.py --dataset dermamnist --num-clients 3 --strategy dirichlet --alpha 0.3

# 3a. Run server + 3 clients with ONE command (simulation)
bash scripts/run_simulation.sh

# 3b. ...or run them as separate processes (deployment), all from one terminal
bash scripts/run_deployment_local.sh

# 3c. ...or open one window per component, and re-run pieces after the first run
bash scripts/run_deployment_terminals.sh
```

You can also start every component by hand in its own terminal; see
[docs/02_running_flower.md](docs/02_running_flower.md).

**HAR instead of images?** Install the zip from Moodle and replace step 2 with:

```bash
python data_splitting/get_data.py --har ~/Downloads/Demo_HAR.zip
python data_splitting/split_data.py --input data/raw/har/train_client*.csv \
    --label-col Action --sample-col IDSample --feature-cols '^J' \
    --strategy group --group-col Camera --num-clients 3 --out data/partitions/har_camera
bash scripts/run_simulation.sh data/partitions/har_camera
```

See [docs/06_har.md](docs/06_har.md).

---

## Data

**No data is stored in this repository** (`data/` is git-ignored). You download what you choose:

| Dataset | What | Source and licence | How |
|---|---|---|---|
| DermaMNIST, PneumoniaMNIST, BloodMNIST, ... | 2D medical images, 28–224 px | [MedMNIST v2](https://medmnist.com/) (Yang et al., *Scientific Data* 2023), public on Zenodo. Licence per dataset, e.g. DermaMNIST CC BY-NC 4.0 (from HAM10000), PneumoniaMNIST CC BY 4.0 | automatic with `split_data.py --dataset <name>`, or in advance with `get_data.py --medmnist <name>` |
| HAR skeletons | 3D skeleton sequences, 5 actions, 3 cameras | subset of [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) (Shahroudy et al., CVPR 2016), academic use after accepting its Release Agreement, **not** redistributable | the course subset from Moodle: `get_data.py --har <zip>`; or build it from the official archive: `get_data.py --ntu <zip>` (choose `--actions`, `--cameras`), see [tutorial 6](docs/06_har.md#1-get-the-data) |

```bash
python data_splitting/get_data.py --list      # what is available / already downloaded
```

Never commit data to your own repository either: removing it later does not remove it from the
git history.

---

## Tutorials

| # | Tutorial | What you learn |
|---|----------|----------------|
| 1 | [Environment setup](docs/01_environment_setup.md) | venv, `requirements.txt`, CPU vs GPU PyTorch, Windows/WSL2 |
| 2 | [Running Flower](docs/02_running_flower.md) | Simulation (`flwr run`) vs. deployment (SuperLink + SuperNodes in separate terminals) |
| 3 | [Biased data splits](docs/03_biased_data_splits.md) | IID, label skew (Dirichlet), pathological, quantity skew, natural (per-subject) splits, feature skew, label noise |
| 4 | [Aggregation strategies](docs/04_aggregation_strategies.md) | FedAvg, FedProx, FedAvgM, FedAdam/Yogi, FedMedian, FedTrimmedAvg, Krum, q-FedAvg, differential privacy; attack experiments |
| 5 | [Grad-CAM](docs/05_gradcam.md) | Explaining predictions, choosing the layer, sanity checks, what to report, and the iterative process: from errors to a better model (worked example) |
| 6 | [Human Activity Recognition](docs/06_har.md) | The HAR skeleton data end to end: splitting, training, trust experiments, feature importance |

---

## Repository structure

```text
AITDM_FLOWER_PROJECT_TUTORIAL/
├── README.md                    <- you are here
├── requirements.txt             <- pinned dependencies (Flower, PyTorch, MONAI, ...)
├── requirements-dev.txt         <- + pytest and ruff (tests / CI)
├── scripts/
│   ├── setup_env.sh             <- creates .venv and installs everything
│   ├── run_simulation.sh        <- server + all clients, one command (Simulation Runtime)
│   ├── run_deployment_local.sh  <- SuperLink + SuperNodes + run, all in one terminal
│   ├── run_deployment_terminals.sh <- one window per component, re-run any of them from a menu
│   ├── _terminal_loop.sh        <- (used by run_deployment_terminals.sh) the menu of each window
│   ├── start_superlink.sh       <- terminal 1: the server side
│   ├── start_supernode.sh       <- terminals 2..N: one client each
│   ├── submit_run.sh            <- last terminal: submit the app (flwr run)
│   └── stop_flower.sh           <- kill all Flower processes (ports in use, stuck runs)
├── data_splitting/
│   ├── get_data.py              <- download MedMNIST / install the HAR zip into data/raw/
│   ├── split_data.py            <- CLI: dataset (MedMNIST / .npz / .csv) -> biased client partitions + plots
│   └── partitioning.py          <- the splitting algorithms (reusable functions)
├── experiments/
│   ├── run_seeds.py             <- one federated experiment with seeds 1, 2, 3 -> mean ± std
│   ├── centralised.py           <- centralised baseline (same model, pooled client data)
│   ├── summarize.py             <- comparison tables of several experiments
│   ├── evaluation.py            <- validation (pooled + per client) and test metrics, shared by all
│   └── reproduce_tutorials.sh   <- re-creates every number in the tutorials
├── explainability/
│   ├── error_analysis.py        <- where does the model fail? per class / client, hardest examples
│   ├── gradcam.py               <- Grad-CAM heat maps (image models)
│   └── feature_importance.py    <- permutation importance (feature models, e.g. HAR)
├── fl_app/                      <- the Flower app (what `flwr run` executes)
│   ├── pyproject.toml           <- app metadata + run configuration (rounds, lr, strategy, ...)
│   └── aitdm_fl/
│       ├── task.py              <- models (MONAI CNN / MLP), data loading, train / evaluate (no Flower here)
│       ├── client_app.py        <- ClientApp: local training & evaluation (+ optional attacker)
│       └── server_app.py        <- ServerApp: aggregation strategy, DP, global evaluation
├── docs/                        <- the tutorials + figures
├── tests/                       <- unit tests, link checker, end-to-end smoke test (run by the CI)
├── .github/workflows/ci.yml     <- GitHub Actions: lint, links, tests, smoke test
├── data/        (git-ignored)   <- raw downloads and generated partitions
├── outputs/     (git-ignored)   <- final_model.pt + metrics.json of every run
└── logs/        (git-ignored)   <- component logs of run_deployment_local.sh
```

---

## How this maps to the project stages

| Project requirement | Where to start |
|---|---|
| *Stage 1 – simulate 3–5 clients with non-IID splits* | `data_splitting/split_data.py` ([tutorial 3](docs/03_biased_data_splits.md)) |
| *Stage 1 – minimal federated experiment (FedAvg)* | `scripts/run_simulation.sh` ([tutorial 2](docs/02_running_flower.md)) |
| *Stage 1 – centralised vs. federated comparison* | `python experiments/centralised.py --partitions data/partitions/<split>`: the same model on the pooled client data, evaluated exactly like the federated runs ([tutorial 2](docs/02_running_flower.md#centralised-baseline-and-seeds)) |
| *Reporting results* | `python experiments/run_seeds.py ...` runs seeds 1, 2, 3; `experiments/summarize.py` prints mean ± std tables. Compare on validation data, report the test set once at the end |
| *Stage 2 – different aggregation types* | `--run-config "strategy='krum'"` etc. ([tutorial 4](docs/04_aggregation_strategies.md)) |
| *Stage 2 – differential privacy* | `--run-config "dp-noise-multiplier=1.0"`, ε is logged and saved ([tutorial 4](docs/04_aggregation_strategies.md#5-privacy-central-differential-privacy)) |
| *Stage 2 – robustness / data poisoning* | `malicious-clients='4'` (model poisoning) or `split_data.py --label-noise` (data poisoning) + robust strategies |
| *Stage 2 – fairness* | `strategy='qfedavg'`, per-client evaluation, quantity-skew splits |
| *Improving the model iteratively* | `explainability/error_analysis.py` + Grad-CAM + `loss-weighting='balanced'` ([worked example](docs/05_gradcam.md#the-iterative-process-from-explanation-to-a-better-model)) |
| *Stage 2 – interpretability* | `python explainability/gradcam.py --run outputs/<run>` ([tutorial 5](docs/05_gradcam.md)); HAR: `explainability/feature_importance.py` ([tutorial 6](docs/06_har.md#5-explain-it)) |
| *Team work – cross-evaluation* | Each member gets a partition (`client_<i>.npz`) and evaluates the others' models on their own validation data; the shared `test.npz` gives the final numbers |

---

## Common troubleshooting

Most Flower problems are **silent waits**, not crashes: a component waits for one that never comes. 

In depth comments in [Tutorial 2](docs/02_running_flower.md#troubleshooting).

| Symptom | Cause | Fix |
|---|---|---|
| On a cluster (SLURM) the server waits forever and the clients find no server | `127.0.0.1` means "this machine", and every job can run on a **different node**: a client looks for the SuperLink on its own node | Run everything in one job on one node (`srun --nodes=1 --cpus-per-task=8 bash scripts/run_deployment_local.sh`). Across nodes: `HOST=0.0.0.0 bash scripts/start_superlink.sh`, then use `<node-name>:8000` (`hostname`) for the SuperNodes and in `~/.flwr/config.toml`. Never train on the login node |
| `Address already in use` (8000, 9094, ...) | An old SuperLink / SuperNode is still alive, or, on a shared node, another user's | `bash scripts/stop_flower.sh`. On shared nodes ask for `--exclusive` |
| `FileNotFoundError: .../test.npz not found` | Split not created (or under another name); relative `partitions-dir`; data not on the node running the component; files re-split / deleted during the run | Run `split_data.py` first and pass its folder; use absolute paths (the scripts do); keep data on the shared filesystem, not `/tmp`; write new splits to a new `--out`. Data is never in the repo: see [Data](#data) |
| `Waiting for nodes to connect: 2 connected (minimum required: 3)` | Fewer SuperNodes than `min-nodes` (simulation defaults to 2 since Flower 1.32) | Start all N SuperNodes; in simulation pass `--federation-config "num-supernodes=N"` (the scripts do) |
| No client connects after a SuperLink restart | The new SuperLink forgets all nodes; old SuperNodes are rejected (`401`) and stop | Restart the SuperNodes (`run_deployment_terminals.sh` does it by itself) |
| After Ctrl+C on `flwr run`, the next run waits | Ctrl+C only stops the log stream; the run keeps the clients busy | `flwr ls local-deployment`, then `flwr stop <run-id> local-deployment` |
| First run "hangs" at `Installing application dependencies...`, or uses old code / another venv | Flower installs GBs of dependencies per run, or a hidden simulation SuperLink from an earlier `flwr run .` is still running | `bash scripts/stop_flower.sh` once; the scripts disable the installation |
| Run "finished" but no `metrics.json` | The ServerApp crashed; `flwr run` still exits with 0 | Read the traceback; the scripts print `The run FAILED` |

---

## Useful links

* Flower docs: <https://flower.ai/docs/framework/>
  * [Quickstart PyTorch](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
  * [Run simulations](https://flower.ai/docs/framework/how-to-run-simulations.html)
  * [Run with the Deployment Engine](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)
  * [Flower Datasets partitioners](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html)
* MONAI: <https://docs.monai.io/>
* MedMNIST (datasets used in the examples): <https://medmnist.com/>
* Federated learning survey: <https://arxiv.org/abs/1907.09693>
* DP book (Dwork & Roth): <https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf>
