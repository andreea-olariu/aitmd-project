# 1. Environment setup

Every project ends up with dependency conflicts sooner or later (one library wants
`numpy<2`, another `numpy>=2`...). To avoid breaking your system Python, install
everything in a **virtual environment** (venv) that lives in the project folder.

## Requirements

* **Linux or macOS.** On **Windows, use WSL2**: open PowerShell as administrator, run
  `wsl --install`, reboot, open the "Ubuntu" app and follow this guide inside it.
* **Python 3.11, 3.12 or 3.13** (Flower needs ≥ 3.11). Check with `python3 --version`.
  * Ubuntu: `sudo apt install python3.12 python3.12-venv`
  * macOS: `brew install python@3.12`
* About 2 GB of free disk space for the CPU build, or 6 GB or more for the CUDA build.

## One-command install

From the repository root:

```bash
bash scripts/setup_env.sh --cpu      # CPU-only PyTorch: small download, works everywhere
# or
bash scripts/setup_env.sh            # default PyTorch build (includes CUDA on Linux)
```

If `python3` is the wrong version, choose the interpreter explicitly:

```bash
PYTHON=python3.12 bash scripts/setup_env.sh --cpu
```

The script:

1. checks the Python version,
2. creates `.venv/` in the repository,
3. installs PyTorch (CPU or default build),
4. installs everything in [`requirements.txt`](../requirements.txt),
5. installs the Flower app in `fl_app/` in editable mode (`pip install -e fl_app`), so that
   `import aitdm_fl` works anywhere (handy for your centralised baseline or a notebook),
6. prints the versions it installed:

```text
=== Environment ready ===
flwr        1.41.0
torch       2.14.1+cpu  (CUDA available: False)
torchvision 0.29.1+cpu
monai       1.6.1
numpy       2.5.2
```

## Using the environment

Activate it in **every new terminal**:

```bash
source .venv/bin/activate        # your prompt now starts with (.venv)
deactivate                       # leave it
```

The `scripts/*.sh` files activate `.venv` for you, so you can run them directly.

## Where the code you will edit lives

The Flower app is the Python package `aitdm_fl`, inside the `fl_app/` folder of the repository.
These are the files you will change during the project (paths from the repository root,
`AITDM_FLOWER_PROJECT_TUTORIAL/`):

| File | Full path | What is in it |
|---|---|---|
| `task.py` | `AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/aitdm_fl/task.py` | The model (`build_model()`, `build_mlp()`), data loading, `train()` and `evaluate()`. It contains no Flower code, so you can reuse it for your centralised baseline |
| `server_app.py` | `AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/aitdm_fl/server_app.py` | The **server**: aggregation strategy (`build_strategy()`), differential privacy, global evaluation, and saving `final_model.pt` + `metrics.json` |
| `client_app.py` | `AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/aitdm_fl/client_app.py` | The **client**: local training / evaluation on one partition, plus the optional attacker |
| `pyproject.toml` | `AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/pyproject.toml` | App settings and the default run configuration (rounds, learning rate, strategy, ...) |

To print the **absolute** paths on your machine (they depend on where you cloned the
repository), run this with the venv activated:

```bash
python -c "import aitdm_fl.task as t, aitdm_fl.server_app as s; print(t.__file__); print(s.__file__)"
```

```text
/home/<you>/AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/aitdm_fl/task.py
/home/<you>/AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/aitdm_fl/server_app.py
```

Because the app is installed in editable mode ([step 5 above](#one-command-install)), Python always uses these files
directly. Your edits take effect at the next run, without reinstalling. If the command prints a
path inside `.venv/lib/.../site-packages/` instead, the app was not installed in editable mode:
run `pip install -e fl_app` from the repository root.

## What is installed and why

| Package | Why |
|---|---|
| `flwr[simulation,dp]` | Flower framework; `[simulation]` adds Ray for the Simulation Runtime, `[dp]` adds `dp-accounting` (privacy budget ε) |
| `flwr-datasets[vision]` | Optional: Flower's own dataset partitioners (Hugging Face datasets) |
| `torch`, `torchvision` | Deep learning |
| `monai` | Medical imaging networks, transforms and losses |
| `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm` | Data handling, metrics (F1, balanced accuracy), plots |
| `grad-cam` | Grad-CAM / Grad-CAM++ / Score-CAM explanations for Stage 2 (pulls in OpenCV), see [tutorial 5](05_gradcam.md) |

Need another library? Add it to `requirements.txt` **with a version** and re-run
`bash scripts/setup_env.sh`. The script reuses the existing `.venv`.

## Doing it by hand (what the script does)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu   # CPU only
pip install -r requirements.txt
pip install -e fl_app
```

## Alternatives

Any environment manager works as long as you install `requirements.txt` into it:
[uv](https://docs.astral.sh/uv/) (`uv venv && uv pip install -r requirements.txt`), conda or pyenv.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: Python 3.11-3.13 required` | Install a supported Python and use `PYTHON=python3.12 bash scripts/setup_env.sh` |
| `The virtual environment was not created successfully because ensurepip is not available` | `sudo apt install python3.12-venv` (match your Python version) |
| Very large download / disk full | Use `--cpu`; delete `.venv/` and start again |
| `CUDA available: False` but you have a GPU | Re-create the venv without `--cpu`; check `nvidia-smi` works |
| University cluster (FEP) | Run the server and all clients **on the same node and queue**, or use the node's real IP instead of `127.0.0.1`.<br><br>Otherwise, pointing to 0.0.0.0:8080 or 127.0.0.1:8080 is going to point to different actual IPs and this will mean that they are not communicating (server is waiting for the clients and clients give an error of "no server to talk to") |
