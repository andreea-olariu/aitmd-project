# 6. Human Activity Recognition (HAR)

The medical examples use images.

This tutorial runs the **same pipeline on skeleton data**:

split → federated training → trust experiments → explanations.

Only the first step and the explanation tool differ, so read tutorials [2](02_running_flower.md) and [3](03_biased_data_splits.md) first.

---

## 1. Get the data

The HAR data comes from **NTU RGB+D** (Shahroudy et al., CVPR 2016): 3D skeletons of 40
people performing 60 actions, recorded by 3 Kinect v2 cameras. Its
[Release Agreement](https://rose1.ntu.edu.sg/dataset/actionRecognition/) allows academic,
non-commercial research only and forbids redistribution, so the data is **not** part of this
repository, and it must never end up in yours. There are two ways to get it.

**A. The course subset from Moodle** (quickest):

```bash
source .venv/bin/activate
python data_splitting/get_data.py --har ~/Downloads/Demo_HAR.zip    # the file from Moodle
```

**B. Build it yourself from the official archive.** Register at the
[ROSE Lab](https://rose1.ntu.edu.sg/dataset/actionRecognition/), accept the Release Agreement and
download the 3D skeletons of NTU RGB+D 60 (`nturgbd_skeletons_s001_to_s017.zip`, ~5.8 GB). The
script never downloads the dataset; it converts the archive you have:

```bash
# the 5 course actions, all 3 cameras: the same files as on Moodle
python data_splitting/get_data.py --ntu ~/Downloads/nturgbd_skeletons_s001_to_s017.zip

# or choose your own actions (label = position) and cameras
python data_splitting/get_data.py --ntu <zip> --actions A008 A009 A023 A027 A043 --cameras 1 2 --out data/raw/har_custom
```

| Option | Default | Meaning |
|---|---|---|
| `--actions` | `course` | `course` = the 5 actions below, or any NTU ids `A001` ... `A060` |
| `--cameras` | `1 2 3` | which of the 3 cameras to keep |
| `--bodies` | `main` | `main` = only the actor's skeleton; `all` = every detected body, as in the Moodle files |
| `--out` | `data/raw/har` | output folder (`--force` replaces existing files) |

It uses the standard NTU **cross-subject** split: the training performers go to
`train_client1/2/3.csv` (a random third each, as on Moodle), and the others to `test.csv`, which
here **has** labels. It skips the samples the NTU authors list as having missing skeletons, and
also writes the columns `Subject`, `Setup` and `NTUAction`. `Subject` makes a real one-client-per-person
split possible (`--group-col Subject`, [tutorial 3](03_biased_data_splits.md)).

Either way you get:

```text
data/raw/har/
├── train_client1.csv
├── train_client2.csv
└── train_client3.csv
```

> `data/` is git-ignored: **never commit data** to your GitHub repository, also not to a
> private one. Deleting it later does not remove it from the git history.

### The 5 course actions

| Label | NTU action |
|---|---|
| 0 | A044 touch head (headache) |
| 1 | A027 jump up |
| 2 | A015 take off jacket |
| 3 | A026 hopping (one foot jumping) |
| 4 | A043 falling |

The Moodle files only contain the labels 0–4. The mapping was **inferred** from two independent
pieces of evidence. First, the number of sequences per label: NTU has exactly 672 training and
276 test samples per action, minus the samples the authors list as having missing skeletons.
The Moodle counts per label leave a handful of candidate actions each, and together the 5 labels
must also explain the 2 sequences missing from the Moodle test set. Second, the motion decides
between the candidates, e.g. label 4 ends with the head 85 cm lower, in label 1 both feet leave
the ground, in label 3 only one does, and in label 0 the right hand rests at the head. With the official archive
you can confirm it exactly, by looking up every Moodle sequence's coordinates in the archive:

```bash
python data_splitting/get_data.py --ntu <zip> --match data/raw/har
```

> **Data quality:** in 138 of the 3 354 Moodle training sequences, Kinect detected a second
> person or a "ghost", and both skeletons were kept, as two rows per frame. The
> per-sequence statistics below then mix two bodies. Option B keeps only the actor
> (`--bodies main`). Worth a sentence in your report if you use the Moodle files.

Each row is **one frame** of a skeleton sequence:

| Column | Meaning |
|---|---|
| `IDSample` | id of the sequence: all rows with the same id form **one** sample |
| `FrameNumber` | position of the frame in the sequence |
| `J1X, J1Y, J1Z ... J25X, J25Y, J25Z` | 3D position of the 25 skeleton joints (metres) |
| `Action` | the class: 5 actions, labelled `0` ... `4` ([which ones](#the-5-course-actions)) |
| `Camera` | which camera (`1`, `2`, `3`) recorded the sequence |

Together the three files contain 3 354 sequences of 30–516 frames (about 75 on average). The classes are balanced, and each file has a random third of the data (same mix of actions and cameras).

---

## 2. Split it

`split_data.py` reads several CSV files at once (they are concatenated, and each row remembers its file in a new column `source_file`):

```bash
source .venv/bin/activate

# Natural split: one client per camera (3 clients)
python data_splitting/split_data.py --input data/raw/har/train_client*.csv \
    --label-col Action --sample-col IDSample --feature-cols '^J' \
    --strategy group --group-col Camera --num-clients 3 --out data/partitions/har_camera
```

* `--label-col Action`: the class.
* `--sample-col IDSample`: the frames of one sequence stay together (in the same client, and in the same train / val / test part).
* `--feature-cols '^J'`: a regular expression: the joint columns are the features. Without it, every numeric column would be used, including `FrameNumber` and `Camera`.

Other useful splits:

```bash
# The split as shipped: train_client1/2/3.csv are one client each (nearly IID, a good baseline)
... --strategy group --group-col source_file --num-clients 3 --out data/partitions/har_files

# Label skew: 5 clients that each see mostly some actions
... --strategy dirichlet --alpha 0.3 --num-clients 5 --out data/partitions/har_dirichlet

# Every camera differently calibrated (gain / offset / noise per coordinate)
... --strategy group --group-col Camera --num-clients 3 --feature-skew --out data/partitions/har_camera_skew

# Give the actions readable names in plots and reports (in the order 0, 1, 2, ...)
... --class-names "touch head,jump up,take off jacket,hopping,falling"
```

(`...` = `python data_splitting/split_data.py --input data/raw/har/train_client*.csv --label-col Action --sample-col IDSample --feature-cols '^J'`)

![HAR, Dirichlet split](images/split_har_dirichlet.png)

`--label-noise` and all other options from [tutorial 3](03_biased_data_splits.md) work too.

### From sequences to feature vectors

Sequences have different lengths, but the model needs inputs of a fixed size. For every sequence, `split_data.py` computes four statistics of every joint coordinate over time:

```text
mean, std, min, max   x   75 columns (J1X ... J25Z)   =   300 features per sample
```

These simple statistics already separate the 5 actions well (see the results in
[step 3](#3-train-it)), so they make a good starting point. [Section 6](#6-going-further) indicates towards richer directions in which you could take your project.

---

## 3. Train it

Nothing changes: the Flower app sees 2D arrays instead of images and builds an **MLP** (300 inputs → 256 → 128 → 5 actions, with dropout) instead of the CNN used for moles classification. See `build_model()` in `fl_app/aitdm_fl/task.py`.

```bash
# Simulation
bash scripts/run_simulation.sh data/partitions/har_camera "num-server-rounds=10 local-epochs=3"

# Deployment (SuperLink + SuperNodes as separate processes)
bash scripts/run_deployment_local.sh data/partitions/har_camera "num-server-rounds=10 local-epochs=3"
```

In separate terminals it is also the same as before:
`bash scripts/start_supernode.sh 0 data/partitions/har_camera` and so on.

Results (10 rounds, 3 local epochs; mean ± std over seeds 1, 2, 3;
`bash experiments/reproduce_tutorials.sh t6` re-creates them):

| Setup | Val acc. (all clients) | Worst client val acc. | Test acc. |
|---|---|---|---|
| Centralised baseline (all training data, 30 epochs) | 0.936 ± 0.012 | 0.922 ± 0.024 | 0.949 ± 0.009 |
| FL, one client per camera | 0.927 ± 0.017 | 0.903 ± 0.026 | 0.934 ± 0.010 |
| FL, one client per file + `--feature-skew` | 0.940 ± 0.009 | 0.926 ± 0.012 | 0.943 ± 0.002 |
| FL, Dirichlet α = 0.3, 3 clients | 0.854 ± 0.018 | 0.764 ± 0.024 | 0.870 ± 0.013 |

The classes are balanced, so accuracy and balanced accuracy are almost the same here. Label skew
costs the most, as for the medical images, and it hits one client hardest (worst client 0.76).
Camera and file splits are close to centralised training: the files are random thirds, and the
statistics of a movement change little between camera angles.

### Centralised baseline

Stage 1 asks for a centralised vs. federated comparison. `experiments/centralised.py` trains the
same model on the union of the clients' training data and evaluates it exactly like the federated
runs (pooled validation, per client, and test):

```bash
python experiments/centralised.py --partitions data/partitions/har_camera --epochs 30
```

### A camera no client has seen

With the split above, every camera is also in the test set. To measure whether the model works for a
**new** camera, hold camera 3 out completely (`--test-groups`, see
[tutorial 3](03_biased_data_splits.md#test-on-a-group-that-no-client-has-seen)): cameras 1 and 2
become the two clients, and the 1117 sequences of camera 3 are the test set.

```bash
python data_splitting/split_data.py --input data/raw/har/train_client*.csv \
    --label-col Action --sample-col IDSample --feature-cols '^J' \
    --strategy group --group-col Camera --test-groups 3 --num-clients 2 --out data/partitions/har_camera_test3
bash scripts/run_simulation.sh data/partitions/har_camera_test3 "num-server-rounds=10 local-epochs=3"
```

| Model | Val acc. (cameras 1, 2) | Test acc. (camera 3, unseen) |
|---|---|---|
| FL, 2 clients (camera 1, camera 2) | 0.960 ± 0.007 | 0.908 ± 0.004 |
| Centralised | 0.950 ± 0.027 | 0.906 ± 0.017 |

The validation data (same cameras) promises 0.96; on the unseen camera the model reaches 0.91.
That gap of about 5 points is the **cross-device generalisation** cost, and the random test set
of the first split cannot show it. Federated and centralised training pay it alike.

---

## 4. Trust experiments

All options from [tutorial 4](04_aggregation_strategies.md) work unchanged. Same setup as
[above](#3-train-it) (camera split, 10 rounds, 3 local epochs, mean ± std over 3 seeds):

| Experiment | Val acc. | Test acc. |
|---|---|---|
| No attacker, FedAvg | 0.927 ± 0.017 | 0.934 ± 0.010 |
| Client 2 sends sign-flipped weights, FedAvg | 0.187 ± 0.008 | 0.199 ± 0.009 (chance: 0.20) |
| ... FedMedian | 0.902 ± 0.028 | 0.896 ± 0.022 |
| ... Krum | 0.925 ± 0.016 | 0.925 ± 0.003 |
| FedProx | 0.919 ± 0.008 | 0.928 ± 0.002 |
| Differential privacy, $\sigma = 0.5$ ($\varepsilon \approx 48.8$, $\delta = 10^{-5}$) | 0.224 ± 0.040 | 0.223 ± 0.034 |

```bash
bash scripts/run_simulation.sh data/partitions/har_camera \
    "num-server-rounds=10 local-epochs=3 strategy='krum' malicious-clients='2'"
```

> With 3 clients, Krum's guarantee (K > 2f + 2) does **not** hold, even if it happens to work here.
> For a robustness experiment you can defend in the report, use 5+ clients (e.g. the Dirichlet
> split with `--num-clients 5`).

---

## 5. Explain it

Grad-CAM needs images, so for feature data use **permutation importance**: shuffle one feature (or a group of features) between the samples, which breaks its link with the label, and measure how much the accuracy drops. The bigger the drop, the more the model relies on it.

```bash
# Which joints matter? (all statistics of a joint are shuffled together)
python explainability/feature_importance.py --run outputs/<run>

# Which statistics matter? (mean / std / min / max)
python explainability/feature_importance.py --run outputs/<run> --group-by stat

# On one client's validation data
python explainability/feature_importance.py --run outputs/<run> --data client_0

# Sanity check: an untrained model
python explainability/feature_importance.py --run outputs/<run> --untrained
```

| Option | Meaning |
|---|---|
| `--group-by joint` (default) | J5 = mean/std/min/max of J5X, J5Y, J5Z (12 features) |
| `--group-by column` | J5X = its 4 statistics |
| `--group-by stat` | `mean`, `std`, `min`, `max` (75 features each) |
| `--group-by feature` | each of the 300 features alone |
| `--repeats 5` | shuffles per group, averaged (the std across them is printed and saved too) |

The script saves a PNG and a JSON (all values) in the run folder. With `--group-by joint`, the plot also shows the **mean pose** of the data, with each joint coloured by its importance:

![Joint importance](images/har_importance_joint.png)

What the camera-split model shows (validation data, mean ± std over the 3 seeds' models):

| Shuffled group | Accuracy drop (trained) | Accuracy drop (untrained) |
|---|---|---|
| `max` of all joints | 0.439 ± 0.017 | −0.009 ± 0.015 |
| `min` of all joints | 0.422 ± 0.023 | 0.000 ± 0.001 |
| `std` of all joints | 0.158 ± 0.010 | −0.001 ± 0.001 |
| `mean` of all joints | 0.027 ± 0.004 | 0.001 ± 0.002 |
| the most important single joint | 0.017 ± 0.006 (J24, J17, J1: a different one per seed) | 0.005 ± 0.008 |

![Statistic importance](images/har_importance_stat.png)

How to read this (and what to write in the report):

* **The range of a movement (min / max) matters, the average position barely.** That makes sense for actions: they differ in how far the body moves, not in where it stands.
* **Every single joint looks unimportant** (≤ 0.02) **although the joints together are crucial,
  and the "most important joint" changes from seed to seed.** A ranking of single joints is
  therefore not a finding. The joints are strongly correlated: if one is shuffled, its neighbours still carry the information. Permutation importance **underestimates correlated features**, so group them (`--group-by stat` or `joint`) before you conclude that something does not matter.
* For trust: does the importance differ between clients (`--data client_<i>`)? Does an attacked or
  DP-trained model rely on different joints?

The joints follow the Kinect v2 layout of NTU RGB+D: J1 spine base, J4 head, J8 / J12 left / right
hand, J15 / J19 left / right ankle, J24 / J25 right hand tip / thumb (see the mean pose above).

`explainability/error_analysis.py` (per-class recall, confusion matrix, hardest examples) also works on HAR runs. See [the iterative process](05_gradcam.md#the-iterative-process-from-explanation-to-a-better-model).


---

## 6. Going further

| I want to... | Where |
|---|---|
| change the MLP | `build_mlp()` in `fl_app/aitdm_fl/task.py` |
| use other per-sequence features (velocities, joint angles, ...) | `sample_features()` in `data_splitting/split_data.py` |
| use a sequence model (LSTM / GRU / 1D CNN / transformer) on the raw frames | read `client_<i>_train.csv` / `client_<i>_val.csv` / `test.csv` and adapt `load_npz_arrays()` / `load_client_data()` / `load_test_data()` in `task.py`; the Flower code stays the same |
| simulate other device differences | `apply_sensor_skew()` in `data_splitting/partitioning.py` |
