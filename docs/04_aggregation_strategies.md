# 4. Aggregation strategies

After every round the server receives one model update per client and has to turn them into **one** global model.

That step is the *aggregation strategy*. It is the server's main lever
for the trust dimensions in the project:

| Goal | Problem | Strategies in this repo |
|---|---|---|
| **Accuracy under non-IID data** | Clients pull the model in different directions (*client drift*) | `fedprox`, `fedavgm`, `fedadam`, `fedyogi` |
| **Robustness** | One faulty or malicious client can wreck the average | `fedmedian`, `fedtrimmedavg`, `krum` |
| **Fairness** | The global model serves big or majority clients and neglects small ones | `qfedavg` |
| **Privacy** | The updates themselves can leak training data | DP wrapper (`dp-noise-multiplier > 0`) |

Every strategy is chosen at run time. **No code changes are needed:**

```bash
bash scripts/run_simulation.sh data/partitions/dirichlet "strategy='krum'"
```

The strategies come from `flwr.serverapp.strategy`. They are created in `build_strategy()` in
[`fl_app/aitdm_fl/server_app.py`](../fl_app/aitdm_fl/server_app.py), and their
hyperparameters live in [`fl_app/pyproject.toml`](../fl_app/pyproject.toml).

---

## Notation

$K$ clients participate. In round $t$:

* $w_t$ is the global model,
* client $k$ has $n_k$ samples, with $n = \sum_k n_k$ in total, and returns its locally trained model $w_t^k$,
* its *update* is $\Delta^k = w_t^k - w_t$.

---

## 1. FedAvg: the baseline

$$
w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \, w_t^k
$$

Weighted average of the client models, weighted by dataset size (the `num-examples` our ClientApp reports).

It is simple, communication-efficient (McMahan et al., 2017) and works well on IID data.

**Weak spots:** strong non-IID data (client drift), a single malicious client can move the average arbitrarily far ([see the experiment below](#experiment-1-attacker-out-of-5-clients)), and large clients dominate.

```bash
--run-config "strategy='fedavg'"
```

## 2. Strategies for non-IID data

### FedProx (`fedprox`)

Each client minimises its loss **plus** a penalty for moving away from the global model:

$$
\min_{w} \; L_k(w) + \frac{\mu}{2} \lVert w - w_t \rVert^2
$$

The server side is identical to FedAvg. Flower only sends `proximal-mu` to the clients, and **our `train()` in `task.py` adds the penalty** (look for `proximal_mu`). $\mu = 0$ is FedAvg; a larger $\mu$ means less drift but slower local progress. (Li et al., 2020).

```bash
--run-config "strategy='fedprox' proximal-mu=0.1"
```

### FedAvgM (`fedavgm`)

Applies **momentum on the server**: the averaged update is accumulated over rounds like SGD momentum, which smooths the zig-zag caused by heterogeneous clients. (Hsu et al., 2019)

```bash
--run-config "strategy='fedavgm' server-momentum=0.9"
```

### FedAdam / FedYogi (`fedadam`, `fedyogi`): adaptive federated optimisation

Treats the averaged update as a "pseudo-gradient" and applies **Adam** (or Yogi, a more stable Adam variant) on the server, with its own learning rate $\eta$ = `server-lr`.

This often helps on heterogeneous data, but adds a hyperparameter to tune. (Reddi et al., 2021 - ("FedOpt")).

```bash
--run-config "strategy='fedadam' server-lr=0.01"
```

## 3. Robust (Byzantine-tolerant) aggregation

The mean has a *breakdown point* of 0: one extreme value can move it anywhere. These strategies replace the mean with statistics that ignore outliers.

| Strategy | Rule | Tolerates | Paper |
|---|---|---|---|
| `fedmedian` | Coordinate-wise **median** of the client models | < 50 % attackers | Yin et al., 2018 |
| `fedtrimmedavg` | Per coordinate, drop the `trim-beta` largest and smallest values, average the rest | `trim-beta` fraction on each side | Yin et al., 2018 |
| `krum` | Pick the **single** client model closest to its $K - f - 2$ nearest neighbours | $f$ = `num-malicious`, needs $K > 2f + 2$ | Blanchard et al., 2017 |

Flower also has `MultiKrum` and `Bulyan` (Bulyan needs $K \geq 4f + 3$).

**The price:** robust rules throw information away. With honest but non-IID clients, an "outlier" may just be a hospital with rare patients, so robustness can hurt **fairness**.

This trade-off is one of the key points worth researching in your reports and experiments.

### Simulating attackers

The ClientApp can turn selected clients into attackers (`poison()` in `client_app.py`):

| `attack` | What the malicious client sends |
|---|---|
| `sign-flip` | $w_t - 5 \, \Delta^k$ (pushes the model 5× in the wrong direction) |
| `noise` | random Gaussian weights |
| `none` | its honestly trained weights (combine with a wrong-label client or `attack-inflate-examples`, see [below](#attackers-can-also-lie-about-their-size)) |

```bash
# Simulation: partition ids of the attackers
bash scripts/run_simulation.sh data/partitions/iid5 "malicious-clients='4' attack='sign-flip' strategy='krum'"

# Deployment: flag a single SuperNode
flower-supernode --insecure --superlink 127.0.0.1:8000 --port 9098 \
    --node-config "data-path='/abs/path/client_4.npz' malicious=true"
```

For *data* poisoning (wrong labels instead of wrong weights), use
`split_data.py --label-noise` ([tutorial 3](03_biased_data_splits.md)).

### Experiment: 1 attacker out of 5 clients

DermaMNIST, IID split over 5 clients (`--num-clients 5 --strategy iid`), 5 rounds, client 4 is
malicious. Mean ± std over seeds 1, 2, 3 (`bash experiments/reproduce_tutorials.sh t4`):

| Attack | Strategy | Val bal. acc. | Test acc. | Test bal. acc. | Test macro-F1 |
|---|---|---|---|---|---|
| none | FedAvg | 0.312 ± 0.041 | 0.726 ± 0.005 | 0.334 ± 0.031 | 0.352 ± 0.028 |
| none | Krum ($f = 1$) | 0.300 ± 0.008 | 0.707 ± 0.008 | 0.315 ± 0.022 | 0.347 ± 0.013 |
| sign-flip | FedAvg | 0.157 ± 0.017 | **0.034 ± 0.001** | 0.149 ± 0.009 | 0.014 ± 0.002 |
| sign-flip | FedMedian | 0.309 ± 0.006 | 0.717 ± 0.004 | 0.332 ± 0.018 | 0.356 ± 0.017 |
| sign-flip | FedTrimmedAvg ($\beta = 0.2$) | 0.301 ± 0.009 | 0.717 ± 0.008 | 0.324 ± 0.029 | 0.349 ± 0.029 |
| sign-flip | Krum ($f = 1$) | 0.321 ± 0.030 | 0.714 ± 0.001 | 0.343 ± 0.027 | 0.354 ± 0.006 |
| noise | FedAvg | 0.130 ± 0.013 | 0.136 ± 0.089 | 0.138 ± 0.018 | 0.056 ± 0.033 |
| noise | FedMedian | 0.303 ± 0.020 | 0.725 ± 0.003 | 0.338 ± 0.014 | 0.363 ± 0.003 |
| noise | Krum ($f = 1$) | 0.321 ± 0.030 | 0.714 ± 0.001 | 0.343 ± 0.027 | 0.354 ± 0.006 |

* **One attacker out of five destroys FedAvg**, on every metric.
* **The robust rules keep both accuracy and balanced accuracy at the attack-free level**: every
  difference to the first row is smaller than the spread between seeds.
* **Krum has the same numbers under both attacks.** That is not a coincidence: Krum discards the
  attacker in every round and then *selects* one honest client's model. That choice no longer
  depends on what the attacker sent, so with the same seed the result is identical.
* **Is Krum with an attacker "better than no attack"?** Its balanced accuracy (0.343) is higher than
  clean FedAvg (0.334), but the difference is far smaller than the seed-to-seed spread (± 0.03): it
  is noise, not an effect. A single run can easily suggest such a "finding", which is why every
  number here is a mean over seeds. Krum pays a small, consistent price in accuracy instead
  (0.707–0.714 vs. 0.726): it keeps **one** client's model per round, trained on a fifth of the
  data, instead of averaging all five.

### Attackers can also lie about their size

FedAvg weights every client by the `num-examples` *the client itself reports*. Nobody checks it.
A data-poisoning client (here: client 4 with 100 % wrong labels, `--label-noise 0,0,0,0,1.0`,
sending its honestly trained weights) is diluted by four honest clients, unless it claims to have
100× more data:

```bash
python data_splitting/split_data.py --dataset dermamnist --num-clients 5 --strategy iid \
    --label-noise 0,0,0,0,1.0 --out data/partitions/iid5_poison4
bash scripts/run_simulation.sh data/partitions/iid5_poison4 \
    "malicious-clients='4' attack='none' attack-inflate-examples=100"
```

| Client 4 | Strategy | Val bal. acc. | Test acc. | Test bal. acc. |
|---|---|---|---|---|
| honest (no poisoning) | FedAvg | 0.312 ± 0.041 | 0.726 ± 0.005 | 0.334 ± 0.031 |
| 100 % wrong labels, reports its true size | FedAvg | 0.371 ± 0.012 | 0.706 ± 0.010 | 0.378 ± 0.002 |
| 100 % wrong labels, reports **100× its size** | FedAvg | 0.101 ± 0.020 | **0.055 ± 0.016** | 0.117 ± 0.009 |
| 100 % wrong labels, reports 100× its size | FedMedian | 0.354 ± 0.023 | 0.722 ± 0.006 | 0.366 ± 0.016 |

* **Honest about its size, the poisoner is diluted** by the four honest clients: accuracy drops by 2
  points. (Its balanced accuracy even rises a little, because the wrong labels push the model away
  from always predicting the majority class. That is not a defence!)
* **Lying about its size, the same client destroys FedAvg**, without sending a single unusual weight:
  100× the size means ~96 % of the weight in every average.
* **FedMedian (and Krum, FedTrimmedAvg) ignore the reported sizes**, so the lie has no effect: with
  the noise attack, FedMedian gives *exactly* the same numbers with and without inflation.

Lesson: in FedAvg, the weights are part of the attack surface. A server should not trust
self-reported sizes (cap them, check them against what is plausible, or use equal weights, as
DP-FedAvg does).


## 4. Fairness: q-FedAvg (`qfedavg`)

Clients with a **higher loss get a larger weight**, proportional to $F_k^q$, where $F_k$ is the loss of client $k$.

With $q = 0$ this is FedAvg; a larger $q$ pushes towards equal performance across clients (a more uniform accuracy distribution) at the cost of some average accuracy. (Li et al., 2020 - ("q-FFL")).

```bash
--run-config "strategy='qfedavg' qfedavg-q=1.0"
```

`qfedavg-client-lr` (default 1.0) sets the step size: Flower estimates the Lipschitz constant as $L = 1 / \text{lr}$, with $\text{lr}$ = `qfedavg-client-lr`. With a too-small value (e.g. our Adam learning rate 0.001) the global model barely moves.

To **measure** fairness, compare per-client metrics: evaluate the final model on every `client_<i>.npz` and report the worst client, or the standard deviation across clients, next to the mean.

The client-side evaluation metrics in `metrics.json` are averaged over clients, so they hide exactly this.

## 5. Privacy: central differential privacy

Setting `dp-noise-multiplier > 0` wraps **any** strategy in
`DifferentialPrivacyServerSideFixedClipping`:

1. each client update $\Delta^k$ is clipped to L2 norm $\lVert \Delta^k \rVert_2 \leq C$, with $C$ = `dp-clipping-norm`,
2. Gaussian noise with standard deviation $\frac{\sigma C}{K}$ is added to the aggregate, with $\sigma$ = `dp-noise-multiplier`,
3. the clients are averaged with **equal** weights instead of by dataset size, as in DP-FedAvg.
   Otherwise one large client could dominate the average and the privacy analysis would not hold.

```bash
--run-config "dp-noise-multiplier=1.0 dp-clipping-norm=1.0"
--run-config "strategy='fedmedian' dp-noise-multiplier=1.0"   # DP + robustness
```

**The privacy budget $\varepsilon$ is computed for you.** For FedAvg-style strategies (`fedavg`, `fedprox`), the server attaches Flower's Rényi-DP accountant (needs the `flwr[dp]` extra, which is already in `requirements.txt`). It prints the cumulative budget after every round and stores it in `metrics.json` under `"privacy"`:

```text
INFO :      Privacy spent: epsilon=12.301691 at delta=1e-05 after 5 releases
```

The guarantee is **client-level**: it protects whether a whole client (hospital) took part, not just a single record. $\delta$ is set with `dp-delta`.

Other strategies still get the noise, but without accounting; a `[DP]` message tells you so.

Same setup as [above](#experiment-1-attacker-out-of-5-clients) (5 IID clients, 5 rounds, $C = 1.0$, $\delta = 10^{-5}$):

| $\sigma$ (noise multiplier) | $\varepsilon$ after 5 rounds | Val bal. acc. | Test acc. | Test macro-F1 |
|---|---|---|---|---|
| 0 (no DP) | $\infty$ | 0.312 ± 0.041 | 0.726 ± 0.005 | 0.352 ± 0.028 |
| 0.1 | 354.9 | 0.274 ± 0.014 | 0.683 ± 0.019 | 0.282 ± 0.011 |
| 0.5 | 30.1 | 0.140 ± 0.006 | 0.261 ± 0.266 | 0.084 ± 0.041 |
| 1.0 | 12.3 | 0.153 ± 0.024 | 0.245 ± 0.341 | 0.068 ± 0.051 |
| 2.0 | 5.4 | 0.147 ± 0.005 | 0.237 ± 0.363 | 0.057 ± 0.054 |

(Mean ± std over seeds 1, 2, 3. $\varepsilon$ is the same for every seed: it depends only on
$\sigma$, the number of rounds and of clients, not on the data. The huge spread of the test
accuracy for $\sigma \geq 0.5$ has a simple cause: the noise pushes the model to predict mostly
one class (39–98 % of all images), and *which* class depends on the seed. With seed 2 it is the
majority class (nevi), so accuracy looks fine (0.57–0.66); with seeds 1 and 3 it is a rare class
(0.02–0.11). Balanced accuracy and macro-F1 show that all of them are useless.)

**The privacy–utility trade-off is brutal with few clients.** The only setting that keeps a useful model has $\varepsilon \approx 355$, which promises essentially nothing; $1 \leq \varepsilon \leq 10$ (the range usually called meaningful) destroys the model.

The noise is divided by $K = 5$ only. Real DP federated learning averages over hundreds or thousands of clients per round, and $\varepsilon$ also grows with every extra round.

**Ideas for the report:**
* Plot accuracy against $\varepsilon$
* Vary the clipping norm
* Explain *why* 5 clients cannot reach a useful $\varepsilon$
* Compare with **local DP** (DP-SGD on every client, e.g. with
[Opacus](https://opacus.ai/), which protects individual patients) or Flower's client-side variants (`DifferentialPrivacyClientSideFixedClipping`, adaptive clipping).  See the [Flower DP guide](https://flower.ai/docs/framework/explanation-differential-privacy.html).

---

## Choosing a strategy (cheat sheet)

| Situation | Try first |
|---|---|
| IID or mildly non-IID, honest clients | `fedavg` |
| Strong label skew / pathological split | `fedprox`, `fedavgm`, `fedadam` **(Tunable!)** |
| Suspected faulty or malicious clients, noisy labels | `fedtrimmedavg`, `fedmedian`, `krum` |
| Small clients are neglected | `qfedavg` |
| Must hide which clients took part / protect records | DP wrapper ($\varepsilon$ is reported); local DP-SGD for record-level privacy |

**Mandatory baselines**: Always compare against **FedAvg** and against a **centralised** model trained on the union of all clients' data. Run several seeds, and report balanced accuracy / macro-F1, not only accuracy.

---

## Writing your own strategy

Subclass an existing strategy and override the step you want to change.

This example logs every client's training loss before the normal FedAvg aggregation:

```python
from flwr.serverapp.strategy import FedAvg

class FedAvgWithLogging(FedAvg):
    """FedAvg that prints every client's training loss before aggregating."""

    def aggregate_train(self, server_round, replies):
        replies = list(replies)  # we iterate twice
        for msg in replies:
            if msg.has_content():
                loss = msg.content["metrics"]["train_loss"]
                print(f"round {server_round} | node {msg.metadata.src_node_id} | train_loss={loss:.3f}")
        return super().aggregate_train(server_round, replies)
```

Put it in `server_app.py` and return it from `build_strategy()`. Useful methods to override:

| Method | Runs | Typical use |
|---|---|---|
| `configure_train` | before each round | choose clients, send per-round config (lr schedule, ...) |
| `aggregate_train` | after training | your own aggregation rule, detect outliers, reputation scores |
| `configure_evaluate` / `aggregate_evaluate` | evaluation phase | custom client-side metrics |

Look at the source of the built-in strategies for inspiration
(`python -c "import inspect, flwr.serverapp.strategy as s; print(inspect.getsource(s.FedTrimmedAvg))"`).

## References

* [McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg), AISTATS 2017](https://arxiv.org/abs/1602.05629)
* [Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys 2020](https://arxiv.org/abs/1812.06127)
* [Hsu et al., *Measuring the Effects of Non-Identical Data Distribution* (FedAvgM), 2019](https://arxiv.org/abs/1909.06335)
* [Reddi et al., *Adaptive Federated Optimization* (FedAdam/FedYogi), ICLR 2021](https://arxiv.org/abs/2003.00295)
* [Yin et al., *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates* (median, trimmed mean), ICML 2018](https://arxiv.org/abs/1803.01498)
* [Blanchard et al., *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* (Krum), NeurIPS 2017](https://arxiv.org/abs/1703.02757)
* [Li et al., *Fair Resource Allocation in Federated Learning* (q-FedAvg), ICLR 2020](https://arxiv.org/abs/1905.10497)
* [McMahan et al., *Learning Differentially Private Recurrent Language Models* (DP-FedAvg), ICLR 2018](https://arxiv.org/abs/1710.06963)
