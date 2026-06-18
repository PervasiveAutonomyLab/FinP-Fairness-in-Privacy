# FinP: Fairness-in-Privacy in Federated Learning

Codebase for the paper
**"FinP: Fairness-in-Privacy in Federated Learning by Addressing Disparities in Privacy Risk"**
([arXiv:2502.17748](https://arxiv.org/abs/2502.17748)).  Our paper has been accepted for publication in Proceedings on Privacy Enhancing Technologies (PoPETs) 2026, Issue 4. 

FinP mitigates per-client disparities in privacy risk under **Source Inference Attacks
(SIA)** and **Membership Inference Attacks (MIA)** via two levers:

- `--opt`  — server-side aggregation that optimizes client weights.
- `--col`  — client-side adaptive loss regularization (sharpness/Hessian-aware).

Everything is driven by `main_fed.py`; the `experiments/` package automates the
full experiment matrix per dataset and tabulates the results for comparison.

---

## 1. Environment setup

Requires Python 3.12. Using Conda:

```bash
conda env create -f environment.yml
conda activate finp_v2
```

Or with pip:

```bash
pip install -r requirements.txt
```

Key dependencies: `torch==2.6.0`, `torchvision==0.21.0`, `opacus==1.6.0` (DP
baseline), `datasets>=2.16.0` (FEMNIST), `scikit-learn`, `scipy`, `numpy`,
`pandas`, `matplotlib`, `seaborn`, `Pillow`.

---

## 2. Datasets

| Dataset  | Source                                    | Notes |
|----------|-------------------------------------------|-------|
| FEMNIST  | HuggingFace `flwrlabs/femnist` (auto-download, cached) | one writer per client |
| CIFAR-10 | torchvision (auto-download to `data/cifar/`) | Dirichlet non-IID split |
| HAR      | ships in `data/har/` (no auto-download)   | UCI Human Activity Recognition |

The first FEMNIST/CIFAR run downloads and caches the data; subsequent runs are offline.

---

## 3. Reproducing the experiments (recommended)

Run an entire dataset's experiment matrix and get a comparison table with a single
command:

```bash
python -m experiments.run --dataset femnist
```

Available datasets: `femnist`, `femnist-mia`, `har`, `cifar-cnn`, `cifar-res`.

What each dataset runs (see `experiments/config.py` for exact commands):

| Dataset       | Experiments |
|---------------|-------------|
| `femnist`     | baseline; FinP β=0.5/0.75/1; DP clip 7.5, noise 0.75/1/2 |
| `femnist-mia` | same matrix, each with `--mia` (white-box MIA) |
| `har`         | baseline; server-only; client-only; FinP; DP clip 5, noise 0.5/1/2 |
| `cifar-cnn`   | baseline; FinP β=0.05/0.1/0.3; DP clip 1.75, noise 0.5/1/2 |
| `cifar-res`   | ResNet-56 baseline; FinP β=0.3 |

Useful flags:

```bash
python -m experiments.run --dataset femnist --list          # show commands, run nothing
python -m experiments.run --dataset femnist --dry-run        # print the exact python commands
python -m experiments.run --dataset femnist --only baseline,finp_beta1
python -m experiments.run --dataset femnist --epochs 2       # quick smoke test
python -m experiments.run --dataset femnist --compare-only   # re-tabulate existing logs
```

Each run is streamed to the console and tee'd to
`experiments/logs/<dataset>/<experiment>_<timestamp>.log`. After the runs, a
comparison table is printed and written to
`experiments/logs/<dataset>/comparison_<dataset>.csv`.

### Comparison metrics

The comparison is parsed from the `RUN SUMMARY` block each run prints. Columns:

| Column            | Meaning |
|-------------------|---------|
| `attack`          | `SIA`, or `MIA` for `--mia` runs |
| `train_last3` / `test_last3` | (1) mean train/test accuracy over the last 3 rounds |
| `attack_acc_mean` | (2) mean of average attack accuracy |
| `attack_acc_max`  | (3) max of average attack accuracy |
| `loss_mad_mean`   | (4) mean `average_loss_mad` |
| `loss_cov_mean` / `loss_fi_mean` | (5) mean Loss CoV / FI |
| `attack_cov_mean` / `attack_fi_mean` | (6) mean attack CoV / FI |
| `sen_welfare_mean`| (7) mean `sen_welfare` |

(8) For `--mia` runs, the attack columns (2)(3)(6)(7) automatically use the MIA
numbers (`mean/max MIA attack accuracy`, `mean MIA CoV/FI`,
`mean reverse_mia sen_welfare`) instead of the SIA ones.

You can also compare arbitrary logs directly:

```bash
python -m experiments.compare experiments/logs/femnist --csv out.csv
python -m experiments.compare run_a.log run_b.log
```

---

## 4. Running `main_fed.py` directly

The runner is a thin wrapper; you can always call `main_fed.py` yourself. Examples:

```bash
# FEMNIST (lr=0.02 and num_classes=62 are set automatically for --dataset FEMNIST)
python main_fed.py --dataset FEMNIST --model femnistnet --num_users 10 \
    --num_samples 100 --epochs 20 --local_ep 5                                  # baseline
python main_fed.py --dataset FEMNIST --model femnistnet --num_users 10 \
    --num_samples 100 --epochs 20 --local_ep 5 --opt --col --beta=1 \
    --hessian_eig_max_iter 20 --hessian_trace_max_iter 20 --hessian_tol 5e-3    # FinP
python main_fed.py --dataset FEMNIST --model femnistnet --num_users 10 \
    --num_samples 100 --epochs 20 --local_ep 5 --run_dp_baseline \
    --dp_clip 7.5 --dp_noise=1                                                   # DP baseline
# add --mia to any FEMNIST command to enable the white-box MIA simulation

# HAR
python main_fed.py --dataset=HAR --model=tcn --alpha=0.1 --num_users=10 --local_ep=5 --epochs=20 --opt --col

# CIFAR-10 CNN / ResNet
python main_fed.py --dataset=CIFAR10 --model=cnn --alpha=0.5 --num_users=10 --local_ep=5 --col --opt --beta=0.1
python main_fed.py --dataset=CIFAR10 --model=res --alpha=0.1 --num_users=10 --local_ep=5 --col --opt --beta=0.3
```

The full reference command list lives in `utils/prompts_need_run.txt`.

---

## 5. Repository layout

```
main_fed.py            Entry point: build model, federated training loop, SIA/MIA, RUN SUMMARY
experiments/           Automation
  config.py            Per-dataset experiment matrices
  run.py               Run a dataset's matrix, tee logs, emit comparison
  compare.py           Parse RUN SUMMARY blocks -> comparison table / CSV
models/                Nets, FedAvg, SIA, MIA, DP local update, Hessian utilities
utils/                 Dataset loaders (FEMNIST/HAR/CIFAR), options, logger, run summary
FedAlign/              FedAlign baseline (used by --model res --runfed)
plotting.py            Figure generation from saved .pkl results
data/har/              HAR dataset (ships with the repo)
archive/               Local-only backups / old logs / figures (gitignored)
```

---

