# Artifact Appendix

Paper title: **FinP: Fairness-in-Privacy in Federated Learning by Addressing Disparities in Privacy Risk**

Requested Badge(s):
  - [X] **Available**
  - [X] **Functional**
  - [X] **Reproduced**

## Description

This artifact contains the full code, the datasets (or their auto-download
hooks), and the automation scripts needed to reproduce the experiments in the
paper *FinP: Fairness-in-Privacy in Federated Learning by Addressing Disparities
in Privacy Risk*. FinP is a federated-learning method that reduces disparity in
per-client privacy risk under **Source Inference Attacks (SIA)** and
**Membership Inference Attacks (MIA)** through two mechanisms: server-side
aggregation optimization (`--opt`) and client-side adaptive loss regularization
(`--col`, controlled by `--beta`).

The artifact provides:

- `main_fed.py` — the federated training loop with SIA/MIA evaluation that prints
  an end-of-run `RUN SUMMARY` of accuracy, attack, and fairness metrics.
- `experiments/` — per-dataset experiment matrices (`config.py`), an automation
  runner (`run.py`) that reproduces every configuration and tees logs, and a
  parser (`compare.py`) that tabulates the comparison metrics automatically.
- `models/`, `utils/`, `FedAlign/` — model definitions, dataset loaders, and the
  FedAlign baseline.

### Security/Privacy Issues and Ethical Concerns

There are no security, privacy, or ethical concerns. All attacks (SIA and MIA)
are *simulated* against models trained on public datasets; running the artifact
does not exfiltrate data, disable security mechanisms, or otherwise compromise
the host machine.

## Basic Requirements

A machine with a GPU is recommended (preferably an Nvidia A30 or above). The
artifact also runs on CPU, but substantially slower.

### Hardware Requirements

1. The artifact can run on a laptop with a capable GPU, but a server with an A30
   (or better) GPU is recommended to keep run times manageable.
2. Our experiments were run on Nvidia A100 GPUs.
3. No specialized hardware (HSMs, custom accelerators, etc.) is required.

### Software Requirements

1. Any Linux-based OS (e.g., a university compute server) is sufficient.
2. Python 3.12.7 is used to run this artifact.
3. All Python dependencies and their versions are listed in `requirements.txt`
   (and installed via `environment.yml`). Key packages: `torch==2.6.0`,
   `torchvision==0.21.0`, `opacus==1.6.0` (DP baseline), `datasets>=2.16.0`
   (FEMNIST), `scikit-learn`, `scipy`, `numpy`, `pandas`, `Pillow`.
4. Machine-learning models are provided in the `models/` folder.
5. Datasets:

   | Dataset  | Source                                    | Notes |
   |----------|-------------------------------------------|-------|
   | FEMNIST  | HuggingFace `flwrlabs/femnist` (auto-download, cached) | one writer per client |
   | CIFAR-10 | torchvision (auto-download to `data/cifar/`) | Dirichlet non-IID split |
   | HAR      | ships in `data/har/` (no auto-download)   | UCI Human Activity Recognition |

### Estimated Time and Storage Consumption

- **Compute time.** On an A100 GPU, a single FEMNIST run (20 communication
  rounds) takes a few minutes; the full FEMNIST matrix (7 runs) takes roughly
  1–2 hours. FEMNIST-mia adds MIA overhead per round. HAR runs complete in
  minutes; CIFAR-10 (CNN/ResNet) runs take longer on the first invocation due to
  the dataset download. On CPU these times grow by roughly an order of magnitude.
  A reviewer can validate functionality in ~2–3 minutes using the smoke test
  below.
- **Human time.** ~15 minutes of active setup; the rest is unattended compute.
- **Disk.** < 4 GB total (cached datasets, per-run pickles, and logs).

## Environment

### Accessibility

The artifact is hosted on GitHub:
<https://github.com/PervasiveAutonomyLab/FinP-Fairness-in-Privacy>

Reviewers should use the latest commit on the default branch. (A stable
commit-id / tag will be provided once artifact evaluation is finalized.)

### Set up the environment

Clone the repository and create the environment (Conda recommended):

```bash
git clone https://github.com/PervasiveAutonomyLab/FinP-Fairness-in-Privacy.git
cd FinP-Fairness-in-Privacy

# Option A: Conda (creates an env named finp_v2 with Python 3.12.7)
conda env create -f environment.yml
conda activate finp_v2

# Option B: pip into an existing Python 3.12 environment
pip install -r requirements.txt
```

Notes:

- The HAR dataset ships in `data/har/` — no download needed.
- The first FEMNIST and CIFAR-10 runs download and cache their datasets
  (HuggingFace and torchvision, respectively); subsequent runs are offline.
- Output directories (`checkpoint/`, `resultsdprun/`, `experiments/logs/`) are
  created automatically at run time.

### Testing the Environment

**Step 1.** Verify that dependencies import and that a GPU is detected (CPU-only
execution is supported but slower):

```bash
python -c "import torch, torchvision, opacus, datasets, sklearn, scipy, pandas, numpy, PIL; print('Environment OK; CUDA available:', torch.cuda.is_available())"
```

Then run a short, end-to-end smoke test (one FEMNIST configuration, 2 rounds):

```bash
python -m experiments.run --dataset femnist --only finp_beta1 --epochs 2
```

Expected output: per-round training logs, a `RUN SUMMARY` block, and finally an
`EXPERIMENT COMPARISON` table with a single `finp_beta1` (SIA) row, e.g.:

```
========================================================================
  EXPERIMENT COMPARISON for dataset 'femnist'
  (attack metrics use MIA when --mia, else SIA)
========================================================================
experiment  attack  train_last3(%)  test_last3(%)  attack_acc_mean  attack_acc_max  ...
----------  ------  --------------  -------------  ------------  -----------  ...
finp_beta1  SIA     ...             ...            ...           ...          ...
========================================================================
```

A CSV copy is written to `experiments/logs/femnist/comparison_femnist.csv`. This
test takes ~2–3 minutes on CPU (well under a minute on a modern GPU). If it
completes and produces the table, the environment is set up correctly.

## Artifact Evaluation (Required for Functional and Reproduced badges)

This section should include all the steps required to evaluate your artifact's
functionality and validate your paper's key results and claims. Therefore,
highlight your paper's main results and claims in the first subsection. And
describe the experiments that support your claims in the subsection after that.

### Main Results and Claims

In the FinP paper, we propose a method that reduces privacy disparity among
clients in a federated-learning setup when SIA and MIA attacks are launched.

#### Main Result 1: Protecting SIA with the FEMNIST dataset

Our paper claims that, under SIA on FEMNIST, FinP reduces loss disparity
(`loss_mad_mean`), `attack_acc_mean`, and `attack_acc_max` relative to the
baseline while keeping utility (train/test accuracy) relatively stable. In
contrast, DP worsens loss disparity (`loss_mad_mean`) and significantly
degrades utility (train/test accuracy). This claim is reproducible by executing
[Experiment 1: femnist](#experiment-1-femnist). In that experiment we run
baseline, FinP with β ∈ {0.5, 0.75, 1}, and DP with noise ∈ {0.75, 1, 2}. We
report these results in Table 4 of the paper.

#### Main Result 2: Protecting MIA with the FEMNIST dataset

Our paper claims that, under MIA on FEMNIST, FinP generally reduces loss
disparity (`loss_mad_mean`), `attack_acc_mean`, and `attack_acc_max` relative to
the baseline while keeping utility (train/test accuracy) relatively stable. In
contrast, DP worsens loss disparity (`loss_mad_mean`) and significantly
degrades utility (train/test accuracy). This claim is reproducible by executing
[Experiment 2: femnist-mia](#experiment-2-femnist-mia). In that experiment we
run baseline-mia, FinP-mia with β ∈ {0.5, 0.75, 1}, and DP-mia with noise ∈
{0.75, 1, 2}. We report these results in Table 5 of the paper.

### Experiments

Each full-dataset run prints a comparison table and writes
`experiments/logs/<dataset>/comparison_<dataset>.csv`. Rows are ordered:
baseline first, then FinP (β from 1 to 0.5), then DP (noise from 2 to 0.75).
Columns (parsed from each run's `RUN SUMMARY` block):

| Column | Meaning |
|--------|---------|
| `attack` | `SIA`, or `MIA` for `--mia` runs |
| `train_last3` / `test_last3` | mean train/test accuracy over the last 3 rounds |
| `attack_acc_mean` | mean of average attack accuracy |
| `attack_acc_max` | max of average attack accuracy |
| `attack_cov_mean` / `attack_fi_mean` | mean attack CoV / FI |
| `loss_cov_mean` / `loss_fi_mean` | mean Loss CoV / FI |
| `loss_mad_mean` | mean `average_loss_mad` |
| `sen_welfare_mean` | mean `sen_welfare` |

For `--mia` runs, the attack columns (`attack_acc_mean`, `attack_acc_max`,
`attack_cov_mean`, `attack_fi_mean`, `sen_welfare_mean`) use MIA metrics
instead of SIA.

#### Experiment 1: femnist

- **Time:** ~15 human-minutes + ~2 compute-hours (A100 GPU)
- **Storage:** < 4 GB

This experiment reproduces
[Main Result 1: Protecting SIA with the FEMNIST dataset](#main-result-1-protecting-sia-with-the-femnist-dataset).
The following command runs the full FEMNIST matrix automatically:

```bash
python -m experiments.run --dataset femnist
```

Runs (in order): baseline; FinP β = 1, 0.75, 0.5; DP clip 7.5, noise 0.75, 1, 2.
Each run is logged to `experiments/logs/femnist/`. At the end, the script
prints a comparison table and writes `comparison_femnist.csv`, which can be
compared directly to Table 4 in the paper.

#### Experiment 2: femnist-mia

- **Time:** ~15 human-minutes + ~3 compute-hours (A100 GPU)
- **Storage:** < 4 GB

This experiment reproduces
[Main Result 2: Protecting MIA with the FEMNIST dataset](#main-result-2-protecting-mia-with-the-femnist-dataset).
The following command runs the full FEMNIST-mia matrix automatically:

```bash
python -m experiments.run --dataset femnist-mia
```

Runs (in order): baseline-mia; FinP-mia β = 1, 0.75, 0.5; DP-mia clip 7.5,
noise 0.75, 1, 2. Each run is logged to `experiments/logs/femnist-mia/`. At the
end, the script prints a comparison table and writes `comparison_femnist-mia.csv`,
which can be compared directly to Table 5 in the paper.

Additional datasets (`har`, `cifar-cnn`, `cifar-res`) are described in
`README.md` and can be run with `python -m experiments.run --dataset <name>`,
but they are not required for the main claims above.

## Limitations

- **Not bit-for-bit reproducible across runs.** All RNGs are seeded from
  `--manualseed` (default 42), but some PyTorch operators are non-deterministic
  and the adaptive `--col` feedback amplifies tiny floating-point differences, so
  headline metrics vary by a small margin run to run. The trends, orderings, and
  conclusions reported in the paper are stable; reviewers should compare results
  as distributions/trends rather than expecting identical digits.
- **Network on first run.** FEMNIST (HuggingFace) and CIFAR-10 (torchvision) are
  downloaded and cached on first use; an internet connection is required for that
  initial download. HAR ships with the artifact.
- **GPU strongly recommended.** CPU execution is supported but markedly slower,
  particularly for FEMNIST FinP runs because of the per-client Hessian
  estimation.
- **Other datasets.** Results for `har`, `cifar-cnn`, and `cifar-res` are
  described in `README.md` but are not listed under Main Results and Claims,
  because they take longer to run and FEMNIST results are sufficient to validate
  our claims.
- **FedAlign baseline.** The FedAlign ResNet path (`--model res --runfed`) is
  included for completeness and is not part of the main FinP comparison.

## Notes on Reusability

This artifact is designed to be extended beyond the paper:

- **Add new experiments or datasets to the automation.** Each experiment in
  `experiments/config.py` is simply a named list of `main_fed.py` CLI flags. Add
  an `Experiment(...)` entry (or a new dataset key) and the runner
  (`experiments/run.py`) and comparison tooling pick it up automatically.
- **Reuse the metric parser.** `experiments/compare.py` can tabulate any logs
  that contain a `RUN SUMMARY` block, independent of how they were produced:

  ```bash
  python -m experiments.compare path/to/logs_or_dir --csv out.csv
  ```

- **Compose the method via flags.** `main_fed.py` exposes the building blocks
  independently: `--opt` (server-side aggregation optimization), `--col`
  (client-side adaptive regularization, with `--beta`), `--run_dp_baseline` (the
  Opacus DP-SGD baseline with `--dp_clip` / `--dp_noise`), and `--mia` (white-box
  membership inference). They can be mixed to study new combinations.
- **Plug in new models/datasets.** Add a network in `models/Nets.py` and a loader
  in `utils/dataset.py`, then wire them into `build_model()` and `get_dataset()`
  in the respective files.
