"""Experiment matrices for each dataset.

Every experiment is a list of CLI arguments passed to ``main_fed.py``. The runner
(``experiments/run.py``) prepends ``python main_fed.py`` and appends logging, so the
definitions here stay declarative and easy to audit against the paper / the
commands documented in ``utils/prompts_need_run.txt``.

Datasets (keys usable with ``--dataset`` on the runner):
    femnist       baseline, FinP (beta 0.5/0.75/1), DP (noise 0.75/1/2)
    femnist-mia   same matrix, each with --mia (white-box membership inference)
    har           baseline, server-only, client-only, FinP, DP (noise 0.5/1/2)
    cifar-cnn     baseline, FinP (beta 0.05/0.1/0.3), DP (noise 0.5/1/2)
    cifar-res     ResNet-56 baseline, FinP (beta 0.3)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Experiment:
    """A single ``main_fed.py`` invocation.

    name : short identifier, also used as the log-file stem and table row label.
    args : CLI tokens appended after ``python main_fed.py``.
    """

    name: str
    args: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Shared argument fragments
# --------------------------------------------------------------------------- #

# FEMNIST: one writer per client; lr=0.02 and num_classes=62 are auto-set in
# utils/options.py when --dataset FEMNIST.
_FEMNIST_COMMON = [
    "--dataset", "FEMNIST", "--model", "femnistnet",
    "--num_users", "10", "--num_samples", "100",
    "--epochs", "20", "--local_ep", "5",
]

# Hessian estimator speed-up used for FEMNIST FinP runs (keeps results stable
# while cutting per-client Hessian time, see prompts_need_run.txt).
_FEMNIST_HESSIAN_FAST = [
    "--hessian_eig_max_iter", "20",
    "--hessian_trace_max_iter", "20",
    "--hessian_tol", "5e-3",
]

# Ordering here drives both run order and the comparison-table row order:
# FinP betas top-to-bottom 1 -> 0.5, DP noises top-to-bottom 2 -> 0.75.
_FEMNIST_DP_CLIP = ["--dp_clip", "7.5"]
_FEMNIST_BETAS = ["1", "0.75", "0.5"]
_FEMNIST_DP_NOISES = ["2", "1", "0.75"]

_HAR_COMMON = [
    "--dataset", "HAR", "--model", "tcn", "--alpha", "0.1",
    "--num_users", "10", "--local_ep", "5", "--epochs", "20",
]

_CIFAR_CNN_COMMON = [
    "--dataset", "CIFAR10", "--model", "cnn", "--alpha", "0.5",
    "--num_users", "10", "--local_ep", "5",
]

_CIFAR_RES_COMMON = [
    "--dataset", "CIFAR10", "--model", "res", "--alpha", "0.1",
    "--num_users", "10", "--local_ep", "5",
]


def _femnist_matrix(mia: bool) -> List[Experiment]:
    """FEMNIST matrix; when ``mia`` is True every run gets ``--mia`` appended."""
    mia_flag = ["--mia"] if mia else []
    tag = "_mia" if mia else ""
    exps: List[Experiment] = []

    exps.append(Experiment(f"baseline{tag}", [*_FEMNIST_COMMON, *mia_flag]))

    for beta in _FEMNIST_BETAS:
        exps.append(
            Experiment(
                f"finp_beta{beta}{tag}",
                [*_FEMNIST_COMMON, "--opt", "--col", f"--beta={beta}",
                 *_FEMNIST_HESSIAN_FAST, *mia_flag],
            )
        )

    for noise in _FEMNIST_DP_NOISES:
        exps.append(
            Experiment(
                f"dp_noise{noise}{tag}",
                [*_FEMNIST_COMMON, *mia_flag, "--run_dp_baseline",
                 *_FEMNIST_DP_CLIP, f"--dp_noise={noise}"],
            )
        )

    return exps


EXPERIMENTS = {
    "femnist": _femnist_matrix(mia=False),
    "femnist-mia": _femnist_matrix(mia=True),
    # Same ordering convention as FEMNIST: FinP variants first, DP noise high -> low.
    "har": [
        Experiment("baseline", [*_HAR_COMMON]),
        Experiment("finp_server", [*_HAR_COMMON, "--opt"]),
        Experiment("finp_client", [*_HAR_COMMON, "--col"]),
        Experiment("finp", [*_HAR_COMMON, "--opt", "--col"]),
        Experiment("dp_noise2", [*_HAR_COMMON, "--run_dp_baseline", "--dp_clip", "5", "--dp_noise=2"]),
        Experiment("dp_noise1", [*_HAR_COMMON, "--run_dp_baseline", "--dp_clip", "5", "--dp_noise=1"]),
        Experiment("dp_noise0.5", [*_HAR_COMMON, "--run_dp_baseline", "--dp_clip", "5", "--dp_noise=0.5"]),
    ],
    # FinP beta high -> low, DP noise high -> low (matching FEMNIST).
    "cifar-cnn": [
        Experiment("baseline", [*_CIFAR_CNN_COMMON]),
        Experiment("finp_beta0.3", [*_CIFAR_CNN_COMMON, "--col", "--opt", "--beta=0.3"]),
        Experiment("finp_beta0.1", [*_CIFAR_CNN_COMMON, "--col", "--opt", "--beta=0.1"]),
        Experiment("finp_beta0.05", [*_CIFAR_CNN_COMMON, "--col", "--opt", "--beta=0.05"]),
        Experiment("dp_noise2", [*_CIFAR_CNN_COMMON, "--run_dp_baseline", "--dp_clip", "1.75", "--dp_noise=2"]),
        Experiment("dp_noise1", [*_CIFAR_CNN_COMMON, "--run_dp_baseline", "--dp_clip", "1.75", "--dp_noise=1"]),
        Experiment("dp_noise0.5", [*_CIFAR_CNN_COMMON, "--run_dp_baseline", "--dp_clip", "1.75", "--dp_noise=0.5"]),
    ],
    "cifar-res": [
        Experiment("baseline", [*_CIFAR_RES_COMMON]),
        Experiment("finp_beta0.3", [*_CIFAR_RES_COMMON, "--col", "--opt", "--beta=0.3"]),
    ],
}

DATASETS = list(EXPERIMENTS.keys())
