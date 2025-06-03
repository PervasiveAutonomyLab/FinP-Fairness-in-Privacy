# FinP: Fairness-in-Privacy in Federated Learning

Codebase for the paper:  
**"FinP: Fairness-in-Privacy in Federated Learning by Addressing Disparities in Privacy Risk"**

This repository contains implementations and experiment scripts to reproduce the results presented in the [paper](https://arxiv.org/abs/2502.17748).

---

## 📦 Environment Setup

We recommend using [Conda](https://docs.conda.io/) for environment management.

```bash
conda env create -f environment.yml
conda activate finp
```

---

## 🚀 How to Run Experiments

Use --opt to activate server side PCA aggregation; Use --col to activate client side adaptive loss regularization.

### 📊 Human Activity Recognition (HAR) Dataset

**Baseline:**
```bash
python main_fed.py --dataset=HAR --model=tcn --alpha=0.1 --num_users=10 --local_ep=1 --epochs=20
```

**Server-Only:**
```bash
python main_fed.py --dataset=HAR --model=tcn --alpha=0.1 --num_users=10 --local_ep=1 --epochs=20 --opt
```

**Client-Only:**
```bash
python main_fed.py --dataset=HAR --model=tcn --alpha=0.1 --num_users=10 --local_ep=1 --epochs=20 --col --beta=2
```

**FinP (Full):**
```bash
python main_fed.py --dataset=HAR --model=tcn --alpha=0.1 --num_users=10 --local_ep=1 --epochs=20 --opt --col --beta=2
```

---

### 🖼️ CIFAR-10 Dataset

#### CNN Models

**Baseline (CNN):**
```bash
python main_fed.py --dataset=CIFAR10 --model=cnn --alpha=0.5 --num_users=10 --local_ep=5 --epochs=20
```

**FinP (CNN):**
```bash
python main_fed.py --dataset=CIFAR10 --model=cnn --alpha=0.5 --num_users=10 --local_ep=5 --epochs=20 --opt --col --beta=0.1
```
*For ablation studies, vary `--beta` among `0.05`, `0.1`, `0.3`, and `0.5`.*

#### ResNet Models

**Baseline (ResNet with FedAvg):**
```bash
python main_fed.py --dataset=CIFAR10 --model=res --runfed --method=fedavg --alpha=0.5 --num_users=10 --local_ep=5 --epochs=20
```

**FedAlign (ResNet):**
```bash
python main_fed.py --dataset=CIFAR10 --model=res --runfed --method=fedalign --alpha=0.5 --num_users=10 --local_ep=5 --epochs=20
```

**FinP (ResNet):**
```bash
python main_fed.py --dataset=CIFAR10 --model=res --alpha=0.5 --num_users=10 --local_ep=5 --opt --col --beta=0.05 --epochs=20
```

---

## 📈 Plotting Results

Each experiment generates a `.pkl` file in the `results/` folder.

To create plots:
1. Copy the desired pickle file into the same directory as `plotting.py`.
2. Follow the instructions inside `plotting.py` to reproduce the figures used in the paper.

---

## 📚 Dependencies

The main Python libraries used are:

```
matplotlib==3.10.3
numpy==2.2.6
pandas==2.2.3
scikit_learn==1.6.1
scipy==1.15.3
seaborn==0.13.2
torch==2.6.0
torchvision==0.21.0
```

All dependencies are included in the `environment.yml` file.

---