# FinP: Fairness-in-Privacy in Federated Learning by Addressing Disparities in Privacy Risk

Code for experiments in paper "FinP: Fairness-in-Privacy in Federated Learning by Addressing Disparities in Privacy Risk"

---


## Environment Setup

Conda Environment


conda env create -f environment.yml
conda activate finp

---

## How to Run

### For reproduce experiments of using HAR dataset:

baseline:
python main_fed.py --dataset=HAR --model=tcn --alpha=0.1 --num_users=10 --local_ep=1 --epochs=20 

Server only:
python main_fed.py --dataset=HAR --model=tcn --alpha=0.1 --num_users=10 --local_ep=1 --epochs=20 --opt 

Client only:
python main_fed.py --dataset=HAR --model=tcn --alpha=0.1 --num_users=10 --local_ep=1 --epochs=20 --col --beta=2

FinP:
python main_fed.py --dataset=HAR --model=tcn --alpha=0.1 --num_users=10 --local_ep=1 --epochs=20 --opt --col --beta=2

### For reproduce experiments of using CIFAR10 dataset:

baseline(CNN):
python main_fed.py --dataset=CIFAR10 --model=cnn --alpha=0.5 --num_users=10 --local_ep=5 --epochs=20

FinP(CNN): (Note: change --beta = 0.05, 0.1, 0.3, 0.5 to reproduce ablation experiments of beta)
python main_fed.py --dataset=CIFAR10 --model=cnn --alpha=0.5 --num_users=10 --local_ep=5 --epochs=20 --opt --col --beta=0.1

baseline(ResNet):
python main_fed.py --dataset=CIFAR10 --model=res --runfed --method=fedavg --alpha=0.5 --num_users=10 --local_ep=5 --epochs=20

FedAlign(ResNet):
python main_fed.py --dataset=CIFAR10 --model=res --runfed --method=fedalign --alpha=0.5 --num_users=10 --local_ep=5 --epochs=20

FinP(ResNet): 
python main_fed.py --dataset=CIFAR10 --model=res  --alpha=0.5 --num_users=10 --local_ep=5 --epochs=20 --opt --col --beta=0.05

### plotting
Each experiment will generate a pickle file in folder results/. copy the pickle file into the same directory as plotting.py and follow the instruction inside plotting.py to reproduce figures in paper

---

## Dependencies

Main libraries used:

matplotlib==3.10.3
numpy==2.2.6
pandas==2.2.3
scikit_learn==1.6.1
scipy==1.15.3
seaborn==0.13.2
torch==2.6.0
torchvision==0.21.0


---