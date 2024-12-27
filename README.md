# Gaudi Suite for Communication-Efficient Federated Learning
Federated learning is a machine learning setting where many clients collaboratively train a model under the orchestration of a central server while keeping the training data decentralized. 
This approach can help address data privacy concerns and allow for lower-latency inference.
However, the communication cost of federated learning can be prohibitively high, especially for clients with slow or expensive communication links.

This repository contains a collection of communication-efficient federated learning algorithms implemented in PyTorch, supporting Gaudi HPU.

The main baseline algorithm is Federated Averaging ([FedAvg](https://arxiv.org/abs/1602.05629)), and the communication-efficient variants include:

- [Subsampling](https://arxiv.org/pdf/1610.05492) (Federated Averaging with Subsampling)
- [Quantization](https://arxiv.org/abs/1710.05492) (Federated Averaging with Quantization)
- [EvoFed](https://arxiv.org/abs/2003.00295) (Evolutionary Federated Learning)
- [MAPA](https://arxiv.org/abs/2003.00295) (Model-Agnostic Private Aggregation)
- [FA-LoRA](https://arxiv.org/abs/2003.00295) (Federated Averaging with Low-Rank Compression)
- [MA-LoRA](https://arxiv.org/abs/2003.00295) (Model-Agnostic Low-Rank Aggregation)

This suite is designed to be modular and extensible, allowing for easy integration of new algorithms and datasets or hybrid use of them together.

## Setup

### For CUDA on the host
```bash
conda create -n fl python=3.12
conda activate fl
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
``` 

### For Gaudi v2 on the docker container
```bash
# For wandb logging support
WANDB_API_KEY=<your_wandb_api_key>
WANDB_MODE="online"

# For Running on Gaudi Eager Mode
PT_HPU_LAZY_MODE=0

# Build the docker image
./docker_build_run.sh <image_name> Dockerfile . ./data
```

## Run

### For CUDA on the host
```bash
# Run the training script on GPU
python train.py

# Run the training script on CPU
python train.py --gpu -1
```

### For Gaudi v2 on the docker container
```bash
# Run the training script
./docker_run.sh python train.py --gaudi

# Run the training script on eager mode
PT_HPU_LAZY_MODE=0
./docker_run.sh python train.py --gaudi --eager

# Run the training script on CPU
./docker_run.sh python train.py --gpu -1
```

See the arguments in [config.py](utils/config.py). 

For example:
> python train.py --dataset mnist --iid --model cnn --epochs 50 --gaudi --all_clients  


## Results
TBD

## Acknowledgements
TBD


## References
This repository is inspired by following works:

- [shaoxiongji/federated-learning](https://github.com/shaoxiongji/federated-learning)
- [mahi97/EvoFL](https://github.com/mahi97/EvoFL)
- [AshwinRJ/Federated-Learning-PyTorch](https://github.com/AshwinRJ/Federated-Learning-PyTorch)
- [lokinko/Federated-Learning](https://github.com/lokinko/Federated-Learning)
- [research/federated](https://github.com/google-research/federated)
- [innovation-cat/Awesome-Federated-Machine-Learning](https://github.com/innovation-cat/Awesome-Federated-Machine-Learning)

