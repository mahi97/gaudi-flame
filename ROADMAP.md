# Roadmap 

This document outlines the roadmap for the project.

----
## 2024 3Q

### General:
- [x] **Env Setup**: Set up the docker workflow, scripts, and project structure.
- [x] **Benchmarking Data**: Benchmarking data load and preprocessing, and comparison with CUDA.

### Federated Learning Algorithms:

### Dataset Benchmarks:
- [x] **[MNIST](http://yann.lecun.com/exdb/mnist/)**: A dataset of handwritten digits (0-9) commonly used for image classification tasks.
- [x] **[Fashion-MNIST (FMNIST)](https://github.com/zalandoresearch/fashion-mnist)**: A dataset of Zalando's article images used for image classification tasks.

### Compression Techniques:

## 2024 4Q

### General:
### Federated Learning Algorithms:
- [x] **[FedAvg](https://arxiv.org/abs/1602.05629)**: Baseline Federated Averaging.

### Dataset Benchmarks:
- [x] **[MNIST](http://yann.lecun.com/exdb/mnist/)**: A dataset of handwritten digits (0-9) commonly used for image classification tasks.
- [x] **[Fashion-MNIST (FMNIST)](https://github.com/zalandoresearch/fashion-mnist)**: A dataset of Zalando's article images used for image classification tasks.
- [x] **[CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)**: Datasets containing 10 or 100 classes of 32x32 color images, widely used for image recognition tasks.


### Compression Techniques:
- [x] **[Sparse Update](https://arxiv.org/pdf/1610.05492)**: Federated Averaging with Sparse updates.
- [x] **[Quantization](https://arxiv.org/abs/1710.05492)**: Federated Averaging with Quantization.


## 2025 1Q


### Dataset Benchmarks:
- [ ] **[Shakespeare Dataset](https://leaf.cmu.edu/)**: A character-level dataset built from Shakespeare’s plays, used for next-character prediction tasks.
- [ ] **[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)**: A dataset for sentiment analysis containing 1.6 million tweets labeled as positive, negative, or neutral.

### Compression Techniques:
- [ ] **[EvoFed](https://arxiv.org/abs/2003.00295)**: Evolutionary Federated Learning.

## 2025 2Q

### General:
### Federated Learning Algorithms:
- [ ] **[FedProx](https://arxiv.org/abs/1812.06127)**: Federated Proximal to address heterogeneity.
- [ ] **[FedAvgM](https://arxiv.org/abs/1602.05629)**: Federated Averaging with Momentum.


### Dataset Benchmarks:
- [ ] **[Reddit Dataset](https://leaf.cmu.edu/)**: A dataset of user comments from Reddit structured for federated learning tasks like next-word prediction or topic modeling.

### Compression Techniques:
- [ ] **[FA-LoRA](https://arxiv.org/abs/2403.13269)**: Frozen-A Low-Rank Adaptation.
- [ ] **[MAPA (under review)]()**: Model-Agnostic Projection Adaptation.

And Benchmarked on:

## 2025 3Q

### General:
### Federated Learning Algorithms:

- [ ] **[FedAdagrad](https://arxiv.org/abs/2003.00295)**: Adaptive Gradient-based FL optimization.
- [ ] **[FedYogi](https://arxiv.org/abs/2003.00295)**: Variant of Adam optimized for FL.
- [ ] **[FedAdam](https://arxiv.org/abs/2003.00295)**: Adam optimizer adapted for federated setups.
- [ ] **[SCAFFOLD](https://arxiv.org/abs/1910.06378)**: Control variates for correcting local updates.

### Dataset Benchmarks:

- [ ] **[Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)**: General instruction tuning dataset with 52k samples.
- [ ] **[Alpaca-GPT4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4)**: GPT-4-generated instruction-response pairs.
- [ ] **[FinGPT](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train)**: Financial sentiment dataset with 77k samples.

### Compression Techniques:
- [ ] **[LoRA](https://arxiv.org/abs/2106.09685)**: Low-Rank Adaptation for computational and communication efficiency.
- [ ] **[MA-LoRA (under review)]()**: Model-Agnostic Low-Rank Adaptation.

## 2025 4Q
### General:
### Federated Learning Algorithms:
### Dataset Benchmarks:

- [ ] **[MedAlpaca](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)**: Medical instruction dataset with 34k samples.
- [ ] **[Code-Alpaca](https://huggingface.co/datasets/lucasmccabe-lmi/CodeAlpaca-20k)**: Code generation dataset with 20k samples.
- [ ] **[MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)**: Mathematical instruction tuning dataset with 225k samples.
- [ ] **[UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)**: Value alignment dataset emphasizing helpfulness.
- [ ] **[HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)**: Harmlessness and helpfulness preference dataset with 161k samples.

### Compression Techniques:

