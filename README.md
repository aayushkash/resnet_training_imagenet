# ResNet Training on ImageNet

This repository contains code for training ResNet models on the ImageNet dataset.

## Overview

This project implements ResNet model training on the ImageNet dataset using PyTorch. The implementation includes training scripts and configurations for reproducing the results.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- numpy
- PIL
- tqdm

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To train the model:

```bash
python src/train.py --data-path /path/to/imagenet \
                    --model resnet50 \
                    --batch-size 256 \
                    --epochs 90 \
                    --lr 0.1
```

## License

This project is licensed under the MIT License.

