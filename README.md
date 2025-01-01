# ResNet Training on ImageNet

This repository contains code for training ResNet models on the ImageNet dataset.

## Overview

This project implements ResNet model training on the ImageNet dataset using PyTorch. The implementation includes training scripts and configurations for reproducing the results.

HuggingFace Spaces: [ResNet50 Demo](https://huggingface.co/spaces/ViksML/RESNET50)

## Project Structure

```
resnet_training_imagenet/
├── src/
│   ├── train.py        # Main training script
│   ├── dataset.py      # Dataset loading and preprocessing
│   └── models/         # Model architectures
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- numpy
- PIL
- tqdm

## Installation

```bash
# Clone the repository
git clone https://github.com/aayushkash/resnet_training_imagenet.git
cd resnet_training_imagenet

# Install dependencies
pip install -r requirements.txt
```

### Data Preprocessing
- Images are resized to 224x224
- Random horizontal flip and crop for training
- Center crop for validation
- Normalization using ImageNet mean and std values

## Usage

### Training
```bash
python src/train.py --data-path /path/to/imagenet \
                    --model resnet50 \
                    --batch-size 256 \
                    --epochs 90 \
                    --lr 0.1
```

### Training Arguments
- `--data-path`: Path to ImageNet dataset directory
- `--model`: Model architecture (resnet18, resnet34, resnet50, resnet101, resnet152)
- `--batch-size`: Number of images per batch
- `--epochs`: Number of training epochs
- `--lr`: Initial learning rate

## Model Architecture

Available ResNet variants:
- ResNet-50


## Results

### Training Logs
Training logs from EC2 instances are available at:
```
/home/ubuntu/resnet_training_imagenet/logs/
```

The logs contain:
- Training loss per epoch
- Validation accuracy
- Learning rate schedules
- GPU utilization metrics
- Batch processing times


## Demo

You can try out the ResNet50 model using our interactive demo:
- 🤗 HuggingFace Space: [ViksML/RESNET50](https://huggingface.co/spaces/ViksML/RESNET50)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [ImageNet](https://www.image-net.org/)
- [PyTorch](https://pytorch.org/)

