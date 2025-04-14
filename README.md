# Plant Disease Detection System

A comprehensive system for plant disease detection using deep learning. This project provides an end-to-end pipeline for preparing data, training models, and running inference for plant disease classification.

## Project Structure

```
├── config/                  # Configuration module
│   ├── __init__.py
│   ├── config.py            # Configuration settings
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── utils.py
├── dataset/                 # Dataset package
│   ├── __init__.py          # Dataset package definitions
│   ├── dataloader.py        # Data loading utilities
│   ├── data_prep.py         # Data preparation utilities
├── models/                  # Model definitions
│   ├── __init__.py
│   ├── model.py             # Model architecture definitions
├── libs/                    # Core functionality
│   ├── __init__.py
│   ├── training.py          # Training functionality
│   ├── inference.py         # Inference functionality
├── main.py                  # Main entry point
├── requirements.txt         # Dependencies
```

## Dataset

This project uses the Agricultural Disease dataset from the AI Challenger competition. 

**Dataset Source**: [https://github.com/spytensor/plants_disease_detection](https://github.com/spytensor/plants_disease_detection)

The dataset contains images of various plant diseases across different plant species, organized by disease class. The data preparation module in this project handles the extraction, processing, and augmentation of this dataset automatically.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Crs10259/Plants-Disease-Detection.git
cd Plants-Disease-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The system provides a unified command-line interface through `main.py` that can run the entire pipeline automatically:

```bash
# Run the entire pipeline (data preparation, training, and inference)
python main.py
```

For more control, you can use the specific commands:

### Data Preparation

Prepare datasets for training and testing:

```bash
# Perform all data preparation steps (default behavior)
python main.py prepare

# Extract datasets from archives only
python main.py prepare --extract

# Process extracted data into training directory only
python main.py prepare --process

# Perform data augmentation only
python main.py prepare --augment
```

### Training

Train a model with various options:

```bash
# Train with default settings (includes data preparation)
python main.py train

# Train with specific model
python main.py train --model efficientnetv2_s

# Train with specific epochs and batch size
python main.py train --epochs 50 --batch-size 32

# Train without running data preparation
python main.py train --no-prepare
```

### Inference

Run inference on images:

```bash
# Run inference on a directory of images
python main.py predict --model weights/best/efficientnetv2_s/0/model_best.pth --input data/test/images

# Run inference on a single image
python main.py predict --model weights/best/efficientnetv2_s/0/model_best.pth --input data/test/images/sample.jpg

# Specify custom output path
python main.py predict --model weights/best/efficientnetv2_s/0/model_best.pth --input data/test/images --output results.json
```

## Key Features

### Automatic Data Merging

The system now automatically merges multiple datasets by default, making it easier to train on combined data sources.

### Enhanced Data Preparation

The data preparation module has been enhanced to:
- Automatically extract datasets
- Process images to the correct format
- Remove invalid or corrupted images
- Perform data augmentation to increase training data

### Streamlined Training

The training process is now more efficient with:
- Automatic data preparation before training
- Improved model checkpointing
- Memory tracking and optimization
- Various data augmentation techniques during training

## Supported Models

The system supports multiple model architectures:

- DenseNet169
- EfficientNet-B4
- EfficientNetV2-S (default)
- ConvNeXt Small
- Swin Transformer
- Hybrid Model (CNN+Transformer)
- Ensemble Model

## Configuration

The system is highly configurable via `config/config.py`. Key configuration options include:

- Data paths
- Training parameters (learning rate, batch size, etc.)
- Data augmentation settings
- Model selection
- Optimization settings

Data merging and augmentation are now enabled by default.

## Data Augmentation

The system supports various data augmentation techniques:

- Gaussian noise
- Brightness adjustment
- Image flipping
- Contrast enhancement
- Advanced augmentations via Albumentations

Data augmentation is enabled by default to improve model performance.

## Requirements

- Python 3.7+
- PyTorch 1.12.0+
- torchvision 0.13.0+
- timm 0.6.12+
- OpenCV
- albumentations
- scikit-image
- Pillow
- tqdm
- numpy
- pandas

For a complete list of dependencies, see `requirements.txt`.

## License

[GNU General Public License v3.0](LICENSE)