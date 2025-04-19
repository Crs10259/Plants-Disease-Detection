# Plant Disease Detection System

A comprehensive system for detecting plant diseases using deep learning.

## Project Structure

```
Plants-Disease-Detection/
│
├── config/                 # Configuration files
├── dataset/                # Dataset handling
├── libs/                   # Core libraries
├── logs/                   # Log files
├── models/                 # Model definitions
└── weights/                # Saved model weights
```

## Dataset Source

This project uses the Agricultural Disease dataset from the AI Challenger competition.
Source: [@spytensor/plants_disease_detection](https://github.com/spytensor/plants_disease_detection)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Plants-Disease-Detection.git
cd Plants-Disease-Detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

The system supports using custom dataset paths, allowing you to point to external datasets:

```bash
# Prepare data from default location
python main.py prepare --all

# Prepare data from a custom location
python main.py prepare --all --dataset-path /path/to/your/dataset

# Check data preparation status
python main.py prepare --status

# Extract dataset only
python main.py prepare --extract

# Process images only
python main.py prepare --process

# Perform data augmentation
python main.py prepare --augment

# Merge datasets (train, test, val, or all)
python main.py prepare --merge all

# Control merging of augmented data
python main.py prepare --augment --merge-augmented
python main.py prepare --augment --no-merge-augmented
```

#### Using Custom Dataset Paths

The system can use dataset files from custom locations:

1. **Direct file paths**: Point directly to dataset ZIP files:
```bash
   python main.py prepare --extract --dataset-path /path/to/dataset.zip
```

2. **Directory paths**: Point to a directory containing dataset files:
```bash
   python main.py prepare --extract --dataset-path /path/to/dataset_dir
   ```

3. **Supported formats**: The system supports `.zip`, `.rar`, `.tar`, `.gz`, and `.tgz` formats.

If files aren't found in the custom path, the system will fall back to the default data directory.

### Training

```bash
# Train with default settings
python main.py train

# Train with custom settings
python main.py train --epochs 100 --model efficientnet_b4 --batch-size 16

# Train with data preparation first
python main.py train --prepare

# Train with a custom dataset path
python main.py train --dataset-path /path/to/your/dataset
```

### Inference

```bash
# Run inference on a folder of images
python main.py predict --model weights/best_weights/model_best.pth.tar --input test_images/

# Run inference on a single image
python main.py predict --model weights/best_weights/model_best.pth.tar --input test_images/apple.jpg

# Specify output path
python main.py predict --model weights/best_weights/model_best.pth.tar --input test_images/ --output results.json
```

## Key Features

- Automatic merging of datasets
- Enhanced data preparation with validation
- Advanced data augmentation
- Flexible model support

## Supported Models

- EfficientNet (B0-B7)
- ResNet (18, 34, 50, 101, 152)
- VGG (16, 19)
- DenseNet (121, 161, 169, 201)

## Configuration Options

Edit `config/config.py` to customize:

- Training parameters
- Data paths
- Augmentation settings
- Model selection

## Data Augmentation Techniques

- Noise addition
- Brightness adjustment
- Horizontal and vertical flips
- Advanced augmentations with Albumentations library

## Requirements

- Python 3.8+
- PyTorch 1.12.0+
- torchvision 0.13.0+
- numpy 1.21.0+
- Pillow 9.0.0+
- opencv-python 4.6.0+
- albumentations 1.2.0+
- tqdm 4.64.0+
- matplotlib 3.5.0+
- scikit-learn 1.1.0+

## License

MIT License