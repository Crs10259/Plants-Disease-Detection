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

**Dataset Source**: [@https://github.com/spytensor/plants_disease_detection](https://github.com/spytensor/plants_disease_detection)

The dataset contains images of various plant diseases across different plant species, organized by disease class. The data preparation module in this project handles the extraction, processing, and augmentation of this dataset automatically.

### Data Preparation Workflow

The system provides a unified data preparation workflow through the `setup_data` function in `dataset/data_prep.py`:

1. **Prepare Data Directory**: Place dataset ZIP files in the `data/` directory.
2. **Data Extraction**: System automatically extracts dataset archives into appropriate directories.
3. **Data Processing**: Processes images and organizes them into class directories.
4. **Data Augmentation**: Applies techniques like noise addition, brightness adjustment, and advanced transformations.
5. **Dataset Merging**: Combines multiple datasets with the following logic:
   - If merging is enabled in the configuration, the system searches for directories containing the word "test" or a user-specified pattern
   - ZIP files are extracted first before merging
   - By default, the system verifies directory structures before merging (directories with incompatible structures are skipped)
   - When "force mode" is enabled, datasets are merged regardless of structure differences

This workflow can be triggered through the main interface:

```bash
# Run all data preparation steps
python main.py prepare --all

# Check current data status
python main.py prepare --status

# Force merge datasets even with structure differences
python main.py prepare --merge --force
```

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

3. Prepare the dataset:
```bash
# Create data directory if it doesn't exist
mkdir -p data

# Place dataset ZIP files in the data directory
# Expected ZIP files include:
# - ai_challenger_pdr2018_trainingset_20181023.zip
# - ai_challenger_pdr2018_validationset_20181023.zip
# - ai_challenger_pdr2018_testa_20181023.zip
# - Any additional dataset ZIP files
```

The system will automatically extract and process these files during the data preparation step.

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

# Merge datasets with default settings
python main.py prepare --merge

# Force merge datasets (ignoring structure differences)
python main.py prepare --merge --force

# Disable dataset merging
python main.py prepare --disable-merge

# Check data preparation status
python main.py prepare --status
```

Dataset merging behavior can be configured in `config.py` with the following options:
- `merge_datasets`: Enable/disable dataset merging (default: True)
- `merge_force`: Force merge even with structure differences (default: False)
- `test_name_pattern`: Pattern to identify test dataset archives
- `use_all_test_datasets`: Use all available test datasets (default: True)
- `primary_test_dataset`: Specific test dataset to use if not using all datasets

### Automatic Data Merging

The system automatically merges multiple datasets by default, making it easier to train on combined data sources. The merging process follows these rules:

- **Dataset Discovery**: The system searches for datasets in the data directory and its subdirectories based on naming patterns.
- **Structure Verification**: Before merging, the system checks if datasets have compatible structures:
  - For training data: Verifies that class directories follow the expected pattern
  - For test data: Verifies that images are in the correct format
- **Merging Logic**:
  - Training datasets: Images are copied to corresponding class directories with prefixes to avoid name conflicts
  - Test datasets: Images are copied to a single directory with source prefixes
  - When merging datasets with different structures, force mode can be enabled to override structure checks
- **Selective Merging**: You can specify which test datasets to use through the configuration:
  - Use all available test datasets (default)
  - Use only a specific test dataset (e.g., "testa")
  - Create custom dataset combinations

**Integration with Training Process**:
- The training, validation, and inference processes automatically use the merged datasets when dataset merging is enabled in the configuration (default setting).
- The system uses the `handle_datasets()` function to intelligently select the appropriate dataset:
  - If merging is enabled, it returns the path to the merged dataset
  - If only one dataset is available, it uses that dataset
  - If multiple datasets are available but merging is disabled, it selects the best dataset based on configuration

> **Important**: You don't need to manually specify the merged directory path for training or inference. The system automatically selects the correct dataset based on the merge configuration.

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

The system automatically merges multiple datasets, making it easier to train on combined data sources. Key aspects:

- Automatically discovers datasets in the data directory based on naming patterns
- Merges training, test, and validation datasets while preserving data organization
- Intelligently selects appropriate datasets for training and inference based on merge configuration
- Training process automatically uses merged datasets without additional configuration

> **Important**: The training, validation, and inference processes automatically use the merged datasets when dataset merging is enabled (default setting).

See the [Detailed Features](#detailed-features) section for more information.

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

## Detailed Features

### Automatic Data Merging Implementation

The dataset merging system is implemented through several key components:

1. **Dataset Discovery and Selection**:
   - The `handle_datasets()` function in `utils/utils.py` is responsible for finding datasets in the data directory
   - It returns either a list of dataset paths (for merging) or selects a specific dataset based on configuration

2. **Merging Process**:
   - Training datasets: Class directories are preserved when merging, with images copied to the corresponding class in the merged directory
   - Test datasets: All images are copied to a single directory with source prefixes to maintain uniqueness
   - File naming strategy: Images in merged directories have prefixes based on source dataset to avoid filename conflicts

3. **Structural Verification**:
   - Before merging, the system verifies directory structures are compatible
   - For training data: Checks that class directories follow the expected pattern (numeric class IDs)
   - For test data: Verifies that images are in the correct format (.jpg or .png)
   - Force mode bypasses these checks when enabled via `--force` parameter or `merge_force=True` configuration

4. **Integration with Training Workflow**:
   - Dataloader automatically calls `handle_datasets()` to determine which dataset to use
   - When merging is enabled, it returns the path to the merged dataset
   - This allows the system to transparently use merged datasets without changes to the training code

5. **Configuration Options**:
   - `merge_datasets`: Controls whether dataset merging is enabled (default: True)
   - `merge_force`: Forces merging regardless of structural differences (default: False)
   - `merge_on_startup`: Automatically merges datasets when application starts (default: True)
   - `use_all_test_datasets`: Uses all available test datasets (default: True)
   - `primary_test_dataset`: Specific test dataset to use if not using all datasets
   - `dataset_to_use`: Strategy for selecting datasets when merging is disabled ("first", "last", "specific", "auto")