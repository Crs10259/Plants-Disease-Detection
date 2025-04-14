#!/usr/bin/env python
"""
Plants Disease Detection System
Main entry point for data preparation, training, and inference
"""

import os
import argparse
import logging
import sys
from typing import Dict, Any, List, Optional
from config.config import config, paths
from libs.training import train_model
from libs.inference import predict
from dataset.data_prep import setup_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Main')

def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Plant Disease Detection System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Data preparation command
    prep_parser = subparsers.add_parser('prepare', 
                                        help='Prepare datasets (extract, process, augment)')
    prep_parser.add_argument('--extract', action='store_true', 
                            help='Extract dataset archives')
    prep_parser.add_argument('--process', action='store_true', 
                            help='Process images from temp directory into training directory')
    prep_parser.add_argument('--augment', action='store_true', 
                            help='Perform data augmentation')
    prep_parser.add_argument('--all', action='store_true', 
                            help='Perform all data preparation steps')
    
    # Training command
    train_parser = subparsers.add_parser('train', 
                                         help='Train a model')
    train_parser.add_argument('--epochs', type=int, 
                             help='Number of epochs (default: from config)')
    train_parser.add_argument('--model', type=str, 
                             help='Model name (default: from config)')
    train_parser.add_argument('--batch-size', type=int, 
                             help='Batch size (default: from config)')
    train_parser.add_argument('--prepare', action='store_true', 
                             help='Run data preparation before training')
    
    # Inference command
    infer_parser = subparsers.add_parser('predict', 
                                         help='Run inference with a trained model')
    infer_parser.add_argument('--model', type=str, required=True, 
                             help='Path to model weights file')
    infer_parser.add_argument('--input', type=str, required=True, 
                             help='Path to image folder or single image')
    infer_parser.add_argument('--output', type=str, 
                             help=f'Output JSON file path (default: {paths.prediction_file})')
    
    return parser

def prepare_data(args) -> Dict[str, Any]:
    """Run data preparation based on command line arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with data preparation status
    """
    # If --all specified or no specific flags set, run all preparation steps
    if args.all or not (args.extract or args.process or args.augment):
        args.extract = True
        args.process = True
        args.augment = True
    
    # Run data preparation
    return setup_data(
        extract=args.extract,
        process=args.process,
        augment=args.augment
    )

def train_pipeline(args) -> Dict[str, Any]:
    """Run training pipeline based on command line arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with training results
    """
    # Override config values if specified in arguments
    if args.model:
        config.model_name = args.model
        logger.info(f"Using model: {config.model_name}")
        
    if args.batch_size:
        config.train_batch_size = args.batch_size
        logger.info(f"Using batch size: {config.train_batch_size}")
        
    epochs = args.epochs or config.epoch
    
    # By default, run data preparation before training
    if args.prepare or True:  # Always run data preparation for better results
        logger.info("Running data preparation before training")
        setup_data(extract=True, process=True, augment=True)
    
    # Run training
    logger.info(f"Starting training for {epochs} epochs")
    return train_model(config)

def run_inference(args) -> None:
    """Run inference based on command line arguments
    
    Args:
        args: Command line arguments
    """
    model_path = args.model
    input_path = args.input
    output_path = args.output or paths.prediction_file
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Check if input exists
    if not os.path.exists(input_path):
        logger.error(f"Input file or folder not found: {input_path}")
        return
    
    # Handle single image vs directory
    if os.path.isfile(input_path):
        logger.info(f"Running inference on single image: {input_path}")
        # Create temporary directory with the single image
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.copy(input_path, temp_dir)
            predict(model_path, temp_dir, output_path)
    else:
        logger.info(f"Running inference on directory: {input_path}")
        predict(model_path, input_path, output_path)
    
    logger.info(f"Inference completed, results saved to {output_path}")

def main():
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()
    
    # If no command specified, run everything in sequence
    if not args.command:
        logger.info("No command specified. Running complete pipeline...")
        
        # 1. Prepare data
        logger.info("Step 1: Preparing data...")
        setup_data(extract=True, process=True, augment=True)
        
        # 2. Train the model (if weights directory doesn't exist)
        if not os.path.exists(paths.weights_dir) or not os.listdir(paths.weights_dir):
            logger.info("Step 2: Training model...")
            train_model(config)
        else:
            logger.info("Step 2: Skipping training as weights already exist.")
        
        # 3. Run inference if test data exists
        test_imgs = os.path.join(paths.test_images_dir)
        if os.path.exists(test_imgs) and os.listdir(test_imgs):
            # Find the latest model file
            model_files = []
            for root, dirs, files in os.walk(paths.best_weights_dir):
                for file in files:
                    if file.endswith('.pth'):
                        model_files.append(os.path.join(root, file))
            
            if model_files:
                latest_model = max(model_files, key=os.path.getmtime)
                logger.info(f"Step 3: Running inference with {latest_model}...")
                predict(latest_model, test_imgs, paths.prediction_file)
            else:
                logger.info("Step 3: No model files found for inference.")
        else:
            logger.info("Step 3: Skipping inference as no test data exists.")
        
        return
    
    # Execute requested command
    if args.command == 'prepare':
        prepare_data(args)
    elif args.command == 'train':
        train_pipeline(args)
    elif args.command == 'predict':
        run_inference(args)

if __name__ == "__main__":
    main() 