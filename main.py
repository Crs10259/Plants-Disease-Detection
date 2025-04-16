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
import torch

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
    train_parser.add_argument('--no-prepare', action='store_true', 
                             help='Skip data preparation before training')
    train_parser.add_argument('--force-train', action='store_true', 
                             help='Force retraining even if model is already trained')
    
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
    config.epoch = epochs  # Update config epoch value for checkpoint checking
    
    # Run data preparation before training if not disabled
    if getattr(args, 'prepare', True) and not getattr(args, 'no_prepare', False):
        logger.info("Running data preparation before training")
        data_result = setup_data(extract=True, process=True, augment=True)
        
        if not data_result.get('completed', False):
            logger.error("Data preparation failed. Check logs for details.")
            return {"error": "Data preparation failed", "details": data_result.get('error', 'Unknown error')}
        
        logger.info("Data preparation completed successfully")
    else:
        logger.info("Skipping data preparation as requested")
    
    # Check if we need to train or continue training the model
    model_needs_training = True
    checkpoint_path = os.path.join(paths.weights_dir, config.model_name, "0", "_checkpoint.pth.tar")
    best_model_path = os.path.join(paths.best_weights_dir, config.model_name, "0", "model_best.pth.tar")
    
    if os.path.exists(checkpoint_path) or os.path.exists(best_model_path):
        model_path = best_model_path if os.path.exists(best_model_path) else checkpoint_path
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            trained_epochs = checkpoint.get('epoch', 0)
            
            if trained_epochs >= epochs:
                logger.info(f"Model already trained for {trained_epochs} epochs (target: {epochs})")
                if not getattr(args, 'force_train', False):
                    logger.info("Skipping training. Use --force-train to override.")
                    return {"completed": True, "epochs_trained": trained_epochs}
                else:
                    logger.info("Forcing retraining as requested.")
            else:
                logger.info(f"Continuing training from epoch {trained_epochs}/{epochs}")
        except Exception as e:
            logger.warning(f"Error checking model training status: {str(e)}. Will train from scratch.")
    
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
    
    # If input path is set to "auto", use the appropriate test directory
    if input_path.lower() == "auto":
        # First check for merged test directory if merging is enabled
        if config.merge_datasets and os.path.exists(paths.merged_test_dir) and os.listdir(paths.merged_test_dir):
            input_path = paths.merged_test_dir
            logger.info(f"Using merged test directory: {input_path}")
        else:
            # Fall back to standard test directory
            input_path = paths.test_images_dir
            logger.info(f"Using standard test directory: {input_path}")
    
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
        
        # 2. Train the model
        logger.info("Step 2: Checking training status...")
        
        # Check if we need to train or continue training
        model_needs_training = True
        checkpoint_path = os.path.join(paths.weights_dir, config.model_name, "0", "_checkpoint.pth.tar")
        best_model_path = os.path.join(paths.best_weights_dir, config.model_name, "0", "model_best.pth.tar")
        
        if os.path.exists(checkpoint_path) or os.path.exists(best_model_path):
            model_path = best_model_path if os.path.exists(best_model_path) else checkpoint_path
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                trained_epochs = checkpoint.get('epoch', 0)
                
                if trained_epochs >= config.epoch:
                    logger.info(f"Model already trained for {trained_epochs} epochs (target: {config.epoch})")
                    model_needs_training = False
                else:
                    logger.info(f"Model needs additional training: {trained_epochs}/{config.epoch} epochs completed")
            except Exception as e:
                logger.warning(f"Error checking model training status: {str(e)}. Will train from scratch.")
        
        if model_needs_training:
            logger.info("Step 2: Training model...")
            train_model(config)
        else:
            logger.info("Step 2: Skipping training as model is already fully trained.")
        
        # 3. Run inference if test data exists
        test_imgs = os.path.join(paths.test_images_dir)
        
        # If merging is enabled, check for merged test directory first
        if config.merge_datasets:
            merged_test_dir = paths.merged_test_dir
            if os.path.exists(merged_test_dir) and os.listdir(merged_test_dir):
                test_imgs = merged_test_dir
                logger.info(f"Using merged test data from: {merged_test_dir}")
            
        if os.path.exists(test_imgs) and os.listdir(test_imgs):
            # Find the latest model file
            model_files = []
            for root, dirs, files in os.walk(paths.best_weights_dir):
                for file in files:
                    if file.endswith('.pth') or file.endswith('.pth.tar'):
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