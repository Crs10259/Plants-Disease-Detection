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
from libs.inference import predict
from dataset.data_prep import setup_data
import torch
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', f'main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    prep_parser.add_argument('--status', action='store_true', 
                            help='Check data preparation status')
    prep_parser.add_argument('--all', action='store_true', 
                            help='Perform all data preparation steps')
    prep_parser.add_argument('--merge', choices=['train', 'test', 'val', 'all'], 
                            help='Merge datasets')
    prep_parser.add_argument('--dataset-path', type=str, 
                            help='Custom path to dataset files or directory')
    prep_parser.add_argument('--merge-augmented', action='store_true', 
                            help='Merge augmented data with original data')
    prep_parser.add_argument('--no-merge-augmented', action='store_true', 
                            help='Do not merge augmented data with original data')
    prep_parser.add_argument('--cleanup', action='store_true', 
                            help='Clean up temporary files after processing')
    
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
    train_parser.add_argument('--merge-augmented', action='store_true', 
                             help='Merge augmented data with original training data')
    train_parser.add_argument('--no-merge-augmented', action='store_true', 
                             help='Do not merge augmented data with original training data')
    train_parser.add_argument('--dataset-path', type=str, 
                             help='Custom path to dataset files or directory')
    
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
    logger.info("Starting data preparation")
    
    # Handle dataset path
    custom_dataset_path = getattr(args, 'dataset_path', None)
    if custom_dataset_path:
        logger.info(f"Using custom dataset path: {custom_dataset_path}")
        # Update config if a custom path is provided
        config.use_custom_dataset_path = True
        
    # If status flag is the only one set, just check status
    if args.status and not any([args.extract, args.process, args.augment, args.all, args.merge]):
        result = setup_data(extract=False, process=False, augment=False, status=True, 
                  merge=None, custom_dataset_path=custom_dataset_path)
        return result
    
    # Handle the --all flag
    if args.all:
        args.extract = args.process = args.augment = True
        if not args.merge:
            args.merge = "all"
    
    # If no flags are set, default to showing status
    if not any([args.extract, args.process, args.augment, args.merge]):
        logger.info("No specific preparation steps specified, showing data status")
        result = setup_data(extract=False, process=False, augment=False, status=True,
                  merge=None, custom_dataset_path=custom_dataset_path)
        return result
    
    # Handle augmented data merging preference
    merge_augmented = None
    if hasattr(args, 'merge_augmented') and args.merge_augmented:
        merge_augmented = True
        logger.info("Enabling merging of augmented data with original data")
    elif hasattr(args, 'no_merge_augmented') and args.no_merge_augmented:
        merge_augmented = False
        logger.info("Disabling merging of augmented data with original data")
    
    # Handle cleanup flag
    cleanup_temp = getattr(args, 'cleanup', False)
    if cleanup_temp:
        logger.info("Will clean up temporary files after processing")
    
    # Run data preparation with specified steps
    merge_option = args.merge if args.merge else None
    result = setup_data(
        extract=args.extract,
        process=args.process,
        augment=args.augment,
        status=args.status if args.status else True,  # Always show status at the end
        merge=merge_option,
        cleanup_temp=cleanup_temp,
        custom_dataset_path=custom_dataset_path,
        merge_augmented=merge_augmented
    )
    
    # Display warnings
    if "warnings" in result and result["warnings"]:
        logger.info("Data preparation completed with warnings:")
        for warning in result["warnings"]:
            logger.warning(f"  - {warning}")
    else:
        logger.info("Data preparation completed successfully")
        
    return result

def train_pipeline(args) -> Dict[str, Any]:
    """Run training pipeline based on command line arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with training results
    """
    logger.info("Starting training pipeline")
    
    # Handle dataset path
    custom_dataset_path = getattr(args, 'dataset_path', None)
    if custom_dataset_path:
        logger.info(f"Using custom dataset path: {custom_dataset_path}")
        # Update config if a custom path is provided
        config.use_custom_dataset_path = True
    
    # Run data preparation if requested
    if args.prepare:
        prepare_args = argparse.Namespace(
            extract=True, process=True, augment=True, 
            all=False, status=True, merge="all",
            dataset_path=custom_dataset_path,
            merge_augmented=args.merge_augmented if hasattr(args, 'merge_augmented') else None,
            no_merge_augmented=args.no_merge_augmented if hasattr(args, 'no_merge_augmented') else None,
            cleanup=getattr(args, 'cleanup', False)
        )
        prepare_data(prepare_args)
    
    # Import here to avoid circular imports
    from libs.training import train_model
    
    # Override config with command line arguments
    if args.epochs:
        config.epoch = args.epochs
    if args.model:
        config.model_name = args.model
    if args.batch_size:
        config.train_batch_size = args.batch_size
    
    # Run training
    train_model()
    logger.info("Training completed")

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
    
    # Create a default Namespace if no arguments were provided
    if len(sys.argv) == 1:
        args = argparse.Namespace(command=None)
    
    # If no command specified, run everything in sequence
    if not args.command:
        logger.info("No command specified. Running complete pipeline...")
        
        # Get possible custom dataset path
        custom_dataset_path = getattr(args, 'dataset_path', None)
        if custom_dataset_path:
            logger.info(f"Using custom dataset path: {custom_dataset_path}")
            config.use_custom_dataset_path = True
        
        # Check if there are settings for merging augmented data
        merge_augmented = None
        if hasattr(args, 'merge_augmented') and args.merge_augmented:
            merge_augmented = True
        elif hasattr(args, 'no_merge_augmented') and args.no_merge_augmented:
            merge_augmented = False
        
        # 1. Prepare data
        logger.info("Step 1: Preparing data...")
        prep_result = setup_data(
            extract=True, 
            process=True, 
            augment=True, 
            cleanup_temp=getattr(args, 'cleanup', False),
            custom_dataset_path=custom_dataset_path, 
            merge_augmented=merge_augmented
        )
                      
        # Display any warnings about dataset path fallback
        if "warnings" in prep_result and prep_result["warnings"]:
            logger.warning("Data preparation completed with warnings:")
            for warning in prep_result["warnings"]:
                logger.warning(f"  - {warning}")
                
        # Check if preparation was successful
        if not prep_result.get("success", True):
            logger.error("Data preparation failed, cannot proceed with training")
            if "errors" in prep_result and prep_result["errors"]:
                for error in prep_result["errors"]:
                    logger.error(f"  - {error}")
            return
        
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
                trained_epochs = checkpoint.get('epoch', 0)val_annotations
                
                if trained_epochs >= config.epoch:
                    logger.info(f"Model already trained for {trained_epochs} epochs (target: {config.epoch})")
                    model_needs_training = False
                else:
                    logger.info(f"Model needs additional training: {trained_epochs}/{config.epoch} epochs completed")
            except Exception as e:
                logger.warning(f"Error checking model training status: {str(e)}. Will train from scratch.")
        
        if model_needs_training:
            logger.info("Step 2: Training model...")
            # Import the train_model function here to avoid circular imports
            from libs.training import train_model
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