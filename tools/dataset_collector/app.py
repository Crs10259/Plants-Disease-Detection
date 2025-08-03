#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
import argparse
from pathlib import Path

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import data collector components
try:
    # Try relative imports for packaged use
    from .dataset_collector import MainWindow
except ImportError:
    # Fall back to direct imports for direct script execution
    from tools.dataset_collector.dataset_collector import MainWindow

# Import the UI components
try:
    from PyQt6.QtWidgets import QApplication
except ImportError:
    print("PyQt6 is not installed. Please install it using:")
    print("pip install PyQt6>=6.4.0")
    sys.exit(1)

# Set up logging
def setup_logging():
    """Set up logging configuration"""
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('App')

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Plant Disease Data Collector Tool")
    parser.add_argument('--source-dir', type=str, help='Source directory for local import')
    parser.add_argument('--output-dir', type=str, help='Output directory for dataset')
    parser.add_argument('--search-terms', type=str, help='Comma-separated search terms for web collection')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()

def main():
    """Main application entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Log startup
    logger.info("Starting Plant Disease Data Collector Tool")
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Plant Disease Data Collector")
    app.setStyle('Fusion')  # Consistent look across platforms
    
    # Check if GUI is available
    if not args.headless:
        try:
            # Create main window
            window = MainWindow()
            
            # Pre-populate fields from command line arguments
            if args.source_dir:
                window.importer_tab.source_dir_edit.setText(args.source_dir)
                window.importer_tab.scan_source_directory()
                
            if args.output_dir:
                window.importer_tab.output_dir_edit.setText(args.output_dir)
                window.scraper_tab.output_dir_edit.setText(args.output_dir)
                
            if args.search_terms:
                window.scraper_tab.search_terms_edit.setPlainText(args.search_terms.replace(',', '\n'))
                window.tab_widget.setCurrentIndex(1)  # Switch to scraper tab
            
            # Show window
            window.show()
            
            # Run application event loop
            return app.exec()
            
        except Exception as e:
            logger.error(f"Error starting GUI: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"Error starting GUI: {str(e)}")
            sys.exit(1)
    else:
        # Headless mode - command line operation
        logger.info("Running in headless mode")
        
        if args.search_terms:
            # Web collection mode
            from dataset_maker import DatasetMaker
            
            search_terms = [term.strip() for term in args.search_terms.split(',')]
            output_dir = args.output_dir or os.path.join(parent_dir, "data")
            
            logger.info(f"Collecting data for search terms: {search_terms}")
            
            dataset_maker = DatasetMaker()
            result = dataset_maker.make_from_web(
                search_terms=search_terms,
                output_dir=output_dir,
                images_per_class=100,
                sources=["google", "bing", "baidu"],
                generate_labels=True
            )
            
            if result:
                logger.info("Data collection completed successfully")
                return 0
            else:
                logger.error("Data collection failed")
                return 1
                
        elif args.source_dir:
            # Local import mode
            from dataset_maker import DatasetMaker
            
            source_dir = args.source_dir
            output_dir = args.output_dir or os.path.join(parent_dir, "data")
            
            logger.info(f"Importing data from: {source_dir}")
            
            dataset_maker = DatasetMaker()
            result = dataset_maker.make_from_local(
                source_dir=source_dir,
                output_dir=output_dir,
                generate_labels=True
            )
            
            if result:
                logger.info("Data import completed successfully")
                return 0
            else:
                logger.error("Data import failed")
                return 1
                
        else:
            logger.error("Headless mode requires --source-dir or --search-terms")
            print("Error: Headless mode requires --source-dir or --search-terms")
            return 1

if __name__ == "__main__":
    sys.exit(main())