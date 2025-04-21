# Final Project: Advanced Node Classification on Graphs

This project presents an in-depth exploration of node classification techniques on graph-structured data, with a focus on high-homophily datasets.

## Overview

The project implements and compares various graph-based model architectures for node classification, analyzing their performance on datasets with high homophily. The implementation includes:
- Multiple graph neural network architectures
- Traditional machine learning baselines
- Comprehensive evaluation metrics
- Detailed performance analysis

## Project Structure

- `models.py`: Main implementation file containing all model architectures and training logic
- `stats.py`: Dataset analysis and statistics
- `imports.py`: Common imports and utility functions
- `import.sh`: Script to set up the environment and install dependencies
- `data/`: Directory containing the datasets
- `img/`: Directory for storing generated visualizations
- `CS410_Final_Report.pdf`: Detailed project report with findings and analysis

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- NetworkX
- Other dependencies as specified in requirements.txt

### Installation

1. Run the import script to set up the environment:
   ```bash
   chmod +x import.sh
   ./import.sh
   ```

### Running the Code

1. To analyze the dataset:
   ```bash
   python3 stats.py
   ```
   This will print information about the dataset structure and characteristics.

2. To train and evaluate all models:
   ```bash
   python3 models.py
   ```
   This will:
   - Train all implemented models
   - Generate performance metrics
   - Create visualizations
   - Save results

## Output

The code will generate:
- Model performance metrics
- Training curves
- Comparison visualizations
- Detailed analysis results

## Project Report

The comprehensive project report (`CS410_Final_Report.pdf`) includes:
- Detailed methodology
- Model architectures
- Experimental results
- Analysis and conclusions
- Future work directions

## Note

Make sure to review the project report for a complete understanding of the implementation details and findings.