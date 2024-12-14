#!/bin/bash

# Update package lists
sudo apt-get update -y

# Install system-level dependencies
sudo apt-get install -y python3 python3-pip

# Create a virtual environment (optional but recommended)
python3 -m venv env
source env/bin/activate

# Install Python libraries
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
torch_geometric_whl="https://data.pyg.org/whl/torch-$(python3 -c 'import torch; print(torch.__version__)').html"
pip install torch-scatter -f "$torch_geometric_whl"
pip install torch-sparse torch-cluster torch-spline-conv -f "$torch_geometric_whl"
pip install scikit-learn matplotlib networkx numpy

# Install argparse (usually part of the standard library, but included for completeness)
pip install argparse

echo "All dependencies have been installed successfully!"