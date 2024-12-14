#!/bin/bash

# Update package lists
sudo apt-get update -y

# Install system-level dependencies
sudo apt-get install -y python3 python3-pip

# Create a virtual environment (optional but recommended)
python3 -m venv env
source env/bin/activate

# Install Python libraries
pip install recbole
pip install "ray[tune]"
pip install ray
pip install numpy==1.23.3
pip install kmeans-pytorch
pip install scipy==1.9.3
pip install pandas
pip install matplotlib

# Install argparse (usually part of the standard library, but included for completeness)
pip install argparse

echo "All dependencies have been installed successfully!"
