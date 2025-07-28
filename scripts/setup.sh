#!/bin/bash

set -e

echo "Setting up the environment..."

# STEP 1: Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev \
#    These packages aren't necessary (I think), Only install them if you faced issues with building the gym wheels.
#    build-essential cmake zlib1g-dev \ 
#    libsdl2-dev libportmidi-dev libfreetype6-dev libjpeg-dev libpng-dev \ 
#    libgl1 libgtk-3-dev



# STEP 2: Create a virtual environment
echo "Creating a virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

# STEP 3: Installing the project's packages
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Setup complete!"