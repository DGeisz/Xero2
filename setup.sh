#!/bin/bash
cd Xero2

# Prevent auto tmux
touch ~/.no_auto_tmux

# Create directory for Miniconda and download the installer
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# Install Miniconda
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Remove the installer script
rm -rf ~/miniconda3/miniconda.sh

# Initialize Conda for Bash shell
~/miniconda3/bin/conda init bash

# Source the bashrc to refresh the environment
source ~/.bashrc

# Create a Conda environment
echo "Creating conda environment"
~/miniconda3/bin/conda create -n arena-env python=3.11 -y

# Add conda activation command to bashrc
echo "conda activate arena-env" >> ~/.bashrc

# Activate the Conda environment
source ~/miniconda3/bin/activate arena-env

git config --global user.email "dannygeisz@berkeley.edu"
git config --global user.name "Danny"
git config --global credential.helper store

pip install -r requirements.txt
