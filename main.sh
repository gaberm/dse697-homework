#!bin/bash

module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Create a new conda environment with Python 3.12.0
conda create -p /gpfs/wolf2/olcf/trn040/scratch/mgaber/envs/hf-transformers python=3.12.0 -y

# Activate the new environment
source activate /gpfs/wolf2/olcf/trn040/scratch/mgaber/envs/hf-transformers

# Install the required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
conda install conda-forge::transformers -y
pip install wandb
pip install peft
pip install scikit-learn
pip install trl
pip install bitsandbytes

# Loading the model and dataset on the login node
echo "Loading model and dataset on the login node..."
python download.py

# Fine-tuning the model on the compute node
echo "Fine-tuning the model on the compute node..."
sbatch --export=NONE fine-tune.sh

echo "Fine-tuning completed."
