#!bin/bash

# Loading the model and dataset on the login node
echo "Loading model and dataset on the login node..."
python download.py

# Fine-tuning the model on the compute node
echo "Fine-tuning the model on the compute node..."
sbatch --export=NONE fine-tune.sh

echo "Fine-tuning completed."