#!bin/bash

module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Activate the new environment
source activate /gpfs/wolf2/olcf/trn040/scratch/mgaber/envs/hf-transformers

# Set environment variables
export HF_CACHE=/gpfs/wolf2/olcf/trn040/scratch/mgaber/hf_cache
export TOKENIZERS_PARALLELISM=false

# Loading the model and dataset on the login node
echo "Loading model and dataset on the login node..."
/gpfs/wolf2/olcf/trn040/scratch/mgaber/envs/hf-transformers/bin/python3.12 download.py

# Fine-tuning the model on the compute node
echo "Fine-tuning the model on the compute node..."
sbatch --export=NONE finetune.sh

echo "Finetuning job submitted to the compute node."
echo "You can minitor the job status using 'squeue --me' command."
