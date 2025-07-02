#!/bin/bash
#SBATCH --job-name=revision_sit_split_TSB                   # Job name
#SBATCH --output=experiments/revision/logs/%x.log         # Standard output and error log
#SBATCH --error=experiments/revision/logs/%x.log          # Error log
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --mem=128G                       # Memory per node
#SBATCH --partition=gpu                 # Partition name (gpu for GPU jobs)
#SBATCH --gres=gpu:h100:1               # Number of GPUs (1 in this case)
#SBATCH --time=10:00:00                  # Time limit hrs:min:sec

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MSAD-E

# Run the Python script
python3 src/run_model.py --model sit --split split_TSB --experiment revision