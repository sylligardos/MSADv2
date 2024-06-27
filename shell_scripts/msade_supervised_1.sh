#!/bin/bash
#SBATCH --job-name=msade_supervised_1        # Job name
#SBATCH --output=logs/%x_%j.log      # Standard output and error log
#SBATCH --error=logs/%x_errors_%j.log        # Error log
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=32        # Number of CPU cores per task
#SBATCH --mem=10G                # Memory per node
#SBATCH --partition=gpu          # Partition name (gpu for GPU jobs)
#SBATCH --gres=gpu:1             # Number of GPUs (1 in this case)
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=48:00:00          # Time limit hrs:min:sec

# Activate the conda environment
source activate MSAD-E

# Run the Python script
python3 src/main.py --experiment supervised --model_idx 1