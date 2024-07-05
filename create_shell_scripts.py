"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""

import itertools
import os


def main():
    saving_dir = "shell_scripts"
    experiment_desc = {
        "job_name": "msade",
        "environment": "MSAD-E",
        "script_name": "src/main.py",
        "args": {
            "experiment": ["supervised", "unsupervised"],
            "model_idx": [0, 1, 2, 3],
        },
        "gpu_required": "0 if model_idx == 3 else 1"
    }
    sh_file_templates = [
"""#!/bin/bash
#SBATCH --job-name={}        # Job name
#SBATCH --output=logs/%x_%j.log      # Standard output and error log
#SBATCH --error=logs/%x_errors_%j.log        # Error log
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=32        # Number of CPU cores per task
#SBATCH --mem=32G                # Memory per node
#SBATCH --time=48:00:00          # Time limit hrs:min:sec

# Activate the conda environment
source activate {}

# Run the Python script
python3 {}""",
"""#!/bin/bash
#SBATCH --job-name={}        # Job name
#SBATCH --output=logs/%x_%j.log      # Standard output and error log
#SBATCH --error=logs/%x_errors_%j.log        # Error log
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=32        # Number of CPU cores per task
#SBATCH --mem=32G                # Memory per node
#SBATCH --partition=gpu          # Partition name (gpu for GPU jobs)
#SBATCH --gres=gpu:1             # Number of GPUs (1 in this case)
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=48:00:00          # Time limit hrs:min:sec

# Activate the conda environment
source activate {}

# Run the Python script
python3 {}"""
    ]
    
    # Analyse json
    environment = experiment_desc["environment"]
    script_name = experiment_desc["script_name"]
    args = experiment_desc["args"]
    arg_names = list(args.keys())
    arg_values = list(args.values())
    gpu_required = experiment_desc["gpu_required"]

    # Generate all possible combinations of arguments
    combinations = list(itertools.product(*arg_values))
    
    # Create the commands
    jobs = []
    for combination in combinations:
        cmd = f"{script_name}"
        gpu_required = experiment_desc["gpu_required"]
        job_name = experiment_desc["job_name"]

        for name, value in zip(arg_names, combination):
            cmd += f" --{name} {value}"
            job_name += f"_{value}"

            if isinstance(gpu_required, str) and name in gpu_required:
                gpu_required = int(eval(gpu_required.replace(name, str(value))))

        # Write the .sh file
        with open(os.path.join(saving_dir, f'{job_name}.sh'), 'w') as rsh:
            rsh.write(sh_file_templates[gpu_required].format(job_name, environment, cmd))
        
        jobs.append(job_name)

    # Create sh file to conduct all experiments 
    run_all_sh = ""
    for job in jobs:
        run_all_sh += f"sbatch {os.path.join(saving_dir, f'{job}.sh')}\n"
    
    with open(os.path.join(saving_dir, f'conduct_{experiment_desc["job_name"]}.sh'), 'w') as rsh:
        rsh.write(run_all_sh)
        

if __name__ == "__main__":
    main()