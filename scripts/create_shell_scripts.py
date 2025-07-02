"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""

from templates import sh_templates

import itertools
import os
from copy import deepcopy
import re
import numpy as np
from tqdm import tqdm


def natural_keys(text):
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

experiments = {
    "revision": {
        "environment": "MSAD-E",
        "script_name": "src/run_model.py",
        "template": 'cleps_gpu',
        "args": {
            "model": ['convnet', 'resnet', 'sit', 'knn'],
            "split": ['split_TSB'],
        },
        "rules": []
    },
    "synthetic_data_generation": {
        "environment": "ffvus",
        "script_name": "src/generate_synthetic.py",
        "template": 'cleps_cpu',
        "args": {
            "save_dir": [f"syn_{i}" for i in range(10)],
        },
        "rules": []
    },
    "vus_ffvus_auc_0": {
        "environment": "ffvus",
        "script_name": "src/compute_metric.py",
        "template": 'cleps_gpu',
        "args": {
            "dataset": ['tsb'], # + os.listdir(os.path.join('data', 'synthetic')),
            "metric": ['ff_vus_pr_gpu'], #, 'rf', 'affiliation', 'range_auc_pr', 'auc_pr', 'vus_pr', 'ff_vus_pr_gpu'
            "slope_size": [0], # [0, 16, 32, 64, 128, 256],
            "step":  [1],
            # "slopes": ['function'], #, 'function'
            "existence": ['None'], #, 'matrix'
            # "conf_matrix": ['dynamic_plus'],
        },
        "rules": []
    },
    "vus_ffvus_auc_0_syn": {
        "environment": "ffvus",
        "script_name": "src/compute_metric.py",
        "template": None,
        "args": {
            "dataset": ['all_synthetic'],
            "metric": ['ff_vus_pr_gpu', 'auc_pr', 'ff_vus_pr', 'vus_pr'], #, 'rf', 'affiliation', 'range_auc_pr', 'auc_pr', 'vus_pr', 'ff_vus_pr_gpu'
            "slope_size": [0], # [0, 16, 32, 64, 128, 256],
            "step":  [1],
            # "slopes": ['function'], #, 'function'
            "existence": ['None'], #, 'matrix'
            # "conf_matrix": ['dynamic_plus'],
        },
        "rules": []
    },
    "allmetrics_defparams_tsb": {
        "environment": "ffvus",
        "script_name": "src/compute_metric.py",
        "template": None,
        "args": {
            "dataset": ['tsb'],
            "metric": ['ff_vus_pr', 'rf', 'affiliation', 'range_auc_pr', 'auc_pr', 'vus_pr', 'ff_vus_pr_gpu']
        },
        "rules": []
    },
    "allmetrics_defparams_syn": {
        "environment": "ffvus",
        "script_name": "src/compute_metric.py",
        "template": None,
        "args": {
            "dataset": ['all_synthetic'],
            "metric": ['ff_vus_pr', 'rf', 'affiliation', 'range_auc_pr', 'auc_pr', 'vus_pr', 'ff_vus_pr_gpu']
        },
        "rules": []
    },
    "vus_buffer_comparison_tsb":{
        "environment": "ffvus",
        "script_name": "src/compute_metric.py",
        "template": None,
        "args": {
            "dataset": ['tsb'],
            "metric": ['ff_vus_pr_gpu', 'ff_vus_pr', 'vus_pr'], #, 'rf', 'affiliation', 'range_auc_pr', 'auc_pr', 'vus_pr', 'ff_vus_pr_gpu'
            "slope_size": [0, 2, 4, 8, 16, 32, 64, 128, 256, 512], 
            # "step":  [1],
        },
        "rules": [],
    },
    "vus_step_comparison_tsb": {
        "environment": "ffvus",
        "script_name": "src/compute_metric.py",
        "template": None,
        "args": {
            "dataset": ['tsb'],
            "metric": ['ff_vus_pr_gpu', 'ff_vus_pr'], #, 'rf', 'affiliation', 'range_auc_pr', 'auc_pr', 'vus_pr', 'ff_vus_pr_gpu'
            "slope_size": [512], 
            "step":  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        },
        "rules": [],
    }
}

def create_shell_scripts():
    parent_dir = "scripts"
    experiment_name = "revision"

    logs_saving_dir = os.path.join("experiments", experiment_name)
    os.makedirs(logs_saving_dir, exist_ok=True)
    experiment_desc = experiments[experiment_name]
    
    # Analyse json
    job_name = experiment_name
    saving_dir = os.path.join(parent_dir, job_name)
    environment = experiment_desc["environment"]
    script_name = experiment_desc["script_name"]
    args = experiment_desc["args"]
    arg_names = list(args.keys())
    arg_values = list(args.values())
    rules = experiment_desc['rules']
    
    jobs = set()
    os.makedirs(saving_dir, exist_ok=True)
    combinations = list(itertools.product(*arg_values))
    for combination in tqdm(combinations):
        curr_cmd = script_name
        curr_job_name = job_name
        curr_rules = deepcopy(rules)

        for name, value in zip(arg_names, combination):
            curr_cmd += f" --{name} {value}"
            curr_job_name += f"_{value}"
            for i in range(len(curr_rules)):
                curr_rules[i] = curr_rules[i].replace(name, str(value))
        
        curr_cmd += f" --experiment {experiment_name}"

        # Evaluate rules to see if we accept this combination, if not skip
        rules_evaluation = [eval(rule) for rule in curr_rules]
        if not all(rules_evaluation):
            continue
            
        # Fill in template and write the .sh file
        if experiment_desc["template"] is None:
            template = sh_templates['cleps_gpu'] if 'gpu' in curr_cmd else sh_templates['cleps_cpu']
        else:
            template = sh_templates[experiment_desc["template"]]
        with open(os.path.join(saving_dir, f'{curr_job_name}.sh'), 'w') as rsh:
            rsh.write(template.format(curr_job_name, logs_saving_dir, logs_saving_dir, environment, curr_cmd))
        
        jobs.add(curr_job_name)

    # Create sh file to conduct all experiments 
    run_all_sh = ""
    jobs = list(jobs)
    jobs.sort(key=natural_keys)
    for job in jobs:
        run_all_sh += f"sbatch {os.path.join(saving_dir, f'{job}.sh')}\n"
    
    with open(os.path.join(saving_dir, f'conduct_{job_name}.sh'), 'w') as rsh:
        rsh.write(run_all_sh)
        

if __name__ == "__main__":
    create_shell_scripts()
