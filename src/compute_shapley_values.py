import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import multiprocessing

from data.scoreloader import Scoreloader
from data.dataloader import Dataloader
from utils.shapley_values import shapley_values
import argparse


def compute_shapley_values(experiment_dir):
    data_dir = 'data'
    scores_path = os.path.join(data_dir, 'scores')
    dataloader = Dataloader(
        os.path.join(data_dir, "raw"), 
        os.path.join(data_dir, "TSB_128"), 
        os.path.join(data_dir, "features", "TSFRESH_TSB_128")
    )
    scoreloader = Scoreloader(scores_path)

    datasets = dataloader.get_dataset_names()
    detectors = scoreloader.get_detector_names()

    for dataset in ['YAHOO']:
        if "NASA" in dataset:
            print( f"Skipping {dataset}")
            continue
        print(f"Processing {dataset}")
        
        x, y, fnames = dataloader.load_raw_dataset_parallel(dataset)
        scores, idx_failed = scoreloader.load_parallel(fnames)
        if idx_failed:
            x = [x[i] for i in range(len(x)) if i not in idx_failed]
            y = [y[i] for i in range(len(y)) if i not in idx_failed]
            fnames = [fnames[i] for i in range(len(fnames)) if i not in idx_failed]
        n_detectors = len(detectors)
        n_timeseries = len(x)
                
        # Prepare arguments for parallel processing
        args_list = [(y[k], scores[k].T) for k in range(n_timeseries)]
        num_procs = max(1, int(multiprocessing.cpu_count() * 2 / 3))
        with multiprocessing.Pool(processes=num_procs) as pool:
            shapley_results = list(tqdm(pool.imap(shapley_values, args_list), total=n_timeseries, desc="Computing auc"))
        shapley_results = np.stack(shapley_results)

        # Save results to CSV
        df = pd.DataFrame(shapley_results, index=fnames, columns=detectors)
        csv_filename = f"shapley_{dataset}.csv"
        
        save_path = os.path.join('experiments', experiment_dir)
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, csv_filename), index=True)
        print(f"Results saved to '{csv_filename}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute shapley values")
    parser.add_argument('--experiment', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()

    compute_shapley_values(
        experiment_dir=args.experiment,
    )