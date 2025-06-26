import pandas as pd
import numpy as np
import os
from sklearn import metrics
from tqdm import tqdm
from multiprocessing import Pool

from data.scoreloader import Scoreloader
from data.dataloader import Dataloader
from utils.config import data_dir, scores_path, raw_data_path
import argparse
from functools import partial



def compute_metrics(labels, scores):
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    return metrics.auc(recall, precision)


def matrix_to_dataframe(matrix, fnames, detectors):
    data = []
    for k in tqdm(range(matrix.shape[0]), desc="Dataframe"):
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[2]):
                # Only store the upper triangle (including the diagonal) of the matrix
                if i >= j:
                    data.append([f'{fnames[k]}', f'{detectors[i]}-{detectors[j]}', matrix[k][i][j]])
    
    df = pd.DataFrame(data, columns=['Time Series', 'Detector Pair', 'AUC-PR'])
    return df



def dataframe_to_matrix(df, fnames, detectors):
    n_time_series = len(fnames)
    n_detectors = len(detectors)
    
    matrix = np.zeros((n_time_series, n_detectors, n_detectors))
    
    for idx, row in df.iterrows():
        ts_idx = fnames.index(row['Time Series'])
        det_pair = row['Detector Pair'].split('-')
        det1_idx = detectors.index(det_pair[0])
        det2_idx = detectors.index(det_pair[1])
        
        matrix[ts_idx][det1_idx][det2_idx] = row['Result']
    
    return matrix

def compute_auc_multiple(args):
    n_detectors, y, scores = args

    compute_metrics_with_y = partial(compute_metrics, y)
    result = np.apply_along_axis(func1d=compute_metrics_with_y, axis=0, arr=scores)
    return result

def compute_matrix_for_dataset(args):
    n_detectors, y, scores = args
    matrix = np.zeros((n_detectors, n_detectors))
    for i in range(n_detectors):
        for j in range(n_detectors):
            if i < j:
                continue
            elif i == j:
                matrix[i][j] = compute_metrics(y, scores[:, i])
            else:
                matrix[i][j] = compute_metrics(y, np.average(scores[:, [i, j]], axis=1))
    return matrix


def compute_matrix_and_save_results(experiment_dir, experiment_type):
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

    for dataset in datasets:
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
        args_list = [(n_detectors, y[k], scores[k]) for k in range(n_timeseries)]
        with Pool() as pool:
            if experiment_type == 'correlation':
                matrices = list(tqdm(pool.imap(compute_auc_multiple, args_list), total=n_timeseries, desc="Computing auc"))
            elif experiment_type == 'combination':
                matrices = list(tqdm(pool.imap(compute_matrix_for_dataset, args_list), total=n_timeseries, desc="Computing matrix"))
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")
        matrix = np.stack(matrices)

        # Save results to CSV
        if experiment_type == 'correlation':
            df = pd.DataFrame(matrix, index=fnames, columns=detectors)
            csv_filename = f"auc_{dataset}.csv"
        elif experiment_type == 'combination':
            df = matrix_to_dataframe(matrix, fnames, detectors)
            csv_filename = f"matrix_{dataset}.csv"
        save_path = os.path.join('experiments', experiment_dir)
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, csv_filename), index=(experiment_type == 'correlation'))
        print(f"Results saved to '{csv_filename}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save detector pairwise AUC-PR matrices.")
    parser.add_argument('--experiment', type=str, required=True, help='Directory to save results')
    parser.add_argument('--type', type=str, required=True, help='Type of experiment (e.g., correlation, combination)')
    args = parser.parse_args()

    compute_matrix_and_save_results(
        experiment_dir=args.experiment,
        experiment_type=args.type,
    )
