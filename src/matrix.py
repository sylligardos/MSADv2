import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from data.scoreloader import Scoreloader
from data.dataloader import Dataloader
from utils.config import data_dir, scores_path, raw_data_path



def compute_metrics(labels, scores):
    """
    Compute the area under the precision-recall curve (AUC-PR) as a performance metric for anomaly detection.

    Parameters:
        labels (array-like): Ground truth binary labels indicating normal (0) or anomalous (1) instances.
        scores (array-like): Predicted anomaly scores or confidence scores for each instance.

    Returns:
        float: The area under the precision-recall curve (AUC-PR), which measures the trade-off between precision and recall.
               Higher values indicate better performance, with a maximum value of 1 indicating perfect precision and recall.
    """
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    result = metrics.auc(recall, precision)
    
    return result


def matrix_to_dataframe(matrix, fnames, detectors):
    """
    Convert a matrix of results to a pandas DataFrame.
    
    Parameters:
        matrix (numpy.ndarray): The matrix containing the results to be converted.
        fnames (list): A list of time series names corresponding to the rows of the matrix.
        detectors (list): A list of detector names corresponding to the columns of the matrix.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the matrix data, where each row represents a combination of time series and detector pair,
        and the columns include 'Time Series', 'Detector Pair', and 'AUC-PR'.
    """
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
    """
    Convert the DataFrame back to a matrix.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame to be converted.
        fnames (list): List of time series names.
        detectors (list): List of detector names.
        
    Returns:
        numpy.ndarray: The matrix containing the data from the DataFrame.
    """
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


def compute_matrix_and_save_results(data_dir, scores_path, save_dir):
    """
    Compute the matrix of performance metrics for anomaly detection for all datasets and save the results per dataset.

    Parameters:
        data_dir (str): The directory containing the raw datasets and features.
        scores_path (str): The path to the anomaly scores.

    Returns:
        None
    """
    dataloader = Dataloader(os.path.join(data_dir, "raw"), os.path.join(data_dir, "TSB_128"), os.path.join(data_dir, "features", "TSFRESH_TSB_128"))
    scoreloader = Scoreloader(scores_path)

    datasets = dataloader.get_dataset_names()
    detectors = scoreloader.get_detector_names()

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        # Load raw dataset, anomaly labels, and feature names
        x, y, fnames = dataloader.load_raw_dataset(dataset)
        
        # Read anomaly scores
        scores, idx_failed = scoreloader.load(fnames)

        # Remove failed samples
        if idx_failed:
            x = [x[i] for i in range(len(x)) if i not in idx_failed]
            y = [y[i] for i in range(len(y)) if i not in idx_failed]
            fnames = [fnames[i] for i in range(len(fnames)) if i not in idx_failed]

        n_detectors = len(detectors)
        n_timeseries = len(x)
        matrix = np.zeros((n_timeseries, n_detectors, n_detectors))

        for k in tqdm(range(n_timeseries), desc="Computing matrix"):
            for i in range(n_detectors):
                for j in range(n_detectors):
                    if i < j:
                        continue
                    elif i == j:
                        matrix[k][i][j] = compute_metrics(y[k], scores[k][:, i])
                    else:
                        matrix[k][i][j] = compute_metrics(y[k], np.average(scores[k][:, [i, j]], axis=1))
        
        # Save results to CSV
        df = matrix_to_dataframe(matrix, fnames, detectors)
        csv_filename = f"matrix_{dataset}.csv"
        df.to_csv(os.path.join(save_dir, csv_filename), index=False)
        print(f"Results saved to '{csv_filename}'.")
        break

def main():
	compute_matrix_and_save_results(data_dir, scores_path, os.path.join("reports", "matrices_04_2024"))



if __name__ == "__main__":
	main()
