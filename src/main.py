"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""


import os
import numpy as np
import csv
import pandas as pd
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import time

from utils.config import raw_data_path, scores_path, supervised_split_file_path, unsupervised_split_file_path, supervised_weights_path, unsupervised_weights_path
from utils.utils import \
	load_data, load_model, predict_timeseries, compute_metrics, generate_window_data_path, generate_feature_data_path,  \
	compute_weighted_scores, find_file_with_substring, detector_names
from data.dataloader import Dataloader



def process_results(
	k_combine_method, 
	window_pred_probabilities, 
	scores, 
	raw_anomalies, 
	timeseries_names, 
	saving_dir, 
	model_name, 
	window_size, 
	testsize=None, 
	split=None,
	dataset=None
):
	# Unpack args
	k, combine_method = k_combine_method

	# Given the window prob. distr. compute weights and weighted score
	weights, weighted_scores = compute_weighted_scores(window_pred_probabilities, scores, combine_method, k)
	
	# Compute the metric value of the combined anomaly score
	metric_results = compute_metrics(raw_anomalies, weighted_scores, k)

	# Create dataframes with results
	weights_df = pd.DataFrame(weights, timeseries_names, columns=[f"weight_{x}" for x in detector_names])
	metric_results_df = pd.DataFrame(metric_results, timeseries_names)
	results_df = pd.concat([metric_results_df, weights_df], axis=1)
		
	# Decide filename based on experiment type
	if (testsize is not None and split is not None) and dataset is None:
		filename = f"testsize_{testsize}_split_{split}_{model_name}{window_size}_{combine_method}_k{k}.csv"
	elif dataset is not None and (testsize is None and split is None):
		filename = f"{dataset}_{model_name}{window_size}_{combine_method}_k{k}.csv"
	else:
		raise ValueError(f"Illegal case detected, please check {testsize} {split} {dataset}")

	# Save the results in a csv
	# results_df.to_csv(os.path.join(saving_dir, filename))
	
	return {
		"combine_method": combine_method,
		"k": k,
		"average_value": np.mean(metric_results_df['AUC-PR'].values),
	}


def run_experiment(k_values, combine_methods, selected_dataset_index, model_name, window_size, saving_dir):
	# Variables and setup
	results = []
	window_data_path = generate_window_data_path(window_size)
	feature_data_path = generate_feature_data_path(window_size)
	dataloader = Dataloader(raw_data_path, window_data_path, feature_data_path)
	datasets = dataloader.get_dataset_names()
	# split_file_path = find_file_with_substring(supervised_split_file_path, str(window_size))
	split_file_path = os.path.join(supervised_split_file_path, "split_TSB.csv")
	

	# SKIP NASA OMG ...
	if "NASA" in datasets[selected_dataset_index]:
		print( f"Skipping {datasets[selected_dataset_index]}")
		return None

	# Load the segmented time series, their ground truth, and their scores
	raw_anomalies, timeseries_names, window_timeseries, scores = load_data(
		raw_data_path, 
		window_data_path,
		feature_data_path,
		selected_dataset_index, 
		scores_path,
		(model_name=="knn"),
		split_file_path,
	)
	if not len(raw_anomalies):
		print( f"Skipping {datasets[selected_dataset_index]}")
		return None

	# Load a model selector (obviously window size between time series and model selector should match)	
	model = load_model(model_name, window_size, supervised_weights_path)

	# Get the 12 number vector that is the predicted weights for each detector
	window_pred_probabilities = predict_timeseries(model_name, model, window_timeseries, timeseries_names)

	# Prepare arguments and function for parallel processing
	k_combine_methods = [(k, combine_method) for k in k_values for combine_method in combine_methods]
	partial_process_results = partial(
		process_results, 
		window_pred_probabilities=window_pred_probabilities, 
		scores=scores,
		raw_anomalies=raw_anomalies, 
		timeseries_names=timeseries_names, 
		saving_dir=saving_dir, 
		model_name=model_name, 
		window_size=window_size, 
		testsize=None, 
		split=None,
		dataset=datasets[selected_dataset_index],
	)

	# Parallel processing
	# with Pool() as pool:
	# 	results = list(tqdm(pool.imap(partial_process_results, k_combine_methods), total=len(k_combine_methods), desc=f"Processing results", leave=False))

	# Single processing
	results = []
	for elem in k_combine_methods:
		curr_result = partial_process_results(elem)
		results.append(curr_result)

	return results


def run_unsupervised_experiment(k_values, combine_methods, selected_dataset_index, model_name, window_size, saving_dir, testsize, split):
	# Variables and setup
	results = []
	window_data_path = generate_window_data_path(window_size)
	feature_data_path = generate_feature_data_path(window_size)
	dataloader = Dataloader(raw_data_path, window_data_path, feature_data_path)
	datasets = dataloader.get_dataset_names()
	split_file_path = os.path.join(unsupervised_split_file_path, f"unsupervised_testsize_{testsize}_split_{split}.csv")

	# SKIP NASA OMG ...
	if selected_dataset_index and "NASA" in datasets[selected_dataset_index]:
		print( f"Skipping {datasets[selected_dataset_index]}")
		return None

	# Deciding dataset
	tmp_df = pd.read_csv(split_file_path, index_col=0)
	selected_dataset_index = datasets.index(os.path.split(tmp_df.loc['test_set'].iloc[0])[0])

	# Load the segmented time series, their ground truth, and their scores
	raw_anomalies, timeseries_names, window_timeseries, scores = load_data(
		raw_data_path, 
		window_data_path,
		feature_data_path,
		selected_dataset_index, 
		scores_path,
		(model_name=="knn"),
		split_file_path,
	)
	if not len(raw_anomalies):
		print( f"Skipping testsize {testsize} split {split}")
		return None

	# Load a model selector (obviously window size between time series and model selector should match)
	model = load_model(model_name, window_size, unsupervised_weights_path, f"testsize_{testsize}_split_{split}")

	# Get the 12 number vector that is the predicted weights for each detector
	window_pred_probabilities = predict_timeseries(model_name, model, window_timeseries, timeseries_names)

	# Prepare arguments and function for parallel processing
	k_combine_methods = [(k, combine_method) for k in k_values for combine_method in combine_methods]
	partial_process_results = partial(
		process_results, 
		window_pred_probabilities=window_pred_probabilities, 
		scores=scores,
		raw_anomalies=raw_anomalies, 
		timeseries_names=timeseries_names, 
		saving_dir=saving_dir, 
		model_name=model_name, 
		window_size=window_size, 
		testsize=testsize, 
		split=split,
		dataset=None,
	)

	# Parallel processing
	with Pool() as pool:
		results = list(tqdm(pool.imap(partial_process_results, k_combine_methods), total=len(k_combine_methods), desc=f"Processing results", leave=False))

	# Single processing
	# results = []
	# for elem in k_combine_methods:
	# 	curr_result = partial_process_results(elem)
	# 	results.append(curr_result)


	return results


def main(experiment, model_idx):
	# Setup variables
	saving_dir = os.path.join("reports", "results_06_2024_2")
	model_selectors = [("convnet", 128), ("resnet", 1024), ("sit", 512), ("knn", 1024)] 	# ("rocket", 128) is off for now
	k_values = np.arange(1, 13)
	selected_dataset_indexes = np.arange(0, 4)  # Index of the selected dataset
	combine_methods = ['average', 'vote']
	splits = np.arange(4, 16)

	if model_idx >= len(model_selectors):
			raise ValueError("Model selector index is not within range")

	start = time.time()

	if experiment == "supervised":
		print("--- MSAD-E Supervised experiments ---")
		for selected_dataset_index in selected_dataset_indexes:
			results = run_experiment(
				k_values, 
				combine_methods, 
				selected_dataset_index, 
				model_name=model_selectors[model_idx][0], 
				window_size=model_selectors[model_idx][1],
				saving_dir=saving_dir
			)

			end = time.time()

			if results:
				print(f'- dataset_id {selected_dataset_index} completed, time: {end - start} secs')

	elif experiment == "unsupervised":
		print("--- MSAD-E Unsupervised experiments ---")	
		for split in splits:
			results = run_unsupervised_experiment(
				k_values, 
				combine_methods, 
				selected_dataset_index=None, 
				model_name=model_selectors[model_idx][0], 
				window_size=model_selectors[model_idx][1],
				saving_dir=saving_dir,
				testsize=1,
				split=split
			)

			end = time.time()

			if results:
				print(f'- split {split} completed, time: {end - start} secs')

	else:
		raise ValueError(f"Experiment {experiment} is not an option...")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run experiments")
	
	parser.add_argument("--experiment", type=str, help="Type of experiment to conduct, namely supervised or unsupervised")
	parser.add_argument("--model_idx", type=int, help="Index of the model to use")

	args = parser.parse_args()

	main(
		experiment = args.experiment, 
		model_idx = args.model_idx,
	)


""" 
{
	"script_name": "main.py",
	"args": {
		"experiment": ["supervised", "unsupervised"],
		"model_idx": np.arange(0, 4),
	}
	"gpu_required": 0 if model_idx == 3 else 1
}
 """
