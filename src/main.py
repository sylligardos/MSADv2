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

from utils.config import raw_data_path, scores_path, split_file_path
from utils.utils import \
	load_data, load_model, predict_timeseries, compute_metrics, generate_window_data_path, generate_feature_data_path,  \
	compute_weighted_scores, find_file_with_substring, detector_names
from data.dataloader import Dataloader


def run_experiment(k_values, combine_methods, selected_dataset_index, model_name, window_size):
	
	# Variables and setup
	results = []
	window_data_path = generate_window_data_path(window_size)
	feature_data_path = generate_feature_data_path(window_size)
	dataloader = Dataloader(raw_data_path, window_data_path, feature_data_path)
	datasets = dataloader.get_dataset_names()
	metric="AUC-PR"
	saving_dir = os.path.join("reports", "aucpr_proba_04_2024")

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
		find_file_with_substring(split_file_path, str(window_size))
	)
	if not len(raw_anomalies):
		print( f"Skipping {datasets[selected_dataset_index]}")
		return None

	# Load a model selector (obviously window size between time series and model selector should match)
	model = load_model(model_name, window_size)

	# Get the 12 number vector that is the predicted weights for each detector
	window_pred_probabilities = predict_timeseries(model_name, model, window_timeseries, timeseries_names)

	for k in k_values:
		for combine_method in combine_methods:
			weights, weighted_scores = compute_weighted_scores(window_pred_probabilities, scores, combine_method, k)
			print(weights)
			exit()
			
			# Compute the metric value of the combined anomaly score
			metric_results = compute_metrics(raw_anomalies, weighted_scores)

			# Save the results dictionary to a CSV file
			weights_df = pd.DataFrame(weights, timeseries_names, columns=[f"weight_{x}" for x in detector_names])
			metric_results_df = pd.DataFrame(metric_results, timeseries_names, columns=[metric])
			results_df = pd.concat([metric_results_df, weights_df], axis=1)
			results_df.to_csv(os.path.join(saving_dir, f"{datasets[selected_dataset_index]}_{model_name}{window_size}_{combine_method}_k{k}.csv"))

			results.append({
				"combine_method": combine_method,
				"k": k,
				"average_value": np.mean(metric_results),
			})

	return results


def main():
	k_values = np.arange(1, 13)
	selected_dataset_indexes = np.arange(0, 18)  # Index of the selected dataset
	model_selectors = [("resnet", 1024), ("convnet", 128), ("sit", 512), ("knn", 1024)] 	# ("rocket", 128) is off for now
	combine_methods = ['average', 'vote']
	model_idx = 3

	for selected_dataset_index in selected_dataset_indexes:
		results = run_experiment(
			k_values, 
			combine_methods, 
			selected_dataset_index, 
			model_name=model_selectors[model_idx][0], 
			window_size=model_selectors[model_idx][1])

		if results is None: continue

		for result in results:
			if result["combine_method"] == 'average':
				print(result)
		for result in results:
			if result["combine_method"] == 'vote':
				print(result)


if __name__ == "__main__":
	main()
