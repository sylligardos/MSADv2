"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""


import os
import numpy as np
import csv

from utils.config import raw_data_path, generate_window_data_path, model_parameters_file, weights_path, scores_path
from utils.utils import load_timeseries, load_model_selector, predict_timeseries, combine_probabilities_average, combine_probabilities_vote, load_anomaly_scores, combine_anomaly_scores, compute_metrics, save_results_to_csv
from models.model.convnet import ConvNet
from data.dataloader import Dataloader


def run_experiment(k_values, combine_methods, selected_dataset_index=None):
	window_size = 128
	results = {method: [] for method in combine_methods}
	window_data_path = generate_window_data_path(window_size)

	# SKIP NASA OMG ...
	dataloader = Dataloader(raw_data_path, window_data_path)
	datasets = dataloader.get_dataset_names()
	if "NASA" in datasets[selected_dataset_index]:
		return f"Skipping {datasets[selected_dataset_index]}"

	# Load the segmented time series and their ground truth, x - y
	raw_timeseries, raw_anomalies, timeseries_names, window_timeseries = load_timeseries(raw_data_path, window_data_path, selected_dataset_index)
	window_labels, window_timeseries = window_timeseries['label'], window_timeseries.drop('label', axis=1)
	
	# Load the 12 anomaly scores of the time series
	scores, idx_failed = load_anomaly_scores(scores_path, timeseries_names)
	if len(idx_failed) > 0:
		df_indexes_to_delete = [timeseries_names[i] for i in idx_failed]
		window_timeseries = window_timeseries[~window_timeseries.index.str.contains('|'.join(df_indexes_to_delete))]
		for idx in sorted(idx_failed, reverse=True):
			del timeseries_names[idx]
			del raw_anomalies[idx]
			del raw_timeseries[idx]

	# Load a model selector (obviously window size between time series and model selector should match)
	model = load_model_selector(
		model=ConvNet, 
		model_parameters_file=os.path.join(model_parameters_file, "convnet_default.json"), 
		weights_path=os.path.join(weights_path, "convnet_default_128"), 
		window_size=window_size
	)

	# Get the 12 number vector that is the predicted weights for each detector
	window_pred_probabilities = predict_timeseries(model, window_timeseries, timeseries_names)

	for k in k_values:
		for combine_method in combine_methods:
			# Compute Top-k weighted average of probabilities
			if combine_method == 'average':
				pred_probabilities = combine_probabilities_average(window_pred_probabilities, k)
			elif combine_method == 'to_vote':
				pred_probabilities = combine_probabilities_vote(window_pred_probabilities, k)
			else:
				raise ValueError("Invalid combine_method. Choose either 'average' or 'to_vote'.")

			# Average them according to the weights
			weighted_scores = combine_anomaly_scores(scores, pred_probabilities)

			# Compute the metrics that you want with the combined anomaly score and the ground truth
			results[combine_method].append(np.mean(compute_metrics(raw_anomalies, weighted_scores)))

	# Save the results dictionary to a CSV file
	save_results_to_csv(results, k_values, datasets[selected_dataset_index])

	return results

	# Save the result in a csv, e.g.:
	# Dataset | Time series | file name | window size | model selector | predicted weight AE | predicted weight NormA | ... | 
	# save_results()



def main():
	# k_values = np.arange(1, 13)  # Different values of k
	k_values = [1, 3, 5, 7]
	combine_methods = ['average', 'to_vote']  # Different ways to combine the probabilities
	selected_dataset_indexes = np.arange(13, 18)  # Index of the selected dataset

	for selected_dataset_index in selected_dataset_indexes:
		results = run_experiment(k_values, combine_methods, selected_dataset_index)
		print(results)
			


if __name__ == "__main__":
	main()
