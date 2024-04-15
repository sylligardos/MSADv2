"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""


from data.dataloader import Dataloader
from data.scoreloader import Scoreloader

import numpy as np
import pandas as pd
import os
import torch
import json
from torch.utils.data import DataLoader
import torch.nn as nn
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
import csv


def load_timeseries(raw_data_path, window_data_path, selected_index=None):
	
	# Create dataloader object
	dataloader = Dataloader(raw_data_path, window_data_path)
	
	# Read available datasets
	datasets = dataloader.get_dataset_names()

	# If selected_index is provided, use only that dataset
	if selected_index is not None and 0 <= selected_index < len(datasets):
		selected_dataset = datasets[selected_index]
		datasets = [selected_dataset]

	# Load datasets
	x = []
	y = []
	fnames = []
	df_list = []
	for dataset in datasets:
		print(dataset)
		curr_x, curr_y, curr_fnames = dataloader.load_raw_dataset(dataset)
		curr_df = dataloader.load_window_timeseries(dataset)

		df_list.append(curr_df)				
		x.extend(curr_x)
		y.extend(curr_y)
		fnames.extend(curr_fnames)

	df = pd.concat(df_list)
	return x, y, fnames, df


def load_model_selector(model, model_parameters_file, weights_path, window_size):
	# Read models parameters
	model_parameters = load_json(model_parameters_file)
	
	# Change input size according to input
	if 'original_length' in model_parameters:
		model_parameters['original_length'] = window_size
	if 'timeseries_size' in model_parameters:
		model_parameters['timeseries_size'] = window_size

	# Load model
	model = model(**model_parameters)

	# Check if weights_path is specific file or dir
	if os.path.isdir(weights_path):
		# Check the number of files in the directory
		files = os.listdir(weights_path)
		if len(files) == 1:
			# Load the single file from the directory
			weights_path = os.path.join(weights_path, files[0])
		else:
			raise ValueError("Multiple files found in the 'model_path' directory. Please provide a single file or specify the file directly.")

	if torch.cuda.is_available():
		model.load_state_dict(torch.load(weights_path))
		model.eval()
		model.to('cuda')
	else:
		model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
		model.eval()

	return model

def predict_timeseries(model, df, fnames):
	preds = []
	tensor_softmax = nn.Softmax(dim=1)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# Compute predictions and inference time
	for fname in tqdm(fnames, desc="Predicting", leave=False):
		# Df to tensor
		x = df.filter(like=fname, axis=0)
		data_tensor = torch.tensor(x.values, dtype=torch.float32).unsqueeze(1).to(device)
		with torch.no_grad():
			curr_prediction = model(data_tensor)
			curr_prediction = tensor_softmax(curr_prediction)
		preds.append(curr_prediction.cpu().detach().numpy())

	return preds


def combine_probabilities_average(pred_probabilities, k):
	all_softmax_probabilities = []

	for probabilities in pred_probabilities:
		# Calculate mean of probabilities
		averaged_probabilities = np.mean(probabilities, axis=0)

		# Find indices of top-k elements
		top_k_indices = np.argsort(averaged_probabilities)[-k:]

		# Zero out elements not in top-k
		averaged_probabilities_filtered = np.zeros_like(averaged_probabilities)
		averaged_probabilities_filtered[top_k_indices] = averaged_probabilities[top_k_indices]

		# Normalize probabilities so that they sum to 1
		averaged_probabilities_filtered = averaged_probabilities_filtered/sum(averaged_probabilities_filtered)

		all_softmax_probabilities.append(averaged_probabilities_filtered)

	return all_softmax_probabilities


def combine_probabilities_vote(pred_probabilities, k):
	all_vote_probabilities = []

	for probabilities in pred_probabilities:
		num_classes = probabilities.shape[1]
		
		# Perform argmax operation along axis 0 to count the votes
		votes = np.argmax(probabilities, axis=1)
		vote_counts = np.bincount(votes, minlength=num_classes)
		
		# Create a probability distribution based on the votes
		vote_probabilities = vote_counts / np.sum(vote_counts)
	
		# Find indices of top-k elements
		top_k_indices = np.argsort(vote_probabilities)[-k:]

		# Zero out elements not in top-k
		vote_probabilities_filtered = np.zeros_like(vote_probabilities)
		vote_probabilities_filtered[top_k_indices] = vote_probabilities[top_k_indices]

		# Normalize probabilities so that they sum to 1
		vote_probabilities_filtered = vote_probabilities_filtered/sum(vote_probabilities_filtered)

		all_vote_probabilities.append(vote_probabilities_filtered)

	return all_vote_probabilities


def load_anomaly_scores(path, fnames):
	scoreloader = Scoreloader(path)

	scores, idx_failed = scoreloader.load(fnames)
	# if len(idx_failed) > 0:
	# 	raise ValueError("Some scores failed, take care")

	return scores, idx_failed


def combine_anomaly_scores(scores, weights):
	all_weighted_scores = []

	for score, weight in zip(scores, weights):
		num_time_series = score.shape[1]

		# Compute the weighted average score
		weighted_score = np.average(score, weights=weight, axis=1)

		# fig, axs = plt.subplots(num_time_series + 1, 1, figsize=(10, 6), sharey=True)

		# # Plot each individual time series in the score vector
		# for i in range(num_time_series):
		# 	axs[i].plot(score[:, i]*weight[i], label=f'Time Series {i+1}')
		# 	axs[i].set_ylabel(f'{weight[i]:.2f}')
		# 	axs[i].legend
		# 	axs[i].set_ylim(0, 1)
		
		# # Plot the weighted averaged score
		# axs[num_time_series].plot(weighted_score, label='Weighted Score')
		# axs[num_time_series].set_xlabel('Index')
		# axs[num_time_series].legend()
		# axs[num_time_series].set_ylim(0, 1)
		# plt.show()

		all_weighted_scores.append(weighted_score)

	
	return all_weighted_scores



def compute_metrics(labels, scores):
	all_results = []

	for label, score in zip(labels, scores):
		precision, recall, _ = metrics.precision_recall_curve(label, score)
		result = metrics.auc(recall, precision)
		all_results.append(result)

	return all_results

def save_results_to_csv(results, k_values, dataset):
	filename = f"{dataset}.csv"
	with open(filename, 'w', newline='') as csvfile:
		fieldnames = ['Combine Method'] + [f'k={k}' for k in k_values]
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()
		for combine_method, result_list in results.items():
			writer.writerow({**{'Combine Method': combine_method}, **{f'k={k}': result for k, result in zip(k_values, result_list)}})


def load_json(file_path):
	"""
	Load JSON data from a file.

	Args:
		file_path (str): Path to the JSON file.

	Returns:
		dict: Dictionary containing the loaded JSON data.
	"""
	try:
		with open(file_path, 'r') as file:
			data = json.load(file)
		return data
	except FileNotFoundError:
		print("File not found.")
		return None
	except json.JSONDecodeError as e:
		print(f"Error decoding JSON: {e}")
		return None
	except Exception as e:
		print(f"Error loading JSON: {e}")
		return None
