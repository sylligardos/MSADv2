"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""


import numpy as np
import pandas as pd
import os
import torch
import json
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import pickle
import math
from tsb_kit.vus.metrics import get_metrics

from data.dataloader import Dataloader
from data.scoreloader import Scoreloader
from models.model.convnet import ConvNet
from models.model.resnet import ResNetBaseline
from models.model.sit import SignalTransformer


def load_data(data_path, window, dataset=None, features=False, split_file=None):
	# Change this according to your data structure
	raw_path = os.path.join(data_path, 'raw')
	window_path = os.path.join(data_path, f'TSB_{window}')
	feature_path = os.path.join(data_path, 'features', f'TSFRESH_TSB_{window}.csv')
	scores_path = os.path.join(data_path, 'scores')
	if split_file is not None:
		split_file_path = os.path.join(data_path, 'splits', 'unsupervised' if 'unsupervised' in split_file else 'supervised')
		split_file = os.path.join(split_file_path, split_file) + '.csv'

	raw_timeseries, raw_anomalies, timeseries_names, window_timeseries = load_timeseries(raw_path, window_path, feature_path, dataset, features)
	window_labels, window_timeseries = window_timeseries['label'], window_timeseries.drop('label', axis=1)

	scores, idx_failed = load_anomaly_scores(scores_path, timeseries_names)
	if len(idx_failed) > 0:
		df_indexes_to_delete = [timeseries_names[i] for i in idx_failed]
		window_timeseries = window_timeseries[~window_timeseries.index.str.contains('|'.join(df_indexes_to_delete))]
		for idx in sorted(idx_failed, reverse=True):
			del timeseries_names[idx]
			del raw_anomalies[idx]
			del raw_timeseries[idx]

	if split_file is not None:
		split_data = pd.read_csv(split_file, index_col=0)
		split_fnames = set([x[:-len('.csv')] for x in split_data.loc['test_set'].tolist() if not isinstance(x, float) or not math.isnan(x)])
		idx_to_keep = [i for i, x in enumerate(timeseries_names) if x in split_fnames]

		timeseries_names = [timeseries_names[i] for i in idx_to_keep]
		raw_anomalies = [raw_anomalies[i] for i in idx_to_keep]
		raw_timeseries = [raw_timeseries[i] for i in idx_to_keep]
		scores = [scores[i] for i in idx_to_keep]
		window_timeseries = window_timeseries[window_timeseries.index.str.contains('|'.join(timeseries_names))]
	
	return raw_anomalies, timeseries_names, window_timeseries, scores


def load_timeseries(raw_path, window_path, feature_path, datasets=None, features=False):	
	dataloader = Dataloader(raw_path, window_path, feature_path)
	datasets = dataloader.get_dataset_names() if datasets is None else datasets

	x = []
	y = []
	fnames = []
	df_list = []
	for dataset in datasets:
		curr_x, curr_y, curr_fnames = dataloader.load_raw_dataset_parallel(dataset)
		curr_df = dataloader.load_window_timeseries_parallel(dataset) if not features else dataloader.load_feature_timeseries(dataset)

		if curr_df is None:
			continue
		df_list.append(curr_df)				
		x.extend(curr_x)
		y.extend(curr_y)
		fnames.extend(curr_fnames)

	df = pd.concat(df_list)
	return x, y, fnames, df


def load_anomaly_scores(path, fnames):
	scoreloader = Scoreloader(path)
	scores, idx_failed = scoreloader.load_parallel(fnames)
	return scores, idx_failed


def predict_timeseries(model_name, model, df, fnames):
	return get_prediction_function(model_name)(model, df, fnames)

def get_prediction_function(model_name):
	if any([True for x in ['convnet', 'resnet', 'sit'] if x in model_name]):
		return predict_deep
	elif 'knn' in model_name:
		return predict_feature
	elif 'rocket' in model_name:
		return predict_rocket
	else:
		raise ValueError("Unrecognized model name. Please provide a valid model name.")


def predict_deep(model, df, fnames):
	preds = []
	tensor_softmax = nn.Softmax(dim=1)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	for fname in tqdm(fnames, desc="Predicting", leave=False):
		x = df.filter(like=fname, axis=0)
		data_tensor = torch.tensor(x.values, dtype=torch.float32).unsqueeze(1).to(device)
		
		with torch.no_grad():
			curr_prediction = model(data_tensor)
			curr_prediction = tensor_softmax(curr_prediction)
		preds.append(curr_prediction.cpu().detach().numpy())

	return preds

def predict_feature(model, df, fnames):
	preds = []
	
	for fname in tqdm(fnames, desc="Predicting", leave=False):
		x = df.filter(like=fname, axis=0)
		curr_prediction = model.predict_proba(x)
		preds.append(curr_prediction)

	return preds

def predict_rocket(model, df, fnames):
	preds = []
	
	for fname in tqdm(fnames, desc="Predicting", leave=False):
		x = df.filter(like=fname, axis=0).values[:, np.newaxis]
		curr_prediction = model.decision_function(x)
		preds.append(curr_prediction)

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

		# # This is the same and faster just saying ;)
		# bottom_indices = np.argsort(vote_counts)[:k]
		# vote_counts[bottom_indices] = 0
		# all_vote_probabilities.append(vote_counts/sum(vote_counts))
		
		all_vote_probabilities.append(vote_probabilities_filtered)

	return all_vote_probabilities


def combine_anomaly_scores(scores, weights, plot=False):
	all_weighted_scores = []

	for score, weight in zip(scores, weights):
		num_time_series = score.shape[1]

		# Compute the weighted average score
		weighted_score = np.average(score, weights=weight, axis=1)

		if plot:
			detector_names = Scoreloader(os.path.join('data', 'scores')).get_detector_names()
			fig, axs = plt.subplots(num_time_series + 1, 1, figsize=(10, 6), sharey=True, sharex=True)

			for i in range(num_time_series):
				axs[i].plot(score[:, i] * weight[i], label=f'Time Series {i+1}')
				axs[i].set_ylabel(f'{detector_names[i]}\n{weight[i]:.2f}')
				axs[i].legend
				axs[i].set_ylim(0, 1)
				axs[i].set_xticks([])
				axs[i].set_yticks([])	
			
			# Plot the weighted averaged score
			axs[num_time_series].plot(weighted_score)
			axs[num_time_series].set_xlabel('Index')
			axs[num_time_series].set_ylabel('Combined\nscore')
			axs[num_time_series].set_ylim(0, 1)
			axs[num_time_series].set_yticks([])
			plt.tight_layout()
			plt.show()

		all_weighted_scores.append(weighted_score)

	
	return all_weighted_scores


def compute_weighted_scores(window_pred_probabilities, combination_method, k):
	if combination_method == 'average':
		weights = combine_probabilities_average(window_pred_probabilities, k)
	elif combination_method == 'vote':
		weights = combine_probabilities_vote(window_pred_probabilities, k)
	else:
		raise ValueError("Invalid combination_method. Choose either 'average' or 'vote'.")

	return np.array(weights)


def compute_metrics(labels, scores, k=None):
	all_results = []

	for label, score in tqdm(zip(labels, scores), desc=f"Computing metrics {k}" if k is not None else "Computing metrics", leave=False, total=len(labels)):
		result = wrapper_get_metrics(score, label)
		all_results.append(result)

	return all_results


def wrapper_get_metrics(score, label):
	metrics_to_keep = ["AUC-ROC", "AUC-PR", "VUS-ROC", "VUS-PR"]
	result = get_metrics(score, label, metric="all", slidingWindow=10)

	return result # {key: result[key.replace('-', '_')] for key in metrics_to_keep}

def load_json(file_path):
	with open(file_path, 'r') as file:
		data = json.load(file)
	return data

def load_model(model_name, window_size, weights_path):
	if any([True for x in ['convnet', 'resnet', 'sit'] if x in model_name]):
		model_parameters_file = os.path.join('src', 'models', 'configuration', model_name.replace(f"_{window_size}", '.json'))
		model_class = get_model_selector_dict(model_name)
		model = load_deep_model(model_class, model_parameters_file, weights_path, window_size)
	elif any([True for x in ['knn', 'rocket'] if x in model_name]):
		model = load_classic_model(path=weights_path)
	else:
		raise ValueError(f"Model {model_name} not valid")
	return model


def load_deep_model(model, model_parameters_file, weights_path, window_size):
	model_parameters = load_json(model_parameters_file)
	
	if 'original_length' in model_parameters:
		model_parameters['original_length'] = window_size
	if 'timeseries_size' in model_parameters:
		model_parameters['timeseries_size'] = window_size

	model = model(**model_parameters)

	if os.path.isdir(weights_path):
		files = os.listdir(weights_path)
		if len(files) == 1:
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

def get_model_selector_dict(model_name):
	model_parameters = {
		'convnet': ConvNet,
		'resnet': ResNetBaseline,
		'sit': SignalTransformer,
	}
	return model_parameters[model_name.split('_')[0]]


def load_classic_model(path):
	if os.path.isdir(path):
		models = [x for x in os.listdir(path) if '.pkl' in x]
		models.sort(key=lambda date: datetime.strptime(date, 'model_%d%m%Y_%H%M%S.pkl'))
		path = os.path.join(path, models[-1])
	elif '.pkl' not in path:
		raise ValueError(f"Can't load this type of file {path}. Only '.pkl' files please")

	filename = Path(path)
	with open(f'{filename}', 'rb') as input:
		output = pickle.load(input)
	
	return output