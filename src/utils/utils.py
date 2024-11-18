"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""


from data.dataloader import Dataloader
from data.scoreloader import Scoreloader
from models.model.convnet import ConvNet
from models.model.resnet import ResNetBaseline
from models.model.sit import SignalTransformer
from utils.config import model_parameters_file, supervised_weights_path, feature_data_path, data_dir

from torchinfo import summary
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import pickle
import math
from tsb_kit.vus.metrics import get_metrics
from multiprocessing import Pool


def load_data(raw_data_path, window_data_path, feature_data_path, selected_dataset_index, scores_path, features, split_file=None):
	"""
	Load segmented time series, ground truth, anomaly scores, and related data.

	Args:
	- raw_data_path (str): Path to the raw data.
	- window_data_path (str): Path to the window data.
	- selected_dataset_index (int): Index of the selected dataset.
	- scores_path (str): Path to the anomaly scores.

	Returns:
	- raw_anomalies (list): List of ground truth anomalies.
	- timeseries_names (list): List of time series names.
	- window_timeseries (DataFrame): DataFrame containing windowed time series.
	- scores (array): Anomaly scores of the time series.
	"""
	# Load the segmented time series and their ground truth, x - y
	raw_timeseries, raw_anomalies, timeseries_names, window_timeseries = load_timeseries(raw_data_path, window_data_path, feature_data_path, selected_dataset_index, features)
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

	# Only if a split has been given, then keep only certain time series
	if split_file is not None:
		split_data = pd.read_csv(split_file, index_col=0)
		if "val_set" in split_data.index:
			split_fnames = set([x[:-len('.csv')] for x in split_data.loc['val_set'].tolist() if not isinstance(x, float) or not math.isnan(x)])
		elif "test_set" in split_data.index:
			split_fnames = set([x[:-len('.csv')] for x in split_data.loc['test_set'].tolist() if not isinstance(x, float) or not math.isnan(x)])
		else:
			raise ValueError("This split file does not have the expected format")
		idx_to_remove = [i for i, x in enumerate(timeseries_names) if x not in split_fnames]

		if len(idx_to_remove):
			df_indexes_to_delete = [timeseries_names[i] for i in idx_to_remove]
			window_timeseries = window_timeseries[~window_timeseries.index.str.contains('|'.join(df_indexes_to_delete))]
			for idx in sorted(idx_to_remove, reverse=True):
				del timeseries_names[idx]
				del raw_anomalies[idx]
				del raw_timeseries[idx]
				del scores[idx]
	
	return raw_anomalies, timeseries_names, window_timeseries, scores


def load_timeseries(raw_data_path, window_data_path, feature_data_path, selected_index=None, features=False):
	
	# Create dataloader object
	dataloader = Dataloader(raw_data_path, window_data_path, feature_data_path)
	
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
	"""
    Predict anomalies in time series data using a specified model. 
	Use deep learning model for prediction if the model is a deep learning model,
    otherwise use classic model for prediction.

    Args:
    - model_name (str): Name of the model.
    - model (object): Trained model object.
    - df (DataFrame): DataFrame containing the time series data.
    - fnames (list): List of file names corresponding to the time series data.

    Returns:
    - predictions (array): Predicted anomalies for the time series data.
    """	
	return get_prediction_function(model_name)(model, df, fnames)

def get_prediction_function(model_name):
	"""
	Check if the model name corresponds to a deep learning model.

	Args:
	- model_name (str): Name of the model.

	Returns:
	- is_deep_learning (function): return the function to predict with
	
	Raises:
	- ValueError: If the model name is not recognized.
	"""
	deep_learning_models = ['convnet', 'resnet', 'sit']
	feature_models = ['knn']
	rocket_models = ['rocket']

	if model_name.lower() in deep_learning_models:
		return predict_deep
	elif model_name.lower() in feature_models:
		return predict_feature
	elif model_name.lower() in rocket_models:
		return predict_rocket
	else:
		raise ValueError("Unrecognized model name. Please provide a valid model name.")


def predict_deep(model, df, fnames):
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

def predict_feature(model, df, fnames):
	preds = []
	
	# Compute predictions and inference time
	for fname in tqdm(fnames, desc="Predicting", leave=False):
		# Df to tensor
		x = df.filter(like=fname, axis=0)
		curr_prediction = model.predict_proba(x)
		preds.append(curr_prediction)

	return preds

def predict_rocket(model, df, fnames):
	preds = []
	
	# Compute predictions and inference time
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


def compute_weighted_scores(window_pred_probabilities, scores, combine_method, k):
	"""
	Compute weighted anomaly scores for a given combine method and value of k.

	Args:
	- window_pred_probabilities (list of arrays): List of predicted probabilities for each window.
	- scores (array): Anomaly scores for the time series.
	- combine_method (str): Method to combine probabilities, either 'average' or 'vote'.
	- k (int): Value of k for combining probabilities.

	Returns:
	- weighted_scores (array): Weighted anomaly scores computed using the specified combine method and value of k.

	"""

	# Combine the predicted probabilities from multiple windows into a single vector
	if combine_method == 'average':
		weights = combine_probabilities_average(window_pred_probabilities, k)
	elif combine_method == 'vote':
		weights = combine_probabilities_vote(window_pred_probabilities, k)
	else:
		raise ValueError("Invalid combine_method. Choose either 'average' or 'vote'.")

	# Average the scores according to the weights
	weighted_scores = combine_anomaly_scores(scores, weights)

	return np.array(weights), weighted_scores


def compute_metrics(labels, scores, k=None):
	all_results = []

	for label, score in tqdm(zip(labels, scores), desc=f"Computing metrics {k}" if k is not None else "Computing metrics", leave=False, total=len(labels)):
		# Scikit-learn way to compute AUC-PR
		# precision, recall, _ = metrics.precision_recall_curve(label, score)
		# result = metrics.auc(recall, precision)

		# tsb-kit way to compute AUC-PR & VUS-PR
		result = wrapper_get_metrics((score, label))
		all_results.append(result)

	# Multi-processor & tsb-kit way to compute AUC-PR & VUS-PR
	# with Pool() as pool:
	# 	all_results = list(tqdm(pool.imap(wrapper_get_metrics, zip(scores, labels)), total=len(labels), desc=f"Computing metrics {k}" if k is not None else "Computing metrics", leave=False))

	return all_results


def wrapper_get_metrics(data_tuple):
	metrics_to_keep = ["AUC-ROC", "AUC-PR", "VUS-ROC", "VUS-PR"]

	score = data_tuple[0]
	label = data_tuple[1]

	result = get_metrics(score, label, metric="all", slidingWindow=10)

	return {key: result[key.replace('-', '_')] for key in metrics_to_keep}


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


def load_model(model_name, window_size, weights_path, unsupervised_extension=None):
	"""
	This function loads a machine learning model based on the specified model name and window size. It supports both deep learning models (e.g., ConvNet, ResNet) and non-deep models (e.g., KNN, ROCKET). It dynamically selects the appropriate loading method based on the model type.
	
	Args:
	- model_name (str): Name of the model to load.
	- window_size (int): Size of the window.

	Returns:
	- model: Loaded machine learning model.
	"""
	deep_models = ['convnet', 'resnet', 'sit']
	classic_models = ['knn', 'rocket']

	if model_name in deep_models:
		model_selector_dict = get_model_selector_dict(model_name, window_size, weights_path, unsupervised_extension)
		model = load_deep_model(**model_selector_dict)
	elif model_name in classic_models:
		model = load_classic_model(path=os.path.join(weights_path, f"{model_name}_{window_size}_{unsupervised_extension}" if unsupervised_extension else f"{model_name}_{window_size}"))
	else:
		raise ValueError(f"Model name \"{model_name}\" not expected")
	return model


def load_deep_model(model, model_parameters_file, weights_path, window_size):
	"""
	This function loads a deep learning model (e.g., ConvNet, ResNet) based on the provided model parameters file and weights path. It also adjusts the model's input size according to the specified window size. The function handles loading the model weights, considering whether CUDA is available or not.
	
	Args:
	- model (nn.Module): Deep learning model class.
	- model_parameters_file (str): File path to the model parameters JSON file.
	- weights_path (str): Directory or file path containing the model weights.
	- window_size (int): Size of the window.

	Returns:
	- model: Loaded deep learning model.
	"""
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


def get_model_selector_dict(model_name, window_size, weights_path, unsupervised_extension=None):
	"""
	Return a dictionary containing information for the model selector.

	Args:
	- model_name (str): Name of the model ('convnet', 'resnet', or 'sit').
	- window_size (int): Size of the window.

	Returns:
	- model_selector_dict (dict): Dictionary containing information for the model selector.
	"""
	model_parameters = {
		'convnet': {
			'model': ConvNet,
			'model_parameters_file': os.path.join(model_parameters_file, 'convnet_default.json'),
			'weights_path': os.path.join(weights_path, f'convnet_default_{window_size}_{unsupervised_extension}' if unsupervised_extension else f'convnet_default_{window_size}'),
			'window_size': window_size
		},
		'resnet': {
			'model': ResNetBaseline,
			'model_parameters_file': os.path.join(model_parameters_file, 'resnet_default.json'),
			'weights_path': os.path.join(weights_path, f'resnet_default_{window_size}_{unsupervised_extension}' if unsupervised_extension else f'resnet_default_{window_size}'),
			'window_size': window_size
		},
		'sit': {
			'model': SignalTransformer,
			'model_parameters_file': os.path.join(model_parameters_file, 'sit_stem_original.json'),
			'weights_path': os.path.join(weights_path, f'sit_stem_original_{window_size}_{unsupervised_extension}' if unsupervised_extension else f'sit_stem_original_{window_size}'),
			'window_size': window_size
		}
	}

	return model_parameters.get(model_name.lower())


def load_classic_model(path):
	"""
	This function loads a non-deep learning model (e.g., KNN, ROCKET) from a pickle (.pkl) file. 
	If the provided path is a directory, it loads the latest model within that directory based on the creation date.
	
	Args:
	- path (str): Path to the specific classifier to load or the directory containing classifiers.
	
	Returns:
	- output: Loaded non-deep learning model.
	"""
	
	# If model is not given, load the latest
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


def find_file_with_substring(directory_path, substring):
	"""
	Find a file within a directory that contains a specified substring in its name.

	Args:
	- directory_path (str): Path to the directory where the file should be searched.
	- substring (str): Substring to search for in the filenames.

	Returns:
	- file_path (str): Path to the file containing the specified substring.
	"""
	for file_name in os.listdir(directory_path):
		if substring in file_name and '.csv' in file_name:
			return os.path.join(directory_path, file_name)
	return None


def generate_window_data_path(window_size):
	if window_size not in [16, 32, 64, 128, 256, 512, 768, 1024]:
		raise ValueError(f"Window size {window_size} is not available")
	return os.path.join(data_dir, f"TSB_{window_size}")

def generate_feature_data_path(window_size):
	if window_size not in [16, 32, 64, 128, 256, 512, 768, 1024]:
		raise ValueError(f"Window size {window_size} is not available")
	return os.path.join(feature_data_path, f"TSFRESH_TSB_{window_size}.csv")


detector_names = [
	'AE', 
	'CNN', 
	'HBOS', 
	'IFOREST', 
	'IFOREST1', 
	'LOF', 
	'LSTM', 
	'MP', 
	'NORMA', 
	'OCSVM', 
	'PCA', 
	'POLY'
]