"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st Year (2024)
@what: Weighted Average MSAD
"""



import os
import pandas as pd
from tqdm import tqdm
import glob
import numpy as np


class Dataloader:
	"""A class for loading the data"""

	def __init__(self, raw_data_path, window_data_path, feature_data_path):
		"""
		Initialize the Dataloader.

		Args:
			dataset_dir (str): Path to the directory containing the dataset.
		"""
		self.raw_data_path = raw_data_path
		self.window_data_path = window_data_path
		self.feature_data_path = feature_data_path


	def get_dataset_names(self):
		"""
		Get the names of all datasets in the dataset directory.

		Returns:
			list: A list of dataset names.
		"""
		
		names = os.listdir(self.raw_data_path)

		return [x for x in names if os.path.isdir(os.path.join(self.raw_data_path, x))]
		

	def load_raw_dataset(self, dataset):
		"""
		Load the raw time series from the given datasets

		Args:
			dataset (str): Name of the dataset to load
		
		Returns:

		"""
		x = []
		y = []
		fnames = []

		# Check if dataset exists
		if dataset not in self.get_dataset_names():
			raise ValueError(f'Dataset {dataset} not in dataset list')
		path = os.path.join(self.raw_data_path, dataset)

		# Load file names
		timeseries = [f for f in os.listdir(path) if f.endswith('.out')]

		# Load time series
		for curr_timeseries in tqdm(timeseries, desc="Loading time series", leave=False):
			curr_data = pd.read_csv(os.path.join(path, curr_timeseries), header=None).to_numpy()
			
			if curr_data.ndim != 2:
				raise ValueError('did not expect this shape of data: \'{}\', {}'.format(curr_timeseries, curr_data.shape))

			# Skip files with no anomalies
			if not np.all(curr_data[0, 1] == curr_data[:, 1]):
				x.append(curr_data[:, 0])
				y.append(curr_data[:, 1])
				fnames.append(os.path.join(dataset, curr_timeseries))
					
		return x, y, fnames


	def load_window_timeseries(self, dataset):
		'''
		Loads the time series of the given datasets and returns a dataframe

		:param dataset: list of datasets
		:return df: a single dataframe of all loaded time series
		'''
		df_list = []

		# Check if dataset exists
		if dataset not in self.get_dataset_names():
			raise ValueError(f'Dataset {dataset} not in dataset list')
		path = os.path.join(self.window_data_path, dataset)
		
		# Load file names
		timeseries = [f for f in os.listdir(path) if f.endswith('.csv')]

		for curr_timeseries in tqdm(timeseries, desc="Loading time series", leave=False):
			curr_df = pd.read_csv(os.path.join(path, curr_timeseries), index_col=0)
			curr_index = [os.path.join(dataset, x) for x in list(curr_df.index)]
			curr_df.index = curr_index

			df_list.append(curr_df)
				
		df = pd.concat(df_list)

		return df
	

	def load_feature_timeseries(self, dataset):
		# Check if dataset exists
		if dataset not in self.get_dataset_names():
			raise ValueError(f'Dataset {dataset} not in dataset list')

		df = pd.read_csv(self.feature_data_path, index_col=0)
		df = df.filter(like=dataset, axis=0)
		
		return df


	# def load_timeseries(self, timeseries):
	# 	'''
	# 	Loads specified timeseries

	# 	:param fnames: list of file names
	# 	:return x: timeseries
	# 	:return y: corresponding labels
	# 	:return fnames: list of names of the timeseries loaded
	# 	'''
	# 	x = []
	# 	y = []
	# 	fnames = []

	# 	for fname in tqdm(timeseries, desc='Loading timeseries'):
	# 		curr_data = pd.read_csv(os.path.join(self.data_path, fname), header=None).to_numpy()
			
	# 		if curr_data.ndim != 2:
	# 			raise ValueError('did not expect this shape of data: \'{}\', {}'.format(fname, curr_data.shape))

	# 		# Skip files with no anomalies
	# 		if not np.all(curr_data[0, 1] == curr_data[:, 1]):
	# 			x.append(curr_data[:, 0])
	# 			y.append(curr_data[:, 1])
	# 			fnames.append(fname)

	# 	return x, y, fnames


# class Dataloader:
#     """A class for loading data from a dataset directory."""

#     def __init__(self, dataset_dir):
#         """
#         Initialize the Dataloader.

#         Args:
#             dataset_dir (str): Path to the directory containing the dataset.
#         """
#         self.dataset_dir = dataset_dir
		

#     def load_dataset(self):
#         """
#         Load the entire dataset.

#         Returns:
#             dict: A dictionary containing loaded data.
#                   Keys are dataset names, and values are DataFrames or other data structures.
#         """
#         data = {}
#         dataset_names = self._get_dataset_names()

#         for dataset_name in dataset_names:
#             dataset_path = os.path.join(self.dataset_dir, dataset_name)
#             # Load dataset here
#             # Example:
#             # data[dataset_name] = self._load_data(dataset_path)

#         return data

#     def _get_dataset_names(self):
#         """
#         Get the names of all datasets in the dataset directory.

#         Returns:
#             list: A list of dataset names.
#         """
#         dataset_names = []
#         for item in os.listdir(self.dataset_dir):
#             if os.path.isdir(os.path.join(self.dataset_dir, item)):
#                 dataset_names.append(item)
#         return dataset_names

#     def _load_data(self, dataset_path):
#         """
#         Load data from a dataset directory.

#         Args:
#             dataset_path (str): Path to the dataset directory.

#         Returns:
#             pd.DataFrame: A DataFrame containing the loaded data.
#         """
#         # Load data from files in the dataset directory
#         # Example:
#         # data = pd.read_csv(os.path.join(dataset_path, 'data.csv'))
#         # return data
#         pass

#     def load_timeseries(self, timeseries_list):
#         """
#         Load specified time series from the dataset.

#         Args:
#             timeseries_list (list): List of file paths or names of time series to load.

#         Returns:
#             dict: A dictionary containing loaded time series data.
#                   Keys are time series names, and values are DataFrames or other data structures.
#         """
#         time_series_data = {}

#         for timeseries_name in tqdm(timeseries_list, desc='Loading time series'):
#             timeseries_path = os.path.join(self.dataset_dir, timeseries_name)
#             # Load time series here
#             # Example:
#             # time_series_data[timeseries_name] = self._load_timeseries(timeseries_path)

#         return time_series_data

#     def _load_timeseries(self, timeseries_path):
#         """
#         Load a single time series from a file.

#         Args:
#             timeseries_path (str): Path to the file containing the time series data.

#         Returns:
#             pd.DataFrame: A DataFrame containing the loaded time series data.
#         """
#         # Load time series from file
#         # Example:
#         # timeseries_data = pd.read_csv(timeseries_path)
#         # return timeseries_data
#         pass



def main():
	dataloader = Dataloader(dataset_dir="data", raw_data_path="data/raw")
	datasets = dataloader.get_dataset_names()

	x, y, timeseries = dataloader.load_raw_dataset(datasets[0])

	print(len(x))
	print(len(y))
	print(len(timeseries))
	
if __name__ == "__main__":
	main()
