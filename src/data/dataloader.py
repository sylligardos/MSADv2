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
from multiprocessing import Pool
# from concurrent.futures import ProcessPoolExecutor, as_completed



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
		
	def load_timeseries(self, filename):
		"""
		Load a single time series file.

		Parameters:
			filename (str): The path to the time series file.

		Returns:
			tuple: A tuple containing the time series data and the filename.
		"""
		data = pd.read_csv(filename, header=None).to_numpy()
		if data.ndim != 2:
			raise ValueError(f"Unexpected shape of data: '{filename}', {data.shape}")
		if not np.all(data[0, 1] == data[:, 1]):
			return data[:, 0], data[:, 1], "/".join(filename.split('/')[-2:])
		else:
			return None



	def load_raw_dataset_parallel(self, dataset):
		"""
		Load the raw time series from the given dataset in parallel.

		Parameters:
			dataset (str): Name of the dataset to load.

		Returns:
			tuple: A tuple containing lists of time series data, labels, and filenames.
		"""
		if dataset not in self.get_dataset_names():
			raise ValueError(f"Dataset {dataset} not in dataset list")

		path = os.path.join(self.raw_data_path, dataset)
		timeseries_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.out')]

		with Pool() as pool:
			results = list(tqdm(pool.imap(self.load_timeseries, timeseries_files), total=len(timeseries_files), desc=f"Loading {dataset}"))

		x, y, fnames = zip(*[result for result in results if result is not None])

		return list(x), list(y), list(fnames)


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
		
		if len(df_list) <= 0:
			return None
		df = pd.concat(df_list)

		return df

	def load_window_timeseries_parallel(self, dataset):
		"""
		Loads the time series of the given dataset in parallel and returns a dataframe.

		Parameters:
			dataset (str): Name of the dataset to load.

		Returns:
			pd.DataFrame: A single dataframe of all loaded time series.
		"""
		df_list = []

		# Check if dataset exists
		if dataset not in self.get_dataset_names():
			raise ValueError(f"Dataset {dataset} not in dataset list")
		
		path = os.path.join(self.window_data_path, dataset)
		
		# Load file names
		timeseries_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]

		# Load time series in parallel
		with Pool() as pool:
			results = list(tqdm(pool.imap(self.load_timeseries_file, timeseries_files), total=len(timeseries_files), desc="Loading windows"))

		# Filter out any None results
		df_list = [result for result in results if result is not None]

		if not df_list:
			return None

		# Concatenate all dataframes
		df = pd.concat(df_list)

		return df

	def load_timeseries_file(self, filepath):
		"""
		Load a single time series file and return it as a DataFrame.

		Parameters:
			filepath (str): Path to the time series file.

		Returns:
			pd.DataFrame: The loaded time series data.
		"""
		curr_df = pd.read_csv(filepath, index_col=0)
		curr_index = [os.path.join(os.path.basename(os.path.dirname(filepath)), x) for x in list(curr_df.index)]
		curr_df.index = curr_index

		return curr_df
		

	def load_feature_timeseries(self, dataset):
		# Check if dataset exists
		if dataset not in self.get_dataset_names():
			raise ValueError(f'Dataset {dataset} not in dataset list')

		df = pd.read_csv(self.feature_data_path, index_col=0)
		df = df.filter(like=dataset, axis=0)
		
		return df


def load_csv(file_path):
	curr_df = pd.read_csv(file_path, index_col=0)
	curr_index = [os.path.join(dataset, x) for x in list(curr_df.index)]
	curr_df.index = curr_index
	return curr_df

def main():
	dataloader = Dataloader(dataset_dir="data", raw_data_path="data/raw")
	datasets = dataloader.get_dataset_names()

	x, y, timeseries = dataloader.load_raw_dataset(datasets[0])

	print(len(x))
	print(len(y))
	print(len(timeseries))
	
if __name__ == "__main__":
	main()
