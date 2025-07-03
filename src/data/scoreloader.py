"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st Year (2024)
@what: Weighted Average MSAD
"""


import os
from tqdm import tqdm
import numpy as np 
import pandas as pd
import multiprocessing


class Scoreloader:
	def __init__(self, scores_path):
		self.scores_path = scores_path

	def get_detector_names(self):
		'''Returns the names of the detectors

		:return: list of strings
		'''
		detectors = []

		for dataset in os.listdir(self.scores_path):
			curr_detectors = []
			for name in os.listdir(os.path.join(self.scores_path, dataset)):
				curr_detectors.append(name)
			if len(detectors) < 1:
				detectors = curr_detectors.copy()
			elif not detectors == curr_detectors:
				raise ValueError('detectors are not the same in this dataset \'{}\''.format(dataset))
		detectors.sort()

		return detectors

	def load(self, file_names):
		'''
		Load the score for the specified files/timeseries. If a time series has no score for all 
		the detectors (e.g. the anomaly score has been computed for 10/12 detectors) the this
		time series is skipped. Its index is returned in the idx_failed for the user to remove 
		it from any other places if needed.

		:param dataset: list of files
		:return scores: the loaded scores
		:return idx_failed: list with indexes of not loaded time series
		'''
		detectors = self.get_detector_names()
		scores = []
		idx_failed = []

		for i, name in enumerate(tqdm(file_names, desc='Loading scores', leave=False)):
			name_split = name.split('/')[-2:]
			paths = [os.path.join(self.scores_path, name_split[0], detector, 'score', name_split[1]) for detector in detectors]
			data = []
			try:
				for path in paths:
					data.append(pd.read_csv(path, header=None).to_numpy())
			except Exception as e:
				idx_failed.append(i)
				continue
			scores.append(np.concatenate(data, axis=1))

		# Delete ts which failed to load
		if len(idx_failed) > 0:
			print('failed to load')
			for idx in sorted(idx_failed, reverse=True):
				print('\t\'{}\''.format(file_names[idx]))
				# del file_names[idx]

		return scores, idx_failed


	def load_score(self, args):
		"""
		Load scores for a single file.

		Parameters:
			filename (str): The path to the score file.

		Returns:
			numpy.ndarray: The loaded scores.
		"""
		filename, detectors = args
		name_split = filename.split('/')[-2:]
		paths = [os.path.join(self.scores_path, name_split[0], detector, 'score', name_split[1]) for detector in detectors]
		data = []
		try:
			for path in paths:
				data.append(pd.read_csv(path, header=None).to_numpy())
			return np.concatenate(data, axis=1)
		except Exception as e:
			return None

	def load_parallel(self, file_names):
		"""
		Load the scores for the specified files/timeseries in parallel.

		Parameters:
			file_names (list): List of file names.

		Returns:
			tuple: A tuple containing the loaded scores and a list of indexes of failed to load time series.
		"""
		detectors = self.get_detector_names()
		scores = []
		idx_failed = []

		# Prepare arguments list
		args_list = [(filename, detectors) for filename in file_names]

		num_procs = max(1, int(multiprocessing.cpu_count() * 2 / 3))
		with multiprocessing.Pool(processes=num_procs) as pool:
			results = list(tqdm(pool.imap(self.load_score, args_list), total=len(file_names), desc='Loading scores'))

		for i, result in enumerate(results):
			if result is not None:
				scores.append(result)
			else:
				idx_failed.append(i)

		return scores, idx_failed
