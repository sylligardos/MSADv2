"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st Year (2024)
@what: Weighted Average MSAD
"""


import os


# Data
data_dir = os.path.join("data")

raw_data_path = os.path.join(data_dir, "raw")
feature_data_path = os.path.join(data_dir, "features")

def generate_window_data_path(window_size):
    if window_size not in [16, 32, 64, 128, 256, 512, 768, 1024]:
        raise ValueError(f"Window size {window_size} is not available")
    return os.path.join(data_dir, f"TSB_{window_size}")
     
scores_path = os.path.join(data_dir, "scores")

# Models
weights_path = os.path.join("models")
supervised_weights_path = os.path.join(weights_path, "supervised")
unsupervised_weights_path = os.path.join(weights_path, "unsupervised")
model_parameters_file = os.path.join("src", "models", "configuration")

# Splits
split_file_path = os.path.join(data_dir, "splits")
supervised_split_file_path = os.path.join(split_file_path, "supervised")
unsupervised_split_file_path = os.path.join(split_file_path, "unsupervised")
