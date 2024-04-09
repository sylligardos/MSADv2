"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD 1st year, 2024
@what: Weighted Average MSAD
"""



# Load the windowed time series and their ground truth, x - y
load_timeseries()


# Load a model selector (obviously window size between time series and model selector should match)
load_model_selector()


# Get the 12 number vector that is the predicted weights for each detector
predict_timeseries()


# Load the 12 anomaly scores of the time series
load_anomaly_scores()


# Average them according to the weights
combine_anomaly_scores()


# Compute the metrics that you want with the combined anomaly score and the ground truth
compute_metrics()


# Save the result in a csv, e.g.:
# Dataset | Time series | file name | window size | model selector | predicted weight AE | predicted weight NormA | ... | 
save_results()