"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: MSADv2
"""


import argparse
import os
import pandas as pd

from utils.utils import load_data, load_model, predict_timeseries, compute_weighted_scores, combine_anomaly_scores, compute_metrics
from data.scoreloader import Scoreloader

def run_model(experiment, model, split, test=False):
    # Setup with your paths
    data_path = os.path.join('data')
    model_info = {
        'convnet': ('convnet_default_128', 128),
        'knn': ('knn_1024', 1024),
        'resnet': ('resnet_default_1024', 1024),
        'sit': ('sit_stem_original_512', 512),
    }
    window_size = model_info[model][1]
    model_small_name, model_name = model, model_info[model][0]
    model_weights_path = os.path.join(
        'models', 
        'unsupervised' if 'unsupervised' in split else 'supervised', 
        split.replace('unsupervised', model_name) if 'unsupervised' in split else model_name
    )
 
    y, fnames, window_data, scores = load_data(
        data_path=data_path,
        window=window_size,
        dataset=['YAHOO'] if test else None,
        features=(model == 'knn'),
        split_file=split
    )
    model = load_model(model_name, window_size, model_weights_path)
    window_pred_probabilities = predict_timeseries(model_name, model, window_data, fnames)

    top_models_combinations = {
        'convnet':  [(4, 'vote'), (4, 'average')],
        'resnet':   [(4, 'vote'), (5, 'average')],
        'sit':      [(7, 'vote'), (8, 'average')],
        'knn':      [(3, 'vote'), (8, 'average')],
    }
    curr_top_combinations = top_models_combinations[model_small_name]
    for k, combination_method in curr_top_combinations:
        weights = compute_weighted_scores(window_pred_probabilities, combination_method, k)
        weighted_scores = combine_anomaly_scores(scores, weights, plot=False)
        metric_values = compute_metrics(y, weighted_scores, k)

        # Save results
        detector_names = Scoreloader(os.path.join('data', 'scores')).get_detector_names()
        weights_df = pd.DataFrame(weights, fnames, columns=[f"weight_{x}" for x in detector_names])
        metric_results_df = pd.DataFrame(metric_values, fnames)
        results_df = pd.concat([metric_results_df, weights_df], axis=1)

        experiment_dir = os.path.join('experiments', experiment, f"{model_small_name}{window_size}")
        os.makedirs(experiment_dir, exist_ok=True)
        filename = f"{split.replace('unsupervised_', '') if 'unsupervised' in split else 'TSB'}_{combination_method}_{k}.csv"
        results_df.to_csv(os.path.join(experiment_dir, filename))
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-model selection")
    
    parser.add_argument("--experiment", type=str, help="Name of experiment")
    parser.add_argument("--model", type=str, help="The name of the model to run")
    parser.add_argument("--split", type=str, default=None, help="Split file for which time series to load")
    parser.add_argument("--test", action="store_true", help="Run in test mode")

    args = parser.parse_args()

    if args.model == 'all':
        models = ['convnet', 'resnet', 'sit', 'knn']
    else:
        models = [args.model]
    for model in models:
        run_model(
            experiment=args.experiment,
            model=model,
            split=args.split, 
            test=args.test,
        )