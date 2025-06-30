from itertools import combinations
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import math

def shapley_values(args):
    """
    Compute Shapley values for each detector.

    Parameters:
    - labels: array of shape (t,)
    - detector_scores: array of shape (m, t), anomaly scores from m detectors

    Returns:
    - shapley: array of shape (m,), Shapley value per detector
    """
    labels, detector_scores = args
    m = detector_scores.shape[0]
    shapley = np.zeros(m)

    for i in tqdm(range(m), desc="Computing Shapley values", disable=True):
        others = [j for j in range(m) if j != i]
        for subset_size in range(m):  # 0 to m-1
            subsets = combinations(others, subset_size)
            for S in subsets:
                S = list(S)
                S_with_i = S + [i]

                # Compute average precision (AUC-PR) for both subsets
                score_with_i = np.mean(detector_scores[S_with_i], axis=0)
                v_with_i = average_precision_score(labels, score_with_i)

                if S:
                    score_without_i = np.mean(detector_scores[S], axis=0)
                    v_without_i = average_precision_score(labels, score_without_i)
                else:
                    # Define baseline when no detectors are present (e.g., mean score)
                    score_without_i = np.full_like(labels, np.mean(detector_scores))
                    v_without_i = average_precision_score(labels, score_without_i)

                # Weight = |S|!(m - |S| - 1)! / m!
                weight = (
                    math.factorial(len(S)) * math.factorial(m - len(S) - 1)
                ) / math.factorial(m)
                shapley[i] += weight * (v_with_i - v_without_i)

    return shapley
