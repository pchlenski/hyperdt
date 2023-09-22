# Let's write some code to generate a yml file for settings

import yaml
import os

from time import time, sleep

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from src.hyperdt.forest import HyperbolicRandomForestClassifier
from src.hyperdt.conversions import convert

# Loop controls:
# datasets = ["gaussian", "neuroseed"]
# datasets = ["neuroseed"]
datasets = ["polblogs_hypll"]
clf_names = ["hrf", "hororf", "rf"]
# dims = [2, 4, 8, 16]
dims = [2]
# seeds = [0, 1, 2, 3, 4]
seeds = [0]
# seeds = [5, 6, 7, 8, 9]
# seeds = [10, 11, 12, 13, 14]
# seeds = [15, 16, 17, 18, 19]
# seeds = [15]
# seeds = [16, 17, 18, 19]
n_samples_train = 800

# Adjust for train_test split
n_samples = int(n_samples_train / 0.8)


# Read params from yml file
def evaluate_hdt():
    params = yaml.safe_load(open("./HoroRF/params.yml", "r"))

    # Dataset
    print(f"Using loader from file: {params['dataset_file']}")
    if params["dataset_file"] == "datasets.gaussian":
        from HoroRF.datasets.gaussian import get_training_data, get_testing_data
    elif params["dataset_file"] == "datasets.neuroseed":
        from HoroRF.datasets.neuroseed import get_training_data, get_testing_data
    elif params["dataset_file"] == "datasets.polblogs_geomstats":
        from HoroRF.datasets.polblogs_geomstats import get_training_data, get_testing_data
    elif params["dataset_file"] == "datasets.polblogs_hypll":
        from HoroRF.datasets.polblogs_hypll import get_training_data, get_testing_data

    # Get data
    X_train, y_train = get_training_data(class_label=params["class_label"], seed=params["seed"])
    X_train = convert(X_train.numpy(), "poincare", "hyperboloid")

    # Hyperparams
    args = {
        "n_estimators": params["num_trees"],
        "max_depth": params["max_depth"],
        "min_samples_leaf": params["min_samples_leaf"],
    }

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=params["seed"])
    iterator = list(kf.split(X_train))

    t1 = time()

    # Hyperbolic
    f1_scores_hrf = []
    for train_index, test_index in iterator:
        hrf = HyperbolicRandomForestClassifier(**args)
        hrf.fit(X_train[train_index], y_train[train_index], use_tqdm=True, seed=params["seed"])
        y_pred = hrf.predict(X_train[test_index])
        f1_scores_hrf.append(f1_score(y_train[test_index], y_pred, average="micro"))

    t2 = time()

    # Euclidean
    f1_scores_rf = []
    for train_index, test_index in iterator:
        rf = RandomForestClassifier(**args, random_state=params["seed"])
        rf.fit(X_train[train_index], y_train[train_index])
        y_pred = rf.predict(X_train[test_index])
        f1_scores_rf.append(f1_score(y_train[test_index], y_pred, average="micro"))

    t3 = time()

    return f1_scores_hrf, f1_scores_rf, t2 - t1, t3 - t2


# datasets = ["gaussian", "neuroseed", "polblogs_geomstats"]
results = pd.DataFrame(columns=["dataset", "dim", "seed", "clf", "fold", "f1_micro"])
times = pd.DataFrame(columns=["dataset", "dim", "clf", "time"])
template = yaml.safe_load(open("HoroRF/params_template.yml", "r"))

for seed in seeds:
    for dataset in datasets:
        for dim in dims:
            # Save new params file
            new_param = template.copy()
            outpath = f"logs/big_bench/hororf_{dataset}_{dim}_{seed}"
            new_param["output_path"] = outpath
            new_param["dataset_file"] = f"datasets.{dataset}"
            new_param["class_label"] = dim
            new_param["seed"] = seed
            new_param["num_samples"] = n_samples
            yaml.safe_dump(new_param, open(f"./HoroRF/params.yml", "w"))

            # Run HoroRF
            t1 = time()
            os.system(f"cd ./HoroRF && python train_hyp_rf.py -c ./HoroRF/params.yml")
            t2 = time()
            hororf_time = t2 - t1
            # This saves a copy of the params file, so it's easy to double-check this looking back

            # Run our evaluations
            f1_scores_hrf, f1_scores_rf, hrf_time, rf_time = evaluate_hdt()

            # Save results
            np.savetxt(f"./HoroRF/{outpath}/hrf.txt", f1_scores_hrf, delimiter="\t", fmt="%s")
            np.savetxt(f"./HoroRF/{outpath}/rf.txt", f1_scores_rf, delimiter="\t", fmt="%s")

            # Load results from HoroRF
            f1_scores_hororf = np.loadtxt(f"./HoroRF/{outpath}/results_micro.txt", delimiter="\t")

            # Save results to dataframe
            scores = [f1_scores_hrf, f1_scores_hororf, f1_scores_rf]
            ts = [hrf_time, hororf_time, rf_time]
            for scores, t, name in zip(scores, ts, clf_names):
                for fold, score in enumerate(scores):
                    results.loc[len(results)] = [dataset, dim, seed, name, fold, score]
                times.loc[len(times)] = [dataset, dim, name, t]

            # Save times
            np.savetxt(f"./HoroRF/{outpath}/times.txt", times, delimiter="\t", fmt="%s")

            # Save dataframes
            results.to_csv("./HoroRF/logs/big_bench/hororf_results.tsv", sep="\t")
            times.to_csv("./HoroRF/logs/big_bench/hororf_times.tsv", sep="\t")
