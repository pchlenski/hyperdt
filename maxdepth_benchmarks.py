# Let's write some code to generate a yml file for settings

import yaml
import os

from time import time, sleep

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.hyperdt.tree import HyperbolicDecisionTreeClassifier
from src.hyperdt.ensemble import HyperbolicRandomForestClassifier
from src.hyperdt.conversions import convert

from HoroRF.datasets.gaussian import get_training_data, get_testing_data

# Loop controls:
dataset = "gaussian"
dim = 2
n_samples_train = 800
clf_names = ["hrf", "hororf", "rf"]
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Tree controls
max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9]
num_classifiers = 12
min_samples_leaf = 1

# Adjust for train_test split
n_samples = int(n_samples_train / 0.8)
if num_classifiers == 1:
    no_resample = True
else:
    no_resample = False


# Read params from yml file
def evaluate_hdt():
    params = yaml.safe_load(open("./HoroRF/params.yml", "r"))

    # Dataset
    t0 = time()
    print(f"Using loader from file: {params['dataset_file']}")

    # Get data
    X_train, y_train = get_training_data(
        class_label=params["class_label"],
        seed=params["seed"],
        num_samples=params["num_samples"],
        convert_to_poincare=False,
    )
    X_train = X_train.numpy()
    print("X shape:", X_train.shape)
    y_train = y_train.numpy()

    # Hyperparams
    args = {
        "n_estimators": params["num_trees"],
        "max_depth": params["max_depth"],
        "min_samples_leaf": params["min_samples_leaf"],
        "random_state": params["seed"],
    }
    use_tree = False
    if args["n_estimators"] == 1:
        del args["n_estimators"]  # This is a decision tree now
        del args["random_state"]
        use_tree = True

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=params["seed"])
    iterator = list(kf.split(X_train))

    t1 = time()

    # Hyperbolic
    f1_scores_hrf = []
    if "hrf" in clf_names:
        for train_index, test_index in iterator:
            try:
                if use_tree:
                    hrf = HyperbolicDecisionTreeClassifier(**args)
                    hrf.fit(X_train[train_index], y_train[train_index])
                else:
                    hrf = HyperbolicRandomForestClassifier(**args)
                    hrf.fit(X_train[train_index], y_train[train_index], use_tqdm=True, seed=params["seed"])
                y_pred = hrf.predict(X_train[test_index])
                f1_scores_hrf.append(f1_score(y_train[test_index], y_pred, average="micro"))
            except Exception as e:
                print(e)
                f1_scores_hrf.append(np.nan)

    t2 = time()

    # Euclidean
    f1_scores_rf = []
    if "rf" in clf_names:
        for train_index, test_index in iterator:
            try:
                if use_tree:
                    rf = DecisionTreeClassifier(**args)
                else:
                    rf = RandomForestClassifier(**args)
                rf.fit(X_train[train_index], y_train[train_index])
                y_pred = rf.predict(X_train[test_index])
                f1_scores_rf.append(f1_score(y_train[test_index], y_pred, average="micro"))
            except Exception as e:
                print(e)
                f1_scores_rf.append(np.nan)

    t3 = time()

    return f1_scores_hrf, f1_scores_rf, t2 - t1, t3 - t2, t1 - t0


results = pd.DataFrame(columns=["max_depth", "seed", "clf", "fold", "f1_micro"])
times = pd.DataFrame(columns=["max_depth", "seed", "clf", "time", "init_time"])
template = yaml.safe_load(open("HoroRF/params_template.yml", "r"))

for max_depth in max_depths:
    for seed in seeds:
        # Save new params file
        new_param = template.copy()
        outpath = f"logs/depth_bench"
        new_param["output_path"] = outpath
        new_param["dataset_file"] = f"datasets.{dataset}"
        new_param["class_label"] = dim
        new_param["seed"] = seed
        new_param["num_samples"] = n_samples
        new_param["num_trees"] = num_classifiers
        new_param["max_depth"] = max_depth
        new_param["min_samples_leaf"] = min_samples_leaf
        new_param["no_resample"] = no_resample
        yaml.safe_dump(new_param, open(f"./HoroRF/params.yml", "w"))

        # Run HoroRF
        t1 = time()
        if "hororf" in clf_names:
            os.system(f"cd ./HoroRF && python train_hyp_rf.py -c ./HoroRF/params.yml")
        t2 = time()
        hororf_time = t2 - t1
        # This saves a copy of the params file, so it's easy to double-check this looking back

        # Run our evaluations
        f1_scores_hrf, f1_scores_rf, hrf_time, rf_time, init_time = evaluate_hdt()

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
                results.loc[len(results)] = [max_depth, seed, name, fold, score]
            times.loc[len(times)] = [max_depth, seed, name, t, 0]

        # Save times
        np.savetxt(f"./HoroRF/{outpath}/times.txt", times, delimiter="\t", fmt="%s")

        # Save dataframes
        results.to_csv("./HoroRF/logs/depth_bench/hororf_results.tsv", sep="\t")
        times.to_csv("./HoroRF/logs/depth_bench/hororf_times.tsv", sep="\t")
