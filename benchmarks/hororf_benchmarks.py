# Let's write some code to generate a yml file for settings

import os
from time import sleep, time

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from src.hyperdt.conversions import convert
from src.hyperdt.ensemble import HyperbolicRandomForestClassifier
from src.hyperdt.tree import HyperbolicDecisionTreeClassifier

# Loop controls:
# datasets = ["gaussian", "neuroseed", "polblogs_hypll"]
clf_names = ["hrf", "hororf", "rf"]
# dims = [2, 4, 8, 16]
datasets = ["binary_wordnet"]
dims = ["animal", "group", "mammal", "occupation", "rodent", "solid", "tree", "worker", 1, 2, 3]
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
n_samples_train = [1000]  # Not used for binary_wordnet

# Tree controls
max_depth = 3
num_classifiers = 1
min_samples_leaf = 1

# Adjust for train_test split
n_samples = [int(x / 0.8) for x in n_samples_train]
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
    if params["dataset_file"] == "datasets.gaussian":
        from HoroRF.datasets.gaussian import get_testing_data, get_training_data
    elif params["dataset_file"] == "datasets.neuroseed":
        from HoroRF.datasets.neuroseed import get_testing_data, get_training_data
    elif params["dataset_file"] == "datasets.polblogs_geomstats":
        from HoroRF.datasets.polblogs_geomstats import get_testing_data, get_training_data
    elif params["dataset_file"] == "datasets.polblogs_hypll":
        from HoroRF.datasets.polblogs_hypll import get_testing_data, get_training_data
    elif params["dataset_file"] == "datasets.binary_wordnet":
        from HoroRF.datasets.binary_wordnet import get_testing_data, get_training_data

    # Get data
    X_train, y_train = get_training_data(
        class_label=params["class_label"],
        seed=params["seed"],
        num_samples=params["num_samples"],
        convert_to_poincare=False,
    )
    X_train = X_train.numpy()
    print("X shape:", X_train.shape)
    # X_train = convert(X_train.numpy(), "poincare", "hyperboloid")
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


# datasets = ["gaussian", "neuroseed", "polblogs_geomstats"]
results = pd.DataFrame(columns=["n_samples", "dataset", "dim", "seed", "clf", "fold", "f1_micro"])
# times = pd.DataFrame(columns=["n_samples", "dataset", "dim", "clf", "time", "init_time"])
times = pd.DataFrame(columns=["n_samples", "dataset", "dim", "seed", "clf", "time", "init_time"])
template = yaml.safe_load(open("HoroRF/params_template.yml", "r"))

for n in n_samples:
    for seed in seeds:
        for dataset in datasets:
            # Some special logic for polblogs_hypll datasets, where we don't do sampling:
            if dataset == "polblogs_hypll":
                if n != 1000:
                    continue

            for dim in dims:
                # Save new params file
                new_param = template.copy()
                outpath = f"logs/big_bench/hororf_{dataset}_{dim}_{seed}"
                new_param["output_path"] = outpath
                new_param["dataset_file"] = f"datasets.{dataset}"
                new_param["class_label"] = dim
                new_param["seed"] = seed
                new_param["num_samples"] = n
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
                        results.loc[len(results)] = [int(n * 0.8), dataset, dim, seed, name, fold, score]
                    # times.loc[len(times)] = [int(n * 0.8), dataset, dim, name, t, init_time]
                    times.loc[len(times)] = [int(n * 0.8), dataset, dim, seed, name, t, 0]

                # Save times
                np.savetxt(f"./HoroRF/{outpath}/times.txt", times, delimiter="\t", fmt="%s")

                # Save dataframes
                results.to_csv("./HoroRF/logs/big_bench/hororf_results.tsv", sep="\t")
                times.to_csv("./HoroRF/logs/big_bench/hororf_times.tsv", sep="\t")
