import argparse
import os
from pathlib import Path
import time
import sys
import string
import itertools
import random
from collections import Counter

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


MATPLOTLIB_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

ALL_CHARS = [
    "doNothing",
    *string.ascii_lowercase,
    "greaterThan",
    "tilde",
    "questionMark",
    "apostrophe",
    "comma",
]
CHAR_REPLACEMENTS = {
    "doNothing": "rest",
    "greaterThan": ">",
    "tilde": "~",
    "questionMark": "?",
    "apostrophe": "'",
    "comma": ",",
}
CHAR_TO_CLASS_MAP = {char: idx for idx, char in enumerate(ALL_CHARS)}
CLASS_TO_CHAR_MAP = {idx: char for idx, char in enumerate(ALL_CHARS)}

REACTION_TIME_BINS = 10
TRAINING_WINDOW_BINS = 120

OUTPUTS_DIR = os.path.abspath("./outputs")


########################################################################################
# Main function.
########################################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_plots", action="store_true")
    args = parser.parse_args()
    save_plots = args.save_plots

    ## Load the data.

    DATA_DIR = os.path.abspath("./handwritingBCIData/Datasets/")
    letters_filepaths = []
    for root, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            if filename == "singleLetters.mat":
                letters_filepaths.append(filepath)

    letters_filepaths = sorted(letters_filepaths)

    data_dicts = []
    for filepath in letters_filepaths:
        print(f"Loading {filepath} ...")
        data_dict = loadmat(filepath)
        data_dicts.append(data_dict)
        # break  # for testing quickly

    ## Preprocess and label the data.

    print("Preparing data ...")

    X_train = []
    X_validation = []
    X_test = []
    y_train = []
    y_validation = []
    y_test = []

    validation_count_by_char = Counter()
    test_count_by_char = Counter()

    # Iterate through the sessions.
    NUM_SESSIONS = 1
    # NUM_SESSIONS = None
    for data_dict in data_dicts[:NUM_SESSIONS]:
        neural = data_dict["neuralActivityTimeSeries"]
        go_cue_bins = data_dict["goPeriodOnsetTimeBin"].ravel().astype(int)
        delay_cue_bins = data_dict["delayPeriodOnsetTimeBin"].ravel().astype(int)
        prompts = np.array([a[0] for a in data_dict["characterCues"].ravel()])
        block_by_bin = data_dict["blockNumsTimeSeries"].ravel()
        block_nums = data_dict["blockList"].ravel()

        # Iterate through each block in this session.
        for block_num in block_nums:
            # Get means and stddevs from a random set of train trials in the block, and
            # the rest of the trials can be used for validation and test.
            block_trial_mask = [block_by_bin[b] == block_num for b in go_cue_bins]
            num_trials_in_block = sum(block_trial_mask)
            random_trial_idxs = list(range(num_trials_in_block))
            random.shuffle(random_trial_idxs)
            train_end_idx = int(num_trials_in_block * 0.6)
            train_trial_idxs = random_trial_idxs[:train_end_idx]
            block_go_cue_bins = go_cue_bins[block_trial_mask]
            block_delay_cue_bins = delay_cue_bins[block_trial_mask]
            block_prompts = prompts[block_trial_mask]
            # Loop through the train trials and add the neural data to our list.
            neural_to_zscore_based_on = []
            for trial_idx in train_trial_idxs:
                # For convenience, ignore the last trial in the block.
                if trial_idx + 1 >= len(block_delay_cue_bins):
                    continue
                start_bin = block_delay_cue_bins[trial_idx]
                end_bin = block_delay_cue_bins[trial_idx + 1]
                neural_to_zscore_based_on.extend(neural[start_bin:end_bin])
            neural_to_zscore_based_on = np.array(neural_to_zscore_based_on)
            block_means = np.mean(neural_to_zscore_based_on, axis=0)
            block_stddevs = np.std(neural_to_zscore_based_on, axis=0)

            print(f"Creating labeled pairs for block {block_num} ...")
            for trial_idx in range(num_trials_in_block):
                # Get the training window for this trial.
                go_cue_bin = block_go_cue_bins[trial_idx]
                window_start_bin = int(go_cue_bin) + REACTION_TIME_BINS
                window_end_bin = window_start_bin + TRAINING_WINDOW_BINS
                # Get the neural data in this window.
                trial_neural = neural[window_start_bin:window_end_bin]
                # Z-score the neural data using the block-specific means and stddevs.
                with np.errstate(divide="ignore", invalid="ignore"):
                    trial_zscored_neural = (trial_neural - block_means) / block_stddevs
                trial_zscored_neural = np.nan_to_num(
                    trial_zscored_neural, nan=0, posinf=0, neginf=0
                )

                # Get the character for this trial.
                trial_label = block_prompts[trial_idx]

                # Skip rest trials.
                if trial_label == "doNothing":
                    continue

                # Add the trial to the appropriate set of data (train, validation, or
                # test).
                if trial_idx in train_trial_idxs:
                    X_train.append(trial_zscored_neural)
                    y_train.append(trial_label)
                else:
                    # Put the trial into either validation or test, whichever has fewer
                    # of this trial's character so far.
                    if (
                        validation_count_by_char[trial_label]
                        < test_count_by_char[trial_label]
                    ):
                        X_validation.append(trial_zscored_neural)
                        y_validation.append(trial_label)
                        validation_count_by_char[trial_label] += 1
                    else:
                        X_test.append(trial_zscored_neural)
                        y_test.append(trial_label)
                        test_count_by_char[trial_label] += 1

    X_train = np.array(X_train)
    X_validation = np.array(X_validation)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_validation = np.array(y_validation)
    y_test = np.array(y_test)

    ## Fit a PCA model (only on the training data) with which to transform z-scored
    ## neural data.

    print("Fitting PCA model ...")

    pca_model = PCA().fit(np.concatenate(X_train))

    ## Iterate through different values of hyperparameters to find good ones.

    NUM_NEIGHBORS_TO_TRY = [3, 5, 7]
    NUM_PCS_TO_TRY = [10, 15, 20, 25, 30]
    SMOOTHING_STDDEV = 3.0

    all_combos = list(
        itertools.product(range(len(NUM_NEIGHBORS_TO_TRY)), range(len(NUM_PCS_TO_TRY)))
    )

    validation_accuracies = np.zeros((len(NUM_NEIGHBORS_TO_TRY), len(NUM_PCS_TO_TRY)))

    best_model_so_far = None
    best_hyperparams_so_far = None
    X_test_to_evaluate_on = None
    best_accuracy_so_far = -1

    for combo_idx, (num_neighbors_idx, num_pcs_idx) in enumerate(all_combos):
        print(f"Hyperparam combo {combo_idx + 1} / {len(all_combos)}")

        num_neighbors = NUM_NEIGHBORS_TO_TRY[num_neighbors_idx]
        num_pcs = NUM_PCS_TO_TRY[num_pcs_idx]

        ## Get PCA-transformed data (all of training, validation, and test).

        PCs_X_train = np.array([pca_model.transform(w)[:, :num_pcs] for w in X_train])
        PCs_X_validation = np.array(
            [pca_model.transform(w)[:, :num_pcs] for w in X_validation]
        )
        PCs_X_test = np.array([pca_model.transform(w)[:, :num_pcs] for w in X_test])
        # Smooth the PCs over time.
        PCs_smoothed_X_train = np.array(
            [gaussian_filter1d(w, sigma=SMOOTHING_STDDEV, axis=0) for w in PCs_X_train]
        )
        PCs_smoothed_X_validation = np.array(
            [
                gaussian_filter1d(w, sigma=SMOOTHING_STDDEV, axis=0)
                for w in PCs_X_validation
            ]
        )
        PCs_smoothed_X_test = np.array(
            [gaussian_filter1d(w, sigma=SMOOTHING_STDDEV, axis=0) for w in PCs_X_test]
        )
        # Flatten the PCs so that a decoder can operate on 1D vectors.
        PCs_flattened_X_train = np.reshape(
            PCs_smoothed_X_train, (PCs_smoothed_X_train.shape[0], -1)
        )
        PCs_flattened_X_validation = np.reshape(
            PCs_smoothed_X_validation, (PCs_smoothed_X_validation.shape[0], -1)
        )
        PCs_flattened_X_test = np.reshape(
            PCs_smoothed_X_test, (PCs_smoothed_X_test.shape[0], -1)
        )

        ## Train a k-nearest neighbors model on the preprocessed training data.

        print("Training k-nearest neighbors model ...")

        knn_model = KNeighborsClassifier(n_neighbors=num_neighbors)
        knn_model.fit(PCs_flattened_X_train, y_train)

        ## Evaluate the k-nearest neighbors model on the preprocessed test data.

        y_pred_validation = knn_model.predict(PCs_flattened_X_validation)

        validation_accuracy = np.sum(y_pred_validation == y_validation) / len(
            y_validation
        )

        validation_accuracies[num_neighbors_idx, num_pcs_idx] = validation_accuracy

        if validation_accuracy > best_accuracy_so_far:
            best_model_so_far = knn_model
            best_hyperparams_so_far = {
                "num_neighbors": num_neighbors,
                "num_pcs": num_pcs,
            }
            X_test_to_evaluate_on = PCs_flattened_X_test
            best_accuracy_so_far = validation_accuracy

    ## Plot the confusion matrix of the best-performing model.

    fig, (confusion_ax, hyperparam_ax) = plt.subplots(1, 2)

    y_pred_test = best_model_so_far.predict(X_test_to_evaluate_on)

    test_accuracy = np.sum(y_pred_test == y_test) / len(y_test)

    confusion_results = confusion_matrix(y_test, y_pred_test, normalize="true")

    heatmap = confusion_ax.imshow(confusion_results, origin="lower")

    fig.colorbar(heatmap, ax=confusion_ax)

    confusion_ax.set_xticks(np.arange(len(ALL_CHARS)))
    confusion_ax.set_xticklabels(ALL_CHARS, rotation=45, ha="right")
    confusion_ax.set_xlabel("predicted character")

    confusion_ax.set_yticks(np.arange(len(ALL_CHARS)))
    confusion_ax.set_yticklabels(ALL_CHARS)
    confusion_ax.set_ylabel("true character")

    accuracy_str = f"{round(test_accuracy, 2):.2f}"
    confusion_ax.plot(
        [], [], alpha=0, label=f"{best_hyperparams_so_far['num_neighbors']} neighbors"
    )
    confusion_ax.plot(
        [], [], alpha=0, label=f"{best_hyperparams_so_far['num_pcs']} PCs"
    )
    confusion_ax.plot([], [], alpha=0, label=f"{accuracy_str} accuracy")
    confusion_ax.legend()

    confusion_ax.set_title("Results on test data")

    ## Plot the performance grid of hyperparameter choices.

    heatmap = hyperparam_ax.imshow(validation_accuracies, cmap=plt.cm.Blues_r)

    # Display the accuracy in each square.
    for num_neighbors_idx, num_pcs_idx in itertools.product(
        range(len(NUM_NEIGHBORS_TO_TRY)), range(len(NUM_PCS_TO_TRY))
    ):
        hyperparam_ax.text(
            num_pcs_idx,
            num_neighbors_idx,
            f"{validation_accuracies[num_neighbors_idx, num_pcs_idx]:.3f}",
            ha="center",
            va="center",
        )

    hyperparam_ax.set_xticks(range(len(NUM_PCS_TO_TRY)))
    hyperparam_ax.set_xticklabels(
        [f"{b:.2f}" for b in NUM_PCS_TO_TRY], rotation=45, ha="right"
    )
    hyperparam_ax.set_xlabel("num PCs")

    hyperparam_ax.set_yticks(np.arange(len(NUM_NEIGHBORS_TO_TRY)))
    hyperparam_ax.set_yticklabels([f"{a:.2f}" for a in NUM_NEIGHBORS_TO_TRY])
    hyperparam_ax.set_ylabel("num neighbors")

    fig.colorbar(heatmap, ax=hyperparam_ax, label="accuracy")

    hyperparam_ax.set_title("Accuracies on validation data")

    model_str = "KNearestNeighbors"
    fig.suptitle(f"{model_str} on single-letter instructed-delay task")

    fig.set_figwidth(20)
    fig.set_figheight(20)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])

    if save_plots:
        # Save the figure.
        Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
        plot_filename = f"first_session_single_character_performance_{model_str}.png"
        plot_filepath = os.path.join(OUTPUTS_DIR, plot_filename)
        plt.savefig(plot_filepath)
    else:
        plt.show()


if __name__ == "__main__":
    main()
