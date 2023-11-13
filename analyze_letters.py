import argparse
import os
from pathlib import Path
import time
import sys

from dateutil.parser import parse as dtparse
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

MATPLOTLIB_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


########################################################################################
# Main function.
########################################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_plots", action="store_true")
    args = parser.parse_args()
    show_plots = args.show_plots

    ## Load the data.

    data_dicts = load_data()

    # ## Visualize the neural data.

    # visualize_neural_data(data_dicts)
    # return

    ## Create labeled pairs.

    X_train, X_test, y_train, y_test = organize_data(data_dicts)

    ## Train a classifier model.

    model = train_classifier_model(X_train, y_train)

    ## Evaluate the classifier model.

    evaluate_classifier_model(model, X_test, y_test, show_plots=show_plots)


########################################################################################
# Helper functions.
########################################################################################


def load_data():
    """
    Scrape the data directory and load the data files from disk into dicts in memory.
    """

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

    return data_dicts


def visualize_neural_data(data_dicts):
    """"""

    print("Visualizing the neural data ...")

    num_plots = len(data_dicts)
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols)

    if type(axs) != np.ndarray:
        axs = np.array([[axs]])

    for session_idx, data_dict in enumerate(data_dicts):
        session_date = dtparse(data_dict["blockStartDates"][0][0][0]).date()

        timestamps = data_dict["clockTimeSeries"]
        bin_lengths = np.diff(timestamps.flatten())
        real_bin_lengths = bin_lengths[bin_lengths > 0]
        bin_length_sec = np.round(np.mean(real_bin_lengths), 4)

        neural_activity = data_dict["neuralActivityTimeSeries"]
        delay_cue_bins = data_dict["delayPeriodOnsetTimeBin"]
        go_cue_bins = data_dict["goPeriodOnsetTimeBin"]
        prompts = data_dict["characterCues"]

        NUM_PCS = 3
        pca_model = PCA(n_components=NUM_PCS)
        pca_neural_activity = pca_model.fit_transform(neural_activity)

        row_idx = session_idx // num_cols
        col_idx = session_idx % num_cols

        ax = axs[row_idx, col_idx]

        neural_sum = np.sum(neural_activity, axis=1)
        smoothed_neural_sum = gaussian_filter1d(neural_sum, 5)
        (sum_line,) = ax.plot(
            smoothed_neural_sum,
            color=MATPLOTLIB_COLORS[0],
            linestyle="--",
            label="summed activity",
        )

        pc_ax = ax.twinx()
        pca_lines = []
        smoothed_pca_neural_activity = gaussian_filter1d(pca_neural_activity, 5, axis=0)
        for i in range(NUM_PCS):
            (pca_line,) = pc_ax.plot(
                smoothed_pca_neural_activity[:, i],
                color=MATPLOTLIB_COLORS[i + 1],
                alpha=0.5,
                label=f"PC {i + 1}",
            )
            pca_lines.append(pca_line)

        pca_avg = np.mean(smoothed_pca_neural_activity, axis=1)
        (pca_avg_line,) = pc_ax.plot(
            pca_avg,
            color=MATPLOTLIB_COLORS[NUM_PCS + 1],
            linewidth=2,
            label="PC avg",
        )

        for delay_cue_bin in delay_cue_bins:
            ax.axvline(delay_cue_bin, color="orange")

        for go_cue_bin in go_cue_bins:
            ax.axvline(go_cue_bin, color="green")

        ax.set_xlabel("time bin")
        ax.set_ylabel("threshold crossings")

        ax.set_title(f"session {session_date}")

        legend_lines = [sum_line, *pca_lines, pca_avg_line]
        legend_labels = [
            sum_line.get_label(),
            *[pca_line.get_label() for pca_line in pca_lines],
            pca_avg_line.get_label(),
        ]
        ax.legend(legend_lines, legend_labels)

    fig.set_figwidth(16)
    fig.set_figheight(10)

    plt.tight_layout()

    plt.show()


def organize_data(data_dicts):
    """
    Take the dicts of session data, slice up trials to get inputs and labels, and split
    these data into train and test portions.
    """

    print("Organizing data ...")

    REACTION_TIME_NUM_BINS = 10
    TRAINING_WINDOW_NUM_BINS = 90

    input_vectors = []
    labels = []
    for session_idx, data_dict in enumerate(data_dicts):
        neural_activity = data_dict["neuralActivityTimeSeries"]
        delay_cue_bins = data_dict["delayPeriodOnsetTimeBin"].ravel().astype(int)
        go_cue_bins = data_dict["goPeriodOnsetTimeBin"].ravel().astype(int)
        prompts = [a[0] for a in data_dict["characterCues"].ravel()]

        print(f"Creating training pairs for session {session_idx} ...")
        for trial_idx, go_cue_bin in enumerate(go_cue_bins):
            # Get the training window for this trial.
            training_window_start_bin = int(go_cue_bin) + REACTION_TIME_NUM_BINS
            training_window_end_bin = (
                training_window_start_bin + TRAINING_WINDOW_NUM_BINS
            )
            training_window_neural_activity = neural_activity[
                training_window_start_bin:training_window_end_bin
            ]
            # Make a separate feature out of every (feature, timestep) pair.
            input_vector = training_window_neural_activity.flatten()

            label = prompts[trial_idx]

            input_vectors.append(input_vector)
            labels.append(label)

    input_vectors = np.array(input_vectors)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        input_vectors,
        labels,
        test_size=0.1,
        shuffle=True,
    )
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def train_classifier_model(X_train, y_train):
    """
    Train on the training data to yield a classifier model we can evaluate.
    """

    print("Training classifier model ...")

    s = time.time()

    model = LogisticRegression(solver="newton-cg")
    model.fit(X_train, y_train)

    print(f"Trained in {round(time.time() - s)} sec.")

    return model


def evaluate_classifier_model(model, X_test, y_test, show_plots=False):
    """
    Evaluate the trained model on the test data.
    """
    y_pred = model.predict(X_test)

    accuracy = np.sum(y_pred == y_test) / len(y_test)

    confusion_results = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots()

    heatmap = ax.imshow(confusion_results)

    fig.colorbar(heatmap, ax=ax)

    ax.set_xticks(np.arange(len(model.classes_)))
    ax.set_xticklabels(model.classes_, rotation=45, ha="right")
    ax.set_xlabel("predicted character")

    ax.set_yticks(np.arange(len(model.classes_)))
    ax.set_yticklabels(model.classes_)
    ax.set_ylabel("true character")

    model_str = str(model).split("(")[0]
    accuracy_str = str(round(accuracy, 2))
    ax.set_title(f"{model_str} on single characters (accuracy: {accuracy_str})")

    fig.set_figwidth(12)
    fig.set_figheight(8)

    plt.tight_layout()

    if show_plots:
        plt.show()
    else:
        # Save the figure.
        outputs_dir = os.path.abspath("./outputs")
        Path(outputs_dir).mkdir(parents=True, exist_ok=True)
        plot_filename = f"single_character_performance_{model_str}.png"
        plot_filepath = os.path.join(outputs_dir, plot_filename)
        plt.savefig(plot_filepath)


if __name__ == "__main__":
    main()
