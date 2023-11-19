import argparse
import os

from dateutil.parser import parse as dtparse
import numpy as np
import scipy.io
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--something", action="store_true")

    args = parser.parse_args()

    something = args.something

    ## Load the data.

    print("Loading the data ...")

    data_dicts = load_data()

    print("Loaded the data.")

    ## Visualize the data.

    print("Visualizing the data ...")

    visualize_data(data_dicts)

    print("Visualized the data.")


def load_data():
    """"""
    DATA_DIR = os.path.abspath("./handwritingBCIData/Datasets/")
    sentences_filepaths = []
    for root, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            if filename == "sentences.mat":
                sentences_filepaths.append(filepath)

    sentences_filepaths = sorted(sentences_filepaths)

    data_dicts = []
    for filepath in sentences_filepaths:
        print(f"Loading {filepath} ...")
        data_dict = scipy.io.loadmat(filepath)
        data_dicts.append(data_dict)
        # break  # for testing quickly

    return data_dicts


def visualize_data(data_dicts):
    """"""
    num_plots = len(data_dicts)
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols)

    if type(axs) != np.ndarray:
        axs = np.array([[axs]])

    for ax_idx, data_dict in enumerate(data_dicts):
        session_date = dtparse(data_dict["blockStartDates"][0][0][0]).date()

        timestamps = data_dict["clockTimeSeries"]
        bin_lengths = np.diff(timestamps.flatten())
        real_bin_lengths = bin_lengths[bin_lengths > 0]
        bin_length_sec = np.round(np.mean(real_bin_lengths), 4)

        neural_activity = data_dict["neuralActivityTimeSeries"]
        delay_cue_bins = data_dict["delayCueOnsetTimeBin"].flatten()
        go_cue_bins = data_dict["goCueOnsetTimeBin"].flatten()
        prompts = data_dict["sentencePrompt"].flatten()

        row_idx = ax_idx // num_cols
        col_idx = ax_idx % num_cols

        ax = axs[row_idx, col_idx]

        ax.plot(np.sum(neural_activity, axis=1))

        for delay_cue_bin in delay_cue_bins:
            ax.axvline(delay_cue_bin, color="orange")

        for go_cue_bin in go_cue_bins:
            ax.axvline(go_cue_bin, color="green")

        ax.set_xlabel("time bin")
        ax.set_ylabel("threshold crossings")

        ax.set_title(f"session {session_date}")

    fig.set_figwidth(16)
    fig.set_figheight(10)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
