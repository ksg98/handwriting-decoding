import os
from pathlib import Path
import time
import string
import random

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
import torch
import torchview
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


MATPLOTLIB_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

ALL_CHARS = [
    *string.ascii_lowercase,
    "greaterThan",
    "tilde",
    "questionMark",
    "apostrophe",
    "comma",
]
CHAR_REPLACEMENTS = {
    "greaterThan": ">",
    "tilde": "~",
    "questionMark": "?",
    "apostrophe": "'",
    "comma": ",",
}
CHAR_TO_CLASS_MAP = {char: idx for idx, char in enumerate(ALL_CHARS)}
CLASS_TO_CHAR_MAP = {idx: char for idx, char in enumerate(ALL_CHARS)}

REACTION_TIME_BINS = 10
TRAINING_WINDOW_BINS = 150
SEQUENCE_LENGTH = TRAINING_WINDOW_BINS

NUM_ELECTRODES = 192

OUTPUTS_DIR = os.path.abspath("./outputs")


########################################################################################
# Main function.
########################################################################################


def main_RNN():
    ## Load the data.

    data_dicts = load_data()

    ## Run the whole process multiple times to get a series of results.

    accuracy_RNN = run_multiple_RNN(data_dicts)
    print(f"accuracy_RNN: {accuracy_RNN}")

    ## Vary number of electrodes.

    NUM_ELECTRODES_TO_TRY = [24, 48, 72, 96, 120, 144, 168, 192]

    accuracies_by_electrodes_RNN = [
        run_multiple_RNN(data_dicts, num_electrodes=num_electrodes)
        for num_electrodes in NUM_ELECTRODES_TO_TRY
    ]
    print(f"accuracies_by_electrodes_RNN: {accuracies_by_electrodes_RNN}")

    ## Vary number of training trials.

    NUM_TRAIN_TRIALS_TO_TRY = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700]

    accuracies_by_train_trials_RNN = [
        run_multiple_RNN(data_dicts, num_train_trials=num_train_trials)
        for num_train_trials in NUM_TRAIN_TRIALS_TO_TRY
    ]
    print(f"accuracies_by_train_trials_RNN: {accuracies_by_train_trials_RNN}")


########################################################################################
# Helper functions.
########################################################################################


def load_data():
    """"""
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

    return data_dicts


def organize_data(data_dicts, limit_electrodes=None, limit_train_trials=None):
    """"""

    print("Preparing data ...")

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # Random electrode order to let us limit electrodes.
    rand_electrode_order = list(range(NUM_ELECTRODES))
    random.shuffle(rand_electrode_order)

    # Iterate through the sessions.
    # NUM_SESSIONS = 1
    NUM_SESSIONS = None
    for data_dict in data_dicts[:NUM_SESSIONS]:
        neural = data_dict["neuralActivityTimeSeries"]
        go_cue_bins = data_dict["goPeriodOnsetTimeBin"].ravel().astype(int)
        delay_cue_bins = data_dict["delayPeriodOnsetTimeBin"].ravel().astype(int)
        prompts = np.array([a[0] for a in data_dict["characterCues"].ravel()])
        block_by_bin = data_dict["blockNumsTimeSeries"].ravel()
        block_nums = data_dict["blockList"].ravel()

        # If specified, only use a random subset of electrodes.
        if limit_electrodes is not None:
            neural = neural[:, rand_electrode_order[:limit_electrodes]]

        # Iterate through each block in this session.
        for block_num in block_nums:
            # Get means and stddevs from a random set of train trials in the block, and
            # the rest of the trials can be used for test.
            block_trial_mask = [block_by_bin[b] == block_num for b in go_cue_bins]
            num_trials_in_block = sum(block_trial_mask)
            random_trial_idxs = list(range(num_trials_in_block))
            random.shuffle(random_trial_idxs)
            train_end_idx = int(num_trials_in_block * 0.8)
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

                # Add the trial to the appropriate set of data (train or test).
                if trial_idx in train_trial_idxs:
                    X_train.append(trial_zscored_neural)
                    y_train.append(trial_label)
                else:
                    X_test.append(trial_zscored_neural)
                    y_test.append(trial_label)

    # Smooth the neural data over time.
    SMOOTHING_STDDEV = 3.0
    X_train = np.array(
        [gaussian_filter1d(w, sigma=SMOOTHING_STDDEV, axis=0) for w in X_train]
    )
    X_test = np.array(
        [gaussian_filter1d(w, sigma=SMOOTHING_STDDEV, axis=0) for w in X_test]
    )

    # Convert the characters to ints, for compatibility with pytorch.
    y_train = np.array([CHAR_TO_CLASS_MAP[ch] for ch in y_train])
    y_test = np.array([CHAR_TO_CLASS_MAP[ch] for ch in y_test])

    # If specified, only use a random subset of train trials.
    if limit_train_trials:
        X_train = X_train[:limit_train_trials]
        y_train = y_train[:limit_train_trials]

    print(f"X_train.shape: {X_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"y_test.shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


class RNNClassifier(torch.nn.Module):
    def __init__(
        self, rnn_type, num_features, num_classes, hidden_size, num_layers, dropout
    ):
        super(RNNClassifier, self).__init__()
        self.rnn_type = rnn_type
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        if self.rnn_type == "rnn":
            self.rnn = torch.nn.RNN(
                self.num_features,
                self.hidden_size,
                self.num_layers,
                dropout=self.dropout,
                batch_first=True,
            )
        elif self.rnn_type == "gru":
            self.rnn = torch.nn.GRU(
                self.num_features,
                self.hidden_size,
                self.num_layers,
                dropout=self.dropout,
                batch_first=True,
            )
        elif self.rnn_type == "lstm":
            self.rnn = torch.nn.LSTM(
                self.num_features,
                self.hidden_size,
                self.num_layers,
                dropout=self.dropout,
                batch_first=True,
            )

        self.rnn_to_out = torch.nn.Linear(
            hidden_size * SEQUENCE_LENGTH, self.num_classes
        )

    def forward(self, inp):
        # Apply the RNN to the sequence(s), and get a sequence(s) of output vectors.
        out, _ = self.rnn(inp)
        # Flatten the sequence of output vectors into a big vector.
        out = out.reshape(-1, self.hidden_size * SEQUENCE_LENGTH)
        # Output layer to map to a 1hot vector of classes.
        out = self.rnn_to_out(out)
        return out


def train_recurrent_neural_network_classifier(
    X_train, X_test, y_train, y_test
):
    """
    Train an RNN model on the training data to yield a classifier we can evaluate.
    """

    print("Training RNN ...")

    s = time.time()

    # Make training data tensors.
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()

    # Define hyperparameters of the model.
    RNN_TYPE = "rnn"
    NUM_FEATURES = X_train.shape[2]
    NUM_CLASSES = len(ALL_CHARS)
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.5

    # Define the model.
    rnn_model = RNNClassifier(
        rnn_type=RNN_TYPE,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    # Visualize the model and save the image to disk.
    torchview.draw_graph(
        rnn_model,
        input_size=[(1, SEQUENCE_LENGTH, NUM_FEATURES)],
        save_graph=True,
        directory=OUTPUTS_DIR,
        filename="graph_for_single_letters_RNN",
    )

    ## Train the model.

    # Define hyperparameters of the training process.
    NUM_SAMPLES = X_train.shape[0]
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0005
    MOMENTUM = 0.99
    WEIGHT_DECAY = 0.001
    NOISE_STDDEV = 0.1
    BATCH_SIZE = 64
    LOG_EVERY_NUM_EPOCHS = 1

    # Define components of the training process.
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        rnn_model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    epoch_train_accuracies = []
    epoch_test_accuracies = []

    # Run multiple epochs of training.
    for epoch_idx in range(NUM_EPOCHS):
        # Shuffle the training data for this epoch.
        random_order = torch.randperm(NUM_SAMPLES)
        epoch_X_train, epoch_y_train = X_train[random_order], y_train[random_order]

        # Data augmentation, to help avoid overfitting to the training data.
        # # Every other epoch, create synthetic training data from the real training data.
        # if epoch_idx % 2 == 0:
        # Add random noise to the neural data, so it looks like brand new samples.
        epoch_noise = torch.randn(*epoch_X_train.shape) * NOISE_STDDEV
        epoch_X_train += epoch_noise

        # Keep track of the total and correct samples in this epoch, for calculating
        # accuracy on the training data.
        epoch_train_samples = 0
        epoch_train_samples_correct = 0

        for batch_idx in range(int(len(epoch_X_train) / BATCH_SIZE)):
            # Zero the parameters' gradients before training on the new batch.
            rnn_model.zero_grad()

            # Get the data for this batch.
            batch_X_train = epoch_X_train[
                BATCH_SIZE * batch_idx : BATCH_SIZE * (batch_idx + 1)
            ]
            batch_y_train = epoch_y_train[
                BATCH_SIZE * batch_idx : BATCH_SIZE * (batch_idx + 1)
            ]

            # Apply the RNN to get outputs for this batch.
            batch_y_pred_1hot = rnn_model(batch_X_train)
            batch_y_pred = np.argmax(batch_y_pred_1hot.detach().numpy(), axis=1)

            # Compare the RNN's prediction to the correct answer to get the loss.
            batch_loss = loss_function(batch_y_pred_1hot, batch_y_train)

            # Back propogate from the loss to calculate each model parameter's gradient.
            batch_loss.backward()

            # Update the model parameters' weights using gradient descent.
            optimizer.step()

            # Keep track of stats for calculating accuracy on training data.
            epoch_train_samples += len(batch_y_pred)
            epoch_train_samples_correct += sum(
                batch_y_pred == batch_y_train.detach().numpy()
            )

        # Calculate accuracy on training set for this epoch.
        epoch_train_accuracy = epoch_train_samples_correct / epoch_train_samples
        epoch_train_accuracies.append(epoch_train_accuracy)

        # Calculate accuracy on test data.
        with torch.no_grad():
            # Apply the RNN to get outputs for the test set.
            y_pred_test_1hot = rnn_model(X_test)
            y_pred_test = np.argmax(y_pred_test_1hot.detach().numpy(), axis=1)

            # Calculate accuracy on test set for this epoch.
            epoch_test_samples = len(y_pred_test)
            epoch_test_samples_correct = sum(y_pred_test == y_test)
            epoch_test_accuracy = epoch_test_samples_correct / epoch_test_samples
            epoch_test_accuracies.append(epoch_test_accuracy)

        # Periodically log progress and stats.
        if epoch_idx % LOG_EVERY_NUM_EPOCHS == 0:
            print(
                f"finished epoch: {epoch_idx}/{NUM_EPOCHS}\t"
                f"train accuracy: {round(epoch_train_accuracy, 4):.4f}\t"
                f"test accuracy: {round(epoch_test_accuracy, 4):.4f}\t"
                f"{round(time.time() - s, 1):.1f} sec\t"
            )

    print(f"Trained RNN in {round(time.time() - s)} sec.")

    ## Plot the performance over the course of the training.

    show_plot = False
    if show_plot:

        fig, ax = plt.subplots()

        ax.plot(epoch_train_accuracies, label="train")
        ax.plot(epoch_test_accuracies, label="test")

        ax.set_xlabel("epoch")

        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel("accuracy")

        ax.set_title("Performance over the course of training (RNN)")

        plt.tight_layout()

        plt.show()

    return rnn_model


def plot_confusion_matrix(y_test, y_pred_test, accuracy_str):
    """"""

    fig, confusion_ax = plt.subplots()

    confusion_results = confusion_matrix(y_test, y_pred_test, normalize="true")

    heatmap = confusion_ax.imshow(confusion_results, origin="lower")

    fig.colorbar(heatmap, ax=confusion_ax)

    confusion_ax.set_xticks(np.arange(len(ALL_CHARS)))
    confusion_ax.set_xticklabels(ALL_CHARS, rotation=45, ha="right")
    confusion_ax.set_xlabel("predicted character")

    confusion_ax.set_yticks(np.arange(len(ALL_CHARS)))
    confusion_ax.set_yticklabels(ALL_CHARS)
    confusion_ax.set_ylabel("true character")

    model_str = "RNN"
    confusion_ax.set_title(
        f"{model_str} on single-letter instructed-delay task (accuracy: {accuracy_str})"
    )

    plt.tight_layout()

    plt.show()


def run_multiple_RNN(
    data_dicts, num_electrodes=None, num_train_trials=None, num_runs=1
):
    """"""

    accuracy_results = []

    for run_idx in range(num_runs):
        print(f"RNN run {run_idx + 1} / {num_runs}")

        ## Preprocess and label the data.

        X_train, X_test, y_train, y_test = organize_data(
            data_dicts,
            limit_electrodes=num_electrodes,
            limit_train_trials=num_train_trials,
        )

        # To get chance-level performance, shuffle the training labels.
        calc_chance = False
        if calc_chance:
            rg = np.random.default_rng()
            rg.shuffle(y_train)
            print("Shuffled training labels to calculate chance performance.")

        ## Train an RNN model on the preprocessed training data.

        rnn_model = train_recurrent_neural_network_classifier(
            X_train, X_test, y_train, y_test
        )

        ## Evaluate the RNN model by calculating accuracy on the test set.

        with torch.no_grad():
            X_test = torch.from_numpy(X_test).float()
            y_pred_test_1hot = rnn_model(X_test)
            y_pred_test = np.argmax(y_pred_test_1hot, axis=1)

        test_accuracy = sum(y_pred_test.detach().numpy() == y_test) / len(y_test)

        # Store this run's result.
        accuracy_results.append(test_accuracy)

        accuracy_str = f"{round(test_accuracy, 3):.3f}"
        print(f"accuracy: {accuracy_str}")

        ## Optionally plot the confusion matrix.

        show_confusion_matrix = False
        if show_confusion_matrix:
            plot_confusion_matrix(y_test, y_pred_test, accuracy_str)

    mean_accuracy = np.mean(accuracy_results)

    print(f"accuracies: {accuracy_results}")
    print(f"mean accuracy: {np.mean(accuracy_results)}")

    return mean_accuracy


if __name__ == "__main__":
    main_RNN()
