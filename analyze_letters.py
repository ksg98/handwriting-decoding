import argparse
import os
from pathlib import Path
import time
import sys
import string

from dateutil.parser import parse as dtparse
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
import torchview
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
CHAR_TO_CLASS_MAP = {char: idx for idx, char in enumerate(ALL_CHARS)}
CLASS_TO_CHAR_MAP = {idx: char for idx, char in enumerate(ALL_CHARS)}

OUTPUTS_DIR = os.path.abspath("./outputs")

########################################################################################
# Main function.
########################################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize_neural", action="store_true")
    parser.add_argument("--run_linear", action="store_true")
    parser.add_argument("--run_nn", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    args = parser.parse_args()
    visualize_neural = args.visualize_neural
    run_linear = args.run_linear
    run_nn = args.run_nn
    save_plots = args.save_plots

    ## Load the data.

    data_dicts = load_data()

    if visualize_neural:
        ## Visualize the neural data.

        visualize_neural_data(data_dicts)

    ## Create labeled pairs.

    X_train, X_validation, X_test, y_train, y_validation, y_test = organize_data(
        data_dicts
    )

    if run_linear:
        ## Train a logistic regression classifier model.

        logistic_regression_model = train_logistic_regression_classifier(
            X_train, y_train
        )

        ## Evaluate the logistic regression classifier model.

        evaluate_classifier_model(
            model_name="LogisticRegression",
            predict=logistic_regression_model.predict,
            X_test=X_test,
            y_test=y_test,
            save_plots=save_plots,
        )

    if run_nn:
        ## Train a neural network classifier model.

        neural_network_model = train_neural_network_classifier(
            X_train,
            X_validation,
            y_train,
            y_validation,
            save_plots=save_plots,
        )

        ## Evaluate the neural network classifier model.

        # Wrap the prediction function for compatibility with the common evaluation
        # function.
        def neural_network_predict(X):
            """"""
            # Convert numpy arrays to tensors to work with the neural network model.
            X = torch.from_numpy(X).float()
            # Use the neural network model to make a prediction.
            y_pred_1hot = neural_network_model(X)
            # The neural network model returns 1-hot vectors, so convert them to char
            # class indices by getting the index of the non-zero element.
            y_pred = np.argmax(y_pred_1hot.detach().numpy(), axis=1)
            return y_pred

        evaluate_classifier_model(
            model_name="NeuralNetwork",
            predict=neural_network_predict,
            X_test=X_test,
            y_test=y_test,
            save_plots=save_plots,
        )


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
        test_size=0.2,
        shuffle=True,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        shuffle=True,
    )
    # Convert the characters to ints, for compatibility with all model libraries.
    y_train = np.array([CHAR_TO_CLASS_MAP[ch] for ch in y_train])
    y_validation = np.array([CHAR_TO_CLASS_MAP[ch] for ch in y_validation])
    y_test = np.array([CHAR_TO_CLASS_MAP[ch] for ch in y_test])

    print(f"X_train: {X_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_validation: {y_validation.shape}")
    print(f"y_test: {y_test.shape}")

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def train_logistic_regression_classifier(X_train, y_train):
    """
    Train a logistic regression model on the training data to yield a classifier we can
    evaluate.
    """

    print("Training logistic regression model ...")

    s = time.time()

    model = LogisticRegression(solver="newton-cg")
    model.fit(X_train, y_train)

    print(f"Trained logistic regression model in {round(time.time() - s)} sec.")

    return model


def train_neural_network_classifier(
    X_train, X_validation, y_train, y_validation, save_plots=False
):
    """
    Train a Neural Network model on the training data to yield a classifier we can
    evaluate.
    """

    print("Training neural network model ...")

    s = time.time()

    # Make training data tensors.
    X_train = torch.from_numpy(X_train).float()
    X_validation = torch.from_numpy(X_validation).float()
    y_train = torch.from_numpy(y_train).long()

    # Define hyperparameters of the model.
    NUM_FEATURES = X_train.shape[1]
    NUM_HIDDEN_LAYER_0 = 64
    NUM_HIDDEN_LAYER_1 = 64
    NUM_CLASSES = len(ALL_CHARS)

    # Define the model.
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(NUM_FEATURES, NUM_HIDDEN_LAYER_0),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(NUM_HIDDEN_LAYER_0, NUM_HIDDEN_LAYER_1),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(NUM_HIDDEN_LAYER_0, NUM_CLASSES),
    # )
    model = torch.nn.Sequential(
        torch.nn.Linear(NUM_FEATURES, NUM_CLASSES),
    )

    # Visualize the model and save the image to disk.
    torchview.draw_graph(
        model,
        input_size=(1, NUM_FEATURES),
        save_graph=True,
        directory=OUTPUTS_DIR,
        filename="graph_for_single_letters_nn",
    )

    ## Train the model.

    # Define hyperparameters of the training process.
    NUM_SAMPLES = X_train.shape[0]
    NUM_EPOCHS = 2000
    BATCH_SIZE = 50
    LEARNING_RATE = 10e-6
    MOMENTUM = 0.9
    LOG_EVERY_NUM_EPOCHS = 10

    # Define components of the training process.
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    epoch_train_accuracies = []
    epoch_validation_accuracies = []

    # Run multiple epochs of training.
    for epoch_idx in range(NUM_EPOCHS):
        # Shuffle the training data for this epoch.
        random_order = torch.randperm(NUM_SAMPLES)
        epoch_X_train, epoch_y_train = X_train[random_order], y_train[random_order]

        # Keep track of the total and correct samples in this epoch, for calculating
        # accuracy on the training data.
        epoch_train_samples = 0
        epoch_train_samples_correct = 0

        # Run the specified batches for this epoch (may be 1 batch, or many).
        num_batches = int(NUM_SAMPLES / BATCH_SIZE)
        for batch_idx in range(num_batches):
            # Get the samples for this batch.
            batch_start_idx = BATCH_SIZE * batch_idx
            batch_end_idx = BATCH_SIZE * (batch_idx + 1)
            batch_X_train = epoch_X_train[batch_start_idx:batch_end_idx]
            batch_y_train = epoch_y_train[batch_start_idx:batch_end_idx]

            # Zero the parameters' gradients before training on the new batch of data.
            model.zero_grad()

            # Use the model to predict outputs for this batch.
            batch_y_pred_1hot = model(batch_X_train)

            # Compare the model predictions to the correct labels to calculate the loss.
            batch_loss = loss_function(batch_y_pred_1hot, batch_y_train)

            # Back propogate from the loss to calculate each model parameter's gradient.
            batch_loss.backward()

            # Update the model parameters' weights using gradient descent.
            optimizer.step()

            # Keep track of stats for calculating accuracy on training data.
            batch_y_pred = np.argmax(batch_y_pred_1hot.detach().numpy(), axis=1)
            batch_y_train = batch_y_train.detach().numpy()
            epoch_train_samples += len(batch_y_pred)
            epoch_train_samples_correct += sum(batch_y_pred == batch_y_train)

        # Calculate accuracy on training for this epoch.
        epoch_train_accuracy = epoch_train_samples_correct / epoch_train_samples
        epoch_train_accuracies.append(epoch_train_accuracy)

        # Calculate accuracy on validation data.
        with torch.no_grad():
            epoch_y_pred_validation_1hot = model(X_validation)
            epoch_y_pred_validation = np.argmax(
                epoch_y_pred_validation_1hot.detach().numpy(), axis=1
            )
            epoch_validation_samples = len(epoch_y_pred_validation)
            epoch_validation_samples_correct = sum(
                epoch_y_pred_validation == y_validation
            )
            epoch_validation_accuracy = (
                epoch_validation_samples_correct / epoch_validation_samples
            )
            epoch_validation_accuracies.append(epoch_validation_accuracy)

        # Periodically log progress and stats.
        if epoch_idx % LOG_EVERY_NUM_EPOCHS == 0:
            print(
                f"epochs: {epoch_idx}/{NUM_EPOCHS}\t"
                f"train accuracy: {round(epoch_train_accuracy, 4):.4f}\t"
                f"validation accuracy: {round(epoch_validation_accuracy, 4):.4f}\t"
                f"{round(time.time() - s, 1):.1f} sec\t"
            )

    print(f"Trained in {round(time.time() - s)} sec.")

    ## Plot the performance over the course of the training.

    fig, ax = plt.subplots()

    ax.plot(epoch_train_accuracies, label="train")
    ax.plot(epoch_validation_accuracies, label="validation")

    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")

    ax.set_title(
        "Performance over the course of training, for the neural network model"
    )

    plt.tight_layout()

    if save_plots:
        # Save the figure.
        Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
        plot_filename = f"single_character_performance_during_training.png"
        plot_filepath = os.path.join(OUTPUTS_DIR, plot_filename)
        plt.savefig(plot_filepath)
    else:
        plt.show()

    return model


def evaluate_classifier_model(model_name, predict, X_test, y_test, save_plots=False):
    """
    Evaluate a trained model on the test data.
    """
    y_pred = predict(X_test)

    accuracy = np.sum(y_pred == y_test) / len(y_test)

    confusion_results = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots()

    heatmap = ax.imshow(confusion_results)

    fig.colorbar(heatmap, ax=ax)

    ax.set_xticks(np.arange(len(ALL_CHARS)))
    ax.set_xticklabels(ALL_CHARS, rotation=45, ha="right")
    ax.set_xlabel("predicted character")

    ax.set_yticks(np.arange(len(ALL_CHARS)))
    ax.set_yticklabels(ALL_CHARS)
    ax.set_ylabel("true character")

    model_str = model_name.split("(")[0]
    accuracy_str = str(round(accuracy, 2))
    ax.set_title(f"{model_str} on single characters (accuracy: {accuracy_str})")

    fig.set_figwidth(12)
    fig.set_figheight(8)

    plt.tight_layout()

    if save_plots:
        # Save the figure.
        Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
        plot_filename = f"single_character_performance_{model_str}.png"
        plot_filepath = os.path.join(OUTPUTS_DIR, plot_filename)
        plt.savefig(plot_filepath)
    else:
        plt.show()


if __name__ == "__main__":
    main()
