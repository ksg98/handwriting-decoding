import os
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


MATPLOTLIB_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

OUTPUTS_DIR = os.path.abspath("./outputs")


def main():
    with open("outputs/manual_results.json") as f:
        results_dict = json.load(f)

    ## Plot performance by train trials.

    train_fig, train_ax = plt.subplots()

    trainfit_fig, trainfit_ax = plt.subplots()

    model_idx = 0
    for model_name, model_results in results_dict.items():
        if "trials" in model_results:
            train_sizes_tried = [
                d["num_trials"] for d in results_dict[model_name]["trials"]
            ]
            train_sizes_to_fit_on = train_sizes_tried[2:-1]

            accuracy_by_train_size = [
                d["mean_accuracy"] for d in results_dict[model_name]["trials"]
            ]
            accuracy_by_train_size_to_fit_on = accuracy_by_train_size[2:-1]

            initial_L = 1.0
            initial_S = -1.0
            initial_A = -0.001
            initial_B = 1.0
            (fit_params_accuracy_by_train_size, *_) = curve_fit(
                exponential_decay,
                train_sizes_to_fit_on,
                accuracy_by_train_size_to_fit_on,
                (initial_L, initial_S, initial_A, initial_B),
            )
            print(f"{model_name} asymptote: {fit_params_accuracy_by_train_size[0]}")

            fit_by_trials_sample_points = np.linspace(
                train_sizes_tried[0], train_sizes_tried[-1], 1000
            )
            fit_accuracy_by_trials_sampled = [
                exponential_decay(num_trials, *fit_params_accuracy_by_train_size)
                for num_trials in fit_by_trials_sample_points
            ]

            plt.figure(train_fig.number)
            train_ax.scatter(
                train_sizes_tried,
                accuracy_by_train_size,
                color=MATPLOTLIB_COLORS[model_idx],
                label=model_name,
            )

            plt.figure(trainfit_fig.number)
            trainfit_ax.scatter(
                train_sizes_tried,
                accuracy_by_train_size,
                color=MATPLOTLIB_COLORS[model_idx],
                label=model_name,
            )
            trainfit_ax.plot(
                fit_by_trials_sample_points,
                fit_accuracy_by_trials_sampled,
                color=MATPLOTLIB_COLORS[model_idx],
                linewidth=3,
                alpha=0.7,
            )

            model_idx += 1

    plt.figure(train_fig.number)
    
    train_ax.set_xlabel("# train trials")

    train_ax.set_ylabel("accuracy")
    train_ax.set_ylim([0, 1])

    train_ax.legend()

    plt.figure(trainfit_fig.number)

    trainfit_ax.set_xlabel("# train trials")

    trainfit_ax.set_ylabel("accuracy")
    trainfit_ax.set_ylim([0, 1])

    trainfit_ax.legend()

    # Save the figures.

    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    
    plt.figure(train_fig.number)
    train_filename = f"performance_by_train_trials.png"
    train_filepath = os.path.join(OUTPUTS_DIR, train_filename)
    plt.savefig(train_filepath, bbox_inches="tight")

    plt.figure(trainfit_fig.number)
    trainfit_filename = f"performance_by_train_trials_with_fit.png"
    trainfit_filepath = os.path.join(OUTPUTS_DIR, trainfit_filename)
    plt.savefig(trainfit_filepath, bbox_inches="tight")

    ## Plot performance by electrodes.

    electrodes_fig, electrodes_ax = plt.subplots()

    electrodesfit_fig, electrodesfit_ax = plt.subplots()

    model_idx = 0
    for model_name, model_results in results_dict.items():
        if "electrodes" in model_results:
            electrodes_tried = [
                d["num_electrodes"] for d in results_dict[model_name]["electrodes"]
            ]
            electrodes_to_fit_on = electrodes_tried[1:-1]

            accuracy_by_electrodes = [
                d["mean_accuracy"] for d in results_dict[model_name]["electrodes"]
            ]
            accuracy_by_electrodes_to_fit_on = accuracy_by_electrodes[1:-1]

            initial_L = 1.0
            initial_S = -1.0
            initial_A = -0.001
            initial_B = 1.0
            (fit_params_accuracy_by_electrodes, *_) = curve_fit(
                exponential_decay,
                electrodes_to_fit_on,
                accuracy_by_electrodes_to_fit_on,
                (initial_L, initial_S, initial_A, initial_B),
            )
            print(f"{model_name} asymptote: {fit_params_accuracy_by_electrodes[0]}")

            fit_by_electrodes_sample_points = np.linspace(
                electrodes_tried[0], electrodes_tried[-1], 1000
            )
            fit_accuracy_by_electrodes_sampled = [
                exponential_decay(num_electrodes, *fit_params_accuracy_by_electrodes)
                for num_electrodes in fit_by_electrodes_sample_points
            ]

            plt.figure(electrodes_fig.number)
            electrodes_ax.scatter(
                electrodes_tried,
                accuracy_by_electrodes,
                color=MATPLOTLIB_COLORS[model_idx],
                label=model_name,
            )

            plt.figure(electrodesfit_fig.number)
            electrodesfit_ax.scatter(
                electrodes_tried,
                accuracy_by_electrodes,
                color=MATPLOTLIB_COLORS[model_idx],
                label=model_name,
            )
            electrodesfit_ax.plot(
                fit_by_electrodes_sample_points,
                fit_accuracy_by_electrodes_sampled,
                color=MATPLOTLIB_COLORS[model_idx],
                linewidth=3,
                alpha=0.7,
            )

            model_idx += 1

    plt.figure(electrodes_fig.number)
    
    electrodes_ax.set_xlabel("# electrodes")

    electrodes_ax.set_ylabel("accuracy")
    electrodes_ax.set_ylim([0, 1])

    electrodes_ax.legend()

    plt.figure(electrodesfit_fig.number)

    electrodesfit_ax.set_xlabel("# electrodes")

    electrodesfit_ax.set_ylabel("accuracy")
    electrodesfit_ax.set_ylim([0, 1])

    electrodesfit_ax.legend()

    # Save the figures.

    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    
    plt.figure(electrodes_fig.number)
    electrodes_filename = f"performance_by_electrodes.png"
    electrodes_filepath = os.path.join(OUTPUTS_DIR, electrodes_filename)
    plt.savefig(electrodes_filepath, bbox_inches="tight")

    plt.figure(electrodesfit_fig.number)
    electrodesfit_filename = f"performance_by_electrodes_with_fit.png"
    electrodesfit_filepath = os.path.join(OUTPUTS_DIR, electrodesfit_filename)
    plt.savefig(electrodesfit_filepath, bbox_inches="tight")


def exponential_decay(x, L, S, A, B):
    return L + S * np.exp(A * (x + B))


if __name__ == "__main__":
    main()
