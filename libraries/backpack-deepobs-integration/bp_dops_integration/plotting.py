from __future__ import print_function

import os
import time
from collections import Counter

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from deepobs.tuner.tuner_utils import generate_tuning_summary
from deepobs.analyzer.analyze_utils import _preprocess_path, _rescale_ax
from bp_dops_integration.plotting_utils import create_setting_analyzer_ranking, _get_optimizer_name_and_testproblem_from_path, _determine_available_metric


def _plot_optimizer_performance(
    path,
    fig=None,
    ax=None,
    mode="most",
    metric="valid_accuracies",
    which="mean_and_std",
):
    """Plots the training curve of an optimizer.

    Args:
        path (str): Path to the optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are plotted).
        fig (matplotlib.Figure): Figure to plot the training curves in.
        ax (matplotlib.axes.Axes): The axes to plot the trainig curves for all metrices. Must have 4 subaxes.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        which (str): ['mean_and_std', 'median_and_quartiles'] Solid plot mean or median, shaded plots standard deviation or lower/upper quartiles.

    Returns:
        matplotlib.axes.Axes: The axes with the plots.
        """
    metrices = [
        "test_losses",
        "train_losses",
        "test_accuracies",
        "train_accuracies",
    ]
    if ax is None:  # create default axis for all 4 metrices
        fig, ax = plt.subplots(4, 1, sharex="col")

    pathes = _preprocess_path(path)
    for optimizer_path in pathes:
        setting_analyzer_ranking = create_setting_analyzer_ranking(
            optimizer_path, mode, metric
        )
        setting = setting_analyzer_ranking[0]

        optimizer_name = os.path.basename(optimizer_path)
        for idx, _metric in enumerate(metrices):
            if _metric in setting.aggregate:

                if which == "mean_and_std":
                    center = setting.aggregate[_metric]["mean"]
                    std = setting.aggregate[_metric]["std"]
                    low, high = center - std, center + std
                elif which == "median_and_quartiles":
                    center = setting.aggregate[_metric]["median"]
                    low = setting.aggregate[_metric]["lower_quartile"]
                    high = setting.aggregate[_metric]["upper_quartile"]
                else:
                    raise ValueError("Unknown value which={}".format(which))

                ax[idx].plot(center, label=optimizer_name)
                ax[idx].fill_between(range(len(center)), low, high, alpha=0.3)

    _, testproblem = _get_optimizer_name_and_testproblem_from_path(
        optimizer_path
    )
    ax[0].set_title(testproblem, fontsize=18)
    return fig, ax


def plot_optimizer_performance(
    path,
    fig=None,
    ax=None,
    mode="area",
    metric="valid_accuracies",
    reference_path=None,
    show=True,
    which="mean_and_std",
):
    """Plots the training curve of optimizers and addionally plots reference results from the ``reference_path``

    Args:
        path (str): Path to the optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are plotted).
        fig (matplotlib.Figure): Figure to plot the training curves in.
        ax (matplotlib.axes.Axes): The axes to plot the trainig curves for all metrices. Must have 4 subaxes (one for each metric).
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        reference_path (str): Path to the reference optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are taken as reference).
        show (bool): Whether to show the plot or not.
        which (str): ['mean_and_std', 'median_and_quartiles'] Solid plot mean or median, shaded plots standard deviation or lower/upper quartiles.

    Returns:
        tuple: The figure and axes with the plots.

        """

    fig, ax = _plot_optimizer_performance(
        path, fig, ax, mode, metric, which=which
    )
    if reference_path is not None:
        fig, ax = _plot_optimizer_performance(
            reference_path, fig, ax, mode, metric, which=which
        )

    metrices = ["Test Loss", "Train Loss", "Test Accuracy", "Train Accuracy"]
    for idx, _metric in enumerate(metrices):
        # set y labels

        ax[idx].set_ylabel(_metric, fontsize=14)
        # rescale plots
        # ax[idx] = _rescale_ax(ax[idx])
        ax[idx].tick_params(labelsize=12)

    # show optimizer legends
    ax[0].legend(fontsize=12)

    ax[3].set_xlabel("Epochs", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.15)

    if show:
        plt.show()
    return fig, ax


def _plot_hyperparameter_sensitivity(
    optimizer_path,
    ax,
    mode="final",
    metric="valid_accuracies",
):

    metric = _determine_available_metric(optimizer_path, metric)
    tuning_summary = generate_tuning_summary(optimizer_path, mode, metric)

    optimizer_name, testproblem = _get_optimizer_name_and_testproblem_from_path(
        optimizer_path
    )

    # create array for plotting
    param_values = [d["params"] for d in tuning_summary]
    target_means = [d[metric + "_mean"] for d in tuning_summary]
    
    plot_list = {}
    for params, means in zip(param_values, target_means):
        try:
            plot_list[(params["beta1"], params["beta2"])].append({params["lr"]: means})
        except:
            plot_list[(params["beta1"], params["beta2"])] = [{params["lr"]: means}]
        
    for i, setting in enumerate(plot_list):
        internal_dict = plot_list[setting]
        x, y = [], []
        for element in internal_dict:
            for k,v in element.items():
                x.append(k); y.append(v)

        x = np.asarray(x); y = np.asarray(y); sorted_indx = np.argsort(x)
        
        idx1 = i // 3; idx2 = i % 3
        
        if "SGD" in optimizer_name:
            idx1 = idx2 = 2

        ax[idx1][idx2].plot(x[sorted_indx], y[sorted_indx], linewidth=1, label=optimizer_name)
        ax[idx1][idx2].set_ylabel(str(setting), fontsize=14)
        ax[idx1][idx2].tick_params(labelsize=12)
        ax[idx1][idx2].set_xscale('log')
        if "SGD" in optimizer_name:
            ax[2][2].legend(prop={'size': 8})
            
        ax[0][0].legend(prop={'size': 8})

    ax[0][0].set_title(testproblem, fontsize=20)
    return ax


def plot_hyperparameter_sensitivity(
    path,
    mode="final",
    metric="valid_accuracies",
    reference_path=None,
    show=True,
    fig=None,
    ax=None,
):
    """Plots the hyperparameter sensitivtiy of the optimizer.

    Args:
        path (str): The path to the optimizer to analyse. Or to a whole testproblem. In that case, all optimizer sensitivities are plotted.
        hyperparam (str): The name of the hyperparameter that should be analyzed.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        xscale (str): The scale for the parameter axes. Is passed to plt.xscale().
        plot_std (bool): Whether to plot markers for individual seed runs or not. If `False`, only the mean is plotted.
        reference_path (str): Path to the reference optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are taken as reference).
        show (bool): Whether to show the plot or not.

    Returns:
        tuple: The figure and axes of the plot.
        """
    if fig is None:
        fig, ax = plt.subplots()

    pathes = _preprocess_path(path)
    for optimizer_path in pathes:
        metric = _determine_available_metric(optimizer_path, metric)
        ax = _plot_hyperparameter_sensitivity(
            optimizer_path, ax, mode, metric
        )
    if reference_path is not None:
        pathes = _preprocess_path(reference_path)
        for reference_optimizer_path in pathes:
            metric = _determine_available_metric(
                reference_optimizer_path, metric
            )
            ax = _plot_hyperparameter_sensitivity(
                reference_optimizer_path, ax, mode, metric
            )

    if show:
        plt.show()
    return fig, ax
