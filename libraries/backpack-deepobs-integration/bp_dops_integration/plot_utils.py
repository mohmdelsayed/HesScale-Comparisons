"""Plotting of extended logging quantities."""
import os
import matplotlib.pyplot as plt

from deepobs.analyzer.analyze import _preprocess_path, plot_optimizer_performance

from deepobs.analyzer.shared_utils import (
    _get_optimizer_name_and_testproblem_from_path,
    _check_if_metric_is_available,
)


def create_setting_analyzer_ranking(
    optimizer_path, mode="final", metric="valid_accuracies"
):
    """Reads in all settings in ``optimizer_path`` and sets up a ranking by returning an ordered list of SettingAnalyzers.
    Args:
        optimizer_path (str): The path to the optimizer to analyse.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
    Returns:
        An ordered list of SettingAnalyzers. I.e. the first item is considered 'the best one' etc.
    """
    metric = _determine_available_metric(optimizer_path, metric)
    setting_analyzers = _get_all_setting_analyzer(optimizer_path)

    if "acc" in metric:
        sgn = -1
    else:
        sgn = 1

    if mode == "final":
        setting_analyzers_ordered = sorted(
            setting_analyzers, key=lambda idx: sgn * idx.get_final_value(metric)
        )
    elif mode == "best":
        setting_analyzers_ordered = sorted(
            setting_analyzers, key=lambda idx: sgn * idx.get_best_value(metric)
        )
    elif mode == "most":
        # if all have the same amount of runs, i.e. no 'most' avalaible, fall back to 'final'
        if all(x.n_runs == setting_analyzers[0].n_runs for x in setting_analyzers):
            (
                optimizer_name,
                testproblem_name,
            ) = _get_optimizer_name_and_testproblem_from_path(optimizer_path)
            warnings.warn(
                "All settings for {0:s} on test problem {1:s} have the same number of seeds runs. Mode 'most' does not make sense and we use the fallback mode 'final'".format(
                    optimizer_path, testproblem_name
                ),
                RuntimeWarning,
            )
            setting_analyzers_ordered = sorted(
                setting_analyzers,
                key=lambda idx: sgn * idx.get_final_value(metric),
            )
        else:
            setting_analyzers_ordered = sorted(
                setting_analyzers, key=lambda idx: idx.n_runs, reverse=True
            )
    else:
        raise RuntimeError("Mode not implemented")

    return setting_analyzers_ordered


def plot_hyperparameter_sensitivity_2d(
    optimizer_path,
    hyperparams,
    mode="final",
    metric="valid_accuracies",
    xscale="linear",
    yscale="linear",
    show=False,
):
    param1, param2 = hyperparams
    metric = _determine_available_metric(optimizer_path, metric)
    tuning_summary = generate_tuning_summary(optimizer_path, mode, metric)

    optimizer_name, testproblem = _get_optimizer_name_and_testproblem_from_path(
        optimizer_path
    )

    param_values1 = np.array([d["params"][param1] for d in tuning_summary])
    param_values2 = np.array([d["params"][param2] for d in tuning_summary])

    target_means = np.array([d[metric + "_mean"] for d in tuning_summary])
    target_stds = [d[metric + "_std"] for d in tuning_summary]

    fig, ax = plt.subplots()

    con = ax.tricontourf(
        param_values1,
        param_values2,
        target_means,
        cmap="CMRmap",
        levels=len(target_means),
    )
    ax.scatter(param_values1, param_values2)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    cbar = plt.colorbar(con)
    cbar.set_label(metric)
    if show:
        plt.show()
    return fig, ax


def _determine_available_metric(optimizer_path, metric, default_metric="valid_losses"):
    """Checks if the metric ``metric`` is availabe for the runs in ``optimizer_path``.
    If not, it returns the fallback metric ``default_metric``."""
    optimizer_name, testproblem_name = _get_optimizer_name_and_testproblem_from_path(
        optimizer_path
    )
    if _check_if_metric_is_available(optimizer_path, metric):
        return metric
    else:

        # TODO remove if-else once validation metrics are available for the baselines
        if _check_if_metric_is_available(optimizer_path, default_metric):
            warnings.warn(
                "Metric {0:s} does not exist for testproblem {1:s}. We now use fallback metric {2:s}".format(
                    metric, testproblem_name, default_metric
                ),
                RuntimeWarning,
            )
            return default_metric
        else:
            warnings.warn(
                "Cannot fallback to metric {0:s} for optimizer {1:s} on testproblem {2:s}. Will now fallback to metric test_losses".format(
                    default_metric, optimizer_name, testproblem_name
                ),
                RuntimeWarning,
            )
            return "test_losses"


def _plot_hyperparameter_sensitivity(
    optimizer_path,
    hyperparam,
    ax,
    mode="final",
    metric="valid_accuracies",
    plot_std=False,
):

    metric = _determine_available_metric(optimizer_path, metric)
    tuning_summary = generate_tuning_summary(optimizer_path, mode, metric)

    optimizer_name, testproblem = _get_optimizer_name_and_testproblem_from_path(
        optimizer_path
    )

    # create array for plotting
    param_values = [d["params"][hyperparam] for d in tuning_summary]
    target_means = [d[metric + "_mean"] for d in tuning_summary]
    target_stds = [d[metric + "_mean"] for d in tuning_summary]

    param_values, target_means, target_stds = (
        list(t) for t in zip(*sorted(zip(param_values, target_means, target_stds)))
    )

    param_values = np.array(param_values)
    target_means = np.array(target_means)
    ax.plot(param_values, target_means, linewidth=3, label=optimizer_name)
    if plot_std:
        ranks = create_setting_analyzer_ranking(optimizer_path, mode, metric)
        for rank in ranks:
            values = rank.get_all_final_values(metric)
            param_value = rank.aggregate["optimizer_hyperparams"][hyperparam]
            for value in values:
                ax.scatter(param_value, value, marker="x", color="b")
            ax.plot(
                (param_value, param_value),
                (min(values), max(values)),
                color="grey",
                linestyle="--",
            )
    ax.set_title(testproblem, fontsize=20)
    return ax


def _get_optimizer_name_and_testproblem_from_path(optimizer_path):
    optimizer_name = os.path.split(optimizer_path)[-1]
    testproblem = os.path.split(os.path.split(optimizer_path)[0])[-1]
    return optimizer_name, testproblem


def plot_optimizer_extended_logs(
    path,
    ax=None,
    mode="most",
    metric="valid_accuracies",
    reference_path=None,
    which="mean_and_std",
    custom_metrics=None,
):
    """Analog to `plot_optimizer_performance` for extended logging metrics."""
    raise NotImplementedError("Custom metrics not supported by DeepOBS yet")
    if custom_metrics is None:
        custom_metrics = []

    num_dobs_plots = 4
    num_plots = num_dobs_plots + len(custom_metrics)

    if ax is None:
        _, ax = plt.subplots(num_plots, 1, sharex="col")

    # DeepOBS plots
    ax = plot_optimizer_performance(
        path,
        ax=ax,
        mode=mode,
        metric=metric,
        reference_path=reference_path,
        which=which,
    )

    # Custom metrics plots
    ax = _plot_optimizer_extended_logs(
        path, ax, mode=mode, metric=metric, which=which, custom_metrics=custom_metrics
    )

    for idx, custom_metric in enumerate(custom_metrics, num_dobs_plots):
        # set y labels
        ax[idx].set_ylabel(custom_metric, fontsize=14)
        ax[idx].tick_params(labelsize=12)
        # show optimizer legends
        ax[idx].legend(fontsize=12)

    ax[-1].set_xlabel("epochs", fontsize=14)

    return ax


def _plot_optimizer_extended_logs(
    path,
    ax,
    mode="most",
    metric="valid_accuracies",
    which="mean_and_std",
    custom_metrics=None,
):
    raise NotImplementedError("Custom metrics not supported by DeepOBS yet")

    all_possible_custom_metrics = [
        "batch_loss_before_step",
        "batch_loss_after_step",
        "l2_reg_before_step",
        "l2_reg_after_step",
        "batch_loss_grad_norm_before_step",
        "batch_loss_grad_norm_after_step",
        "damping",
        "trust_damping",
        "inv_damping",
        "parameter_change_norm",
        "batch_loss_improvement",
    ]

    if custom_metrics is None:
        custom_metrics = []

    num_dobs_plots = 4

    pathes = _preprocess_path(path)

    for optimizer_path in pathes:
        setting_analyzer_ranking = create_setting_analyzer_ranking(
            optimizer_path, mode, metric, custom_metrics=all_possible_custom_metrics
        )
        setting = setting_analyzer_ranking[0]

        def items_in_aggregate_with_key_containing(string):
            return [
                (key, value)
                for (key, value) in setting.aggregate.items()
                if string in key
            ]

        optimizer_name = os.path.basename(optimizer_path)

        for idx, metric in enumerate(custom_metrics, num_dobs_plots):
            if idx == num_dobs_plots:
                _, testproblem = _get_optimizer_name_and_testproblem_from_path(
                    optimizer_path
                )
                ax[idx].set_title(testproblem, fontsize=18)

            # reuse same color/style if multiple lines
            color, linestyle = None, None

            for metric_name, metric_data in items_in_aggregate_with_key_containing(
                metric
            ):

                if which == "mean_and_std":
                    center = metric_data["mean"]
                    std = metric_data["std"]
                    low, high = center - std, center + std
                elif which == "median_and_quartiles":
                    center = metric_data["median"]
                    low = metric_data["lower_quartile"]
                    high = metric_data["upper_quartile"]
                else:
                    raise ValueError("Unknown value which={}".format(which))

                # label = "{}, {}".format(optimizer_name,
                #                         metric_name).replace("_", "\_")
                label = metric_name.replace("_", "\_")

                (line,) = ax[idx].plot(
                    center, label=label, color=color, linestyle=linestyle
                )
                if color is None and linestyle is None:
                    color = line.get_color()
                    linestyle = line.get_linestyle()
                ax[idx].fill_between(
                    range(len(center)), low, high, facecolor=color, alpha=0.3
                )

    return ax
