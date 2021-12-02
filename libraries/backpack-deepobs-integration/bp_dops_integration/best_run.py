import os

import numpy as np

from deepobs.analyzer.shared_utils import _get_all_setting_analyzer
from deepobs.analyzer.shared_utils import  _read_all_settings_folders, _check_if_metric_is_available

from .runners import BPOptimRunner, BPOptimRunnerExtendedLogging


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
        if all(
            x.n_runs == setting_analyzers[0].n_runs for x in setting_analyzers
        ):
            optimizer_name, testproblem_name = _get_optimizer_name_and_testproblem_from_path(
                optimizer_path
            )
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



def _get_optimizer_name_and_testproblem_from_path(optimizer_path):
    optimizer_name = os.path.split(optimizer_path)[-1]
    testproblem = os.path.split(os.path.split(optimizer_path)[0])[-1]
    return optimizer_name, testproblem

def _determine_available_metric(
    optimizer_path, metric, default_metric="valid_losses"
):
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


def my_rerun_setting(runner,
                     optimizer_class,
                     hyperparam_names,
                     optimizer_path,
                     output_dir,
                     seeds=np.arange(42, 52),
                     rank=1,
                     mode='final',
                     metric='valid_accuracies'):
    """Modification from DeepOBS."""

    metric = _determine_available_metric(optimizer_path, metric)
    optimizer_path = os.path.join(optimizer_path)

    setting_analyzer_ranking = create_setting_analyzer_ranking(
        optimizer_path, mode, metric)
    setting = setting_analyzer_ranking[rank - 1]

    runner = runner(optimizer_class, hyperparam_names)

    hyperparams = setting.aggregate['optimizer_hyperparams']
    training_params = setting.aggregate['training_params']
    testproblem = setting.aggregate['testproblem']
    num_epochs = setting.aggregate['num_epochs']
    batch_size = setting.aggregate['batch_size']

    results_path = output_dir
    for seed in seeds:
        runner.run(testproblem,
                   hyperparams=hyperparams,
                   random_seed=int(seed),
                   num_epochs=num_epochs,
                   batch_size=batch_size,
                   output_dir=results_path,
                   **training_params)


class BestRunBase():
    """Rerun the best run from the grid search for multiple seeds."""
    def __init__(self, grid_search, mode, metric, output_dir="../best_run"):
        self.grid_search = grid_search
        self.mode = mode
        self.metric = metric
        self.output_dir = os.path.join(output_dir,
                                       "{}_{}".format(self.mode, self.metric))

    def get_mode(self):
        return self.mode

    def get_metric(self):
        return self.metric

    def get_runner_cls(self, extended_logs=False):
        """Use BPOptimRunnerExtendedLogging if enabled."""
        runner_cls = self.grid_search.get_runner_cls()
        if extended_logs:
            if runner_cls == BPOptimRunner:
                runner_cls = BPOptimRunnerExtendedLogging
            else:
                raise ValueError(
                    "Extended logs not supported for runner class {}".format(
                        runner_cls))
        return runner_cls

    def rerun_best(self, extended_logs=False):
        my_rerun_setting(self.get_runner_cls(extended_logs=extended_logs),
                         self.grid_search.get_optim_cls(),
                         self.grid_search.get_hyperparams(),
                         self.grid_search.get_path(),
                         self.output_dir,
                         mode=self.mode,
                         metric=self.metric)

    def rerun_best_for_seeds(self, seeds, extended_logs=False):
        my_rerun_setting(self.get_runner_cls(extended_logs=extended_logs),
                         self.grid_search.get_optim_cls(),
                         self.grid_search.get_hyperparams(),
                         self.grid_search.get_path(),
                         self.output_dir,
                         seeds=seeds,
                         mode=self.mode,
                         metric=self.metric)

    def get_output_dir(self):
        return self.output_dir

    def get_path(self):
        return os.path.join(self.output_dir,
                            self.grid_search.get_path_appended_by_deepobs())

    def get_problem_path(self):
        return os.path.join(
            self.grid_search.get_generation_dir(), self.output_dir,
            self._get_dirname(),
            self.grid_search.get_problem_path_appended_by_deepobs())

    def _get_dirname(self):
        return "{}_{}".format(self.mode, self.metric)


class BPBestRun(BestRunBase):
    pass
