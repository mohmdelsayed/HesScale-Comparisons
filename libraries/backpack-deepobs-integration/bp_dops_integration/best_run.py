import os
import numpy as np
from bp_dops_integration.plotting_utils import create_setting_analyzer_ranking, _determine_available_metric

class BPBestRun:
    def __init__(self, grid_search, mode, metric):
        self.grid_search = grid_search
        self.mode = mode
        self.metric = metric

    def get_mode(self):
        return self.mode

    def get_metric(self):
        return self.metric

    def get_best_config(self, extended_logs=False):
        return self.best_config(self.grid_search.get_optim_cls(),
                         self.grid_search.get_hyperparams(),
                         self.grid_search.get_path(),
                         mode=self.mode,
                         metric=self.metric)

    def best_config(self,
                    optimizer_class,
                    hyperparam_names,
                    optimizer_path,
                    seeds=np.arange(42, 52),
                    rank=1,
                    mode='final',
                    metric='valid_accuracies'):
        metric = _determine_available_metric(optimizer_path, metric)
        optimizer_path = os.path.join(optimizer_path)

        setting_analyzer_ranking = create_setting_analyzer_ranking(
            optimizer_path, mode, metric)
        setting = setting_analyzer_ranking[rank - 1]
        return setting
