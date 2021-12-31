import sys

from bp_dops_integration.best_run import BPBestRun
from exp01_grid_search import create_grid_search
from bp_dops_integration.experiments import GridSearchFactory
from utils import DEFAULT_TEST_PROBLEMS_SETTINGS, allowed_combinations

PROBLEMS = [
    # 'cifar10_3c3d_tanh',
    # 'cifar10_3c3d_relu',
    # 'cifar100_3c3d_tanh',
    # 'cifar100_3c3d_relu',
    # 'cifar100_allcnnc_tanh',
    # 'cifar100_allcnnc_relu',
    'fmnist_2c2d_tanh',
    'fmnist_2c2d_relu',
    'fmnist_mlp_tanh',
    'fmnist_mlp_relu',
    # 'mnist_2c2d_tanh',
    # 'mnist_2c2d_relu',
    # 'mnist_mlp_tanh',
    # 'mnist_mlp_relu',
    # 'mnist_mlp_tanh_obd',
    # 'mnist_logreg_custom',
    # 'mnist_logreg_custom_obd',
]
def create_runscripts(filter_func=None):
    """Write the runscripts to execute all experiments."""
    for problem in PROBLEMS:
        for search in create_grid_search(filter_func=filter_func):
            grid = search._get_grid()
            best_run = BPBestRun(search, "final", "valid_accuracies")
            setting = best_run.get_best_config(extended_logs=False)
            search.deepobs_problem = problem
            for param in ["beta1", "beta2", "lr"]:
                grid[param] = [setting.aggregate['optimizer_hyperparams'][param]]
            search.create_runscript_multi_batch(DEFAULT_TEST_PROBLEMS_SETTINGS, grid=grid)


def create_grid_search(filter_func=None):
    """Return list of grid searches for all experiments.

    Allow filtering by specifying `filter_func`. It maps a tuple of strings
    for (curvature, damping, problem) to a boolean value which specifies
    whether the experiment should be included or not
    """
    factory = GridSearchFactory()
    experiments = []
    for (curv, damp, prob) in allowed_combinations(filter_func=filter_func):
        experiments.append(factory.make_grid_search(curv, damp, prob))
    return experiments


if __name__ == "__main__":
    from control import make_filter_func

    filter_func = make_filter_func()

    create_runscripts(filter_func=filter_func)
