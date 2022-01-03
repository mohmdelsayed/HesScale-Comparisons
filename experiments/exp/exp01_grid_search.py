"""Generate grid search run scripts."""

from bp_dops_integration.experiments import GridSearchFactory
from utils import DEFAULT_TEST_PROBLEMS_SETTINGS, allowed_combinations
from control import make_filter_func


def create_runscripts(filter_func=None):
    """Write the runscripts to execute all experiments."""
    for search in create_grid_search(filter_func=filter_func, copy_files=True):
        search.create_runscript_multi_batch(DEFAULT_TEST_PROBLEMS_SETTINGS)


def create_grid_search(filter_func=None, copy_files=False):
    """Return list of grid searches for all experiments.

    Allow filtering by specifying `filter_func`. It maps a tuple of strings
    for (curvature, damping, problem) to a boolean value which specifies
    whether the experiment should be included or not
    """
    factory = GridSearchFactory()
    experiments = []
    for (curv, damp, prob) in allowed_combinations(filter_func=filter_func):
        experiments.append(factory.make_grid_search(curv, damp, prob, copy_files=copy_files))
    return experiments


if __name__ == "__main__":
    filter_func = make_filter_func()
    create_runscripts(filter_func=filter_func)
