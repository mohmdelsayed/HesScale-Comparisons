"""
Define the values for parameters of the curvature and damping scheme that
are varied in the grid search.

Notes:
------
- Hyperparameters of curvature computation (e.g. moving average) are not
  tuned in these experiments
- For the damping scheme, a reasonable grid over the damping parameter and
  and the learning rate is chosen
- No grid search over the fancy damping scheme, since it should be
  sophisticated enough to stabilize itself
"""
from contextlib import contextmanager

import numpy, random

class TuningBase():
    """Base class for tuning hyperparameters.

    Parameters:
    -----------
    hyperparams: dict
        Nested dictionary that lists the tunable hyperparameters and data
        types, e.g. {"lr": {"type": float}, ...}.
    grid : dict
        Nested dictionary mapping tunable hyperparameter names to values, e.g.
        {"lr": [0.001, 0.01, 0.1], ...}. The grid is defined by the Cartesian
        product of values
    """
    def __init__(self, hyperparams=None, grid=None):
        if hyperparams is None:
            hyperparams = self.default_hyperparams()
        self.hyperparams = hyperparams

        if grid is None:
            grid = self.default_grid()
        self.grid = grid

    def default_hyperparams(self):
        raise NotImplementedError

    def default_grid(self):
        raise NotImplementedError

    def get_hyperparams(self):
        self._verify_hyperparams()
        return self.hyperparams

    def get_grid(self):
        self._verify_grid()
        return self.grid

    def _verify_hyperparams(self):
        """Do not allow default values."""
        DEFAULT = "default"
        has_default = []
        for param, param_prop in self.hyperparams.items():
            if DEFAULT in param_prop.keys():
                has_default.append(param)

        throw_exception = len(has_default) != 0
        if throw_exception:
            raise ValueError(
                "Parameters {} have default value.".format(has_default))

    def _verify_grid(self):
        """Grid has to be specified for all parameters."""
        not_specified = []
        for param in self.hyperparams.keys():
            if param not in self.grid.keys():
                not_specified.append(param)

        throw_exception = len(not_specified) != 0
        if throw_exception:
            raise ValueError(
                "Parameters {} not specified in grid.".format(not_specified))


##############################################################################
# TUNABLE PARAMETERS FOR OPTIMIZERS                                          #
##############################################################################


class NoTuning(TuningBase):
    """No hyperparameters that need to be tuned"""
    def default_hyperparams(self):
        return {}

    def default_grid(self):
        return {}


##############################################################################
# TUNABLE PARAMETERS FOR DAMPING SCHEMES                                     #
##############################################################################


class TuningBaseDamping(TuningBase):
    """Grid search over damping scheme hyperparameters."""
    LEARNING_RATES = list(numpy.logspace(-5, -1, 5))
    DAMPINGS = [1e-8]
    BETA1s = [0.0, 0.9]
    BETA2s = [0.99, 0.999, 0.9999]
    start_seed = 1234
    n_seeds = 2
    SEEDS = range(start_seed, start_seed + n_seeds)

    LEARNING_RATE_STR = "lr"
    SEEDS_STR = "random_seed"
    DAMPING_STR = "damping"
    BETA1_STR = "beta1"
    BETA2_STR = "beta2"

    def _learning_rate_info(self):
        return {self.LEARNING_RATE_STR: {**self.parameter_type_float()}}

    def _learning_rate_grid(self):
        return {
            self.LEARNING_RATE_STR: self.LEARNING_RATES,
        }
    def _seed_info(self):
        return {self.SEEDS_STR: {**self.parameter_type_int()}}

    def _seeds_grid(self):
        return {
            self.SEEDS_STR: self.SEEDS,
        }

    def _damping_info(self):
        return {self.DAMPING_STR: {**self.parameter_type_float()}}

    def _damping_grid(self):
        return {
            self.DAMPING_STR: self.DAMPINGS,
        }

    def _beta1_info(self):
        return {self.BETA1_STR: {**self.parameter_type_float()}}

    def _beta1_grid(self):
        return {
            self.BETA1_STR: self.BETA1s,
        }

    def _beta2_info(self):
        return {self.BETA2_STR: {**self.parameter_type_float()}}

    def _beta2_grid(self):
        return {
            self.BETA2_STR: self.BETA2s,
        }
    def default_hyperparams(self):
        return {
            **self._learning_rate_info(),
            **self._damping_info(),
            **self._beta1_info(),
            **self._beta2_info(),
            **self._seed_info(),
        }

    def default_grid(self):
        return {
            **self._learning_rate_grid(),
            **self._damping_grid(),
            **self._beta1_grid(),
            **self._beta2_grid(),
            **self._seeds_grid(),
        }

    @staticmethod
    def parameter_type_float():
        return {"type": float}
    
    @staticmethod
    def parameter_type_int():
        return {"type": int}


@contextmanager
def use_1d_dummy_grid_for_damping(lrs=[0.1234], dampings=[5.678]):
    """Use one learning rate and one damping for debugging."""
    orig_lrs = TuningBaseDamping.LEARNING_RATES
    orig_dampings = TuningBaseDamping.DAMPINGS

    try:
        TuningBaseDamping.LEARNING_RATES = lrs
        TuningBaseDamping.DAMPINGS = dampings
        yield None
    except Exception as e:
        raise e
    finally:
        TuningBaseDamping.LEARNING_RATES = orig_lrs
        TuningBaseDamping.DAMPINGS = orig_dampings


class TuningConstantDampingNoCurvature(TuningBaseDamping):
    BETA1s = [0.0]
    BETA2s = [0.0]

