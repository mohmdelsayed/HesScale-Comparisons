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

from .tuning_base import TuningBase

##############################################################################
# TUNABLE PARAMETERS FOR OPTIMIZERS                                          #
##############################################################################


class NoTuning(TuningBase):
    """No hyperparameters that need to be tuned"""
    def default_hyperparams(self):
        return {}

    def default_grid(self):
        return {}


class TuningZero(NoTuning):
    pass


class TuningDiagGGNExact(NoTuning):
    pass


class TuningDiagGGNMC(NoTuning):
    pass


class TuningKFAC(NoTuning):
    pass


class TuningKFLR(NoTuning):
    pass


class TuningKFRA(NoTuning):
    pass


##############################################################################
# TUNABLE PARAMETERS FOR DAMPING SCHEMES                                     #
##############################################################################


class TuningBaseDamping(TuningBase):
    """Grid search over damping scheme hyperparameters."""
    LEARNING_RATES = list(numpy.logspace(-4, 0, 5))
    DAMPINGS = [1e-8]
    BETA1s = [0.9, 0.0]
    BETA2s = [0.99, 0.999, 0.9999]
    SEEDS = random.sample(range(0, 99999), 30)

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
        return {self.SEEDS_STR: {**self.parameter_type_float()}}

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


class TuningConstantDamping(TuningBaseDamping):
    pass


class TuningConstantDampingNoCurvature(TuningConstantDamping):
    DAMPINGS = [1.]


class TuningAdaptiveDamping(TuningBaseDamping):
    MINIMUM_DAMPINGS = [1e-4]
    MINIMUM_DAMPING_STR = "minimum_damping"

    def _minimum_damping_info(self):
        return {self.MINIMUM_DAMPING_STR: {**self.parameter_type_float()}}

    def _minimum_damping_grid(self):
        return {
            self.MINIMUM_DAMPING_STR: self.MINIMUM_DAMPINGS,
        }

    def default_hyperparams(self):
        return {
            **super(TuningAdaptiveDamping, self).default_hyperparams(),
            **self._minimum_damping_info()
        }

    def default_grid(self):
        return {
            **super(TuningAdaptiveDamping, self).default_grid(),
            **self._minimum_damping_grid(),
        }


class TuningLMDamping(TuningAdaptiveDamping):
    DAMPING_STR = "initial_damping"


class TuningFancyDamping(TuningAdaptiveDamping):
    DAMPING_STR = "initial_trust_damping"

    def _learning_rate_grid(self):
        return {}

    def _learning_rate_info(self):
        return {}
