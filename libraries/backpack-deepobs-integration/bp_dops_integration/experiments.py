import pprint

import bpoptim
import shutil, os
from .grid_search import BPGridSearch
from .tuning import (TuningConstantDamping, TuningConstantDampingNoCurvature,
                     TuningDiagGGNExact, TuningDiagGGNMC, TuningFancyDamping,
                     TuningKFAC, TuningKFLR, TuningKFRA, TuningLMDamping, 
                     TuningZero)

PROBLEMS = [
    'cifar10_3c3d_custom',
    'cifar100_3c3d_custom',
    'cifar100_allcnnc_custom',
    'fmnist_2c2d_custom',
    'fmnist_mlp_custom',
    'mnist_2c2d_custom',
    'mnist_mlp_custom',
    'mnist_logreg_custom',
]

DEFAULT_TEST_PROBLEMS_SETTINGS = {
    "cifar10_3c3d_custom": {"batch_size": 128, "num_epochs": 100},
    'cifar100_3c3d_custom': {"batch_size": 128, "num_epochs": 200},
    "cifar100_allcnnc_custom": {"batch_size": 256, "num_epochs": 350},
    "mnist_2c2d_custom": {"batch_size": 128, "num_epochs": 100},
    "mnist_mlp_custom": {"batch_size": 128, "num_epochs": 100},
    "mnist_logreg_custom": {"batch_size": 128, "num_epochs": 50},
    "fmnist_mlp_custom": {"batch_size": 128, "num_epochs": 100},
    "fmnist_2c2d_custom": {"batch_size": 128, "num_epochs": 100},
}

class GridSearchFactory():
    Zero = "Zero"
    DiagGGNExact = "DiagGGN"
    DiagGGNMC = "DiagGGN_MC"
    HesScaleAbs = "HesScaleAbs"
    HesScaleMax = "HesScaleMax"
    KFAC = "KFAC"
    KFLR = "KFLR"
    CURVATURES = [
        # Zero,
        DiagGGNMC,
        DiagGGNExact,
        HesScaleAbs,
        HesScaleMax,
        KFAC,
        KFLR,
    ]

    CURVATURES_TUNING = {
        Zero: TuningZero,
        DiagGGNExact: TuningDiagGGNExact,
        DiagGGNMC: TuningDiagGGNMC,
        HesScaleAbs: TuningDiagGGNMC,
        HesScaleMax: TuningDiagGGNMC,
        KFAC: TuningKFAC,
        KFLR: TuningKFLR,
    }

    CONSTANT = "const"

    DAMPINGS = [CONSTANT]

    DAMPINGS_TUNING = {
        CONSTANT: TuningConstantDamping,
    }

    DAMPED_OPTIMS = {
        (Zero, CONSTANT): bpoptim.ZeroConstantDampingOptimizer,
        (DiagGGNExact, CONSTANT): bpoptim.DiagGGNConstantDampingOptimizer,
        (DiagGGNMC, CONSTANT): bpoptim.DiagGGNMCConstantDampingOptimizer,
        (HesScaleAbs, CONSTANT): bpoptim.HesScaleConstantDampingOptimizerAbs,
        (HesScaleMax, CONSTANT): bpoptim.HesScaleConstantDampingOptimizerMax,
        (KFAC, CONSTANT): bpoptim.KFACConstantDampingOptimizer,
        (KFLR, CONSTANT): bpoptim.KFLRConstantDampingOptimizer,
    }

    def make_grid_search(self,
                         curv_str,
                         damping_str,
                         deepobs_problem,
                         output_dir="../grid_search",
                         generation_dir="../grid_search_command_scripts"):
        optim_cls = self._get_damped_optimizer(curv_str, damping_str)
        tune_curv, tune_damping = self.get_tunings(curv_str, damping_str)
        if not os.path.exists(generation_dir):
            os.mkdir(generation_dir)
        srcs = "../../libraries/backpack-deepobs-integration/bp_dops_integration/custom/"
        files = os.listdir(srcs)
        for f in files:
            if os.path.isfile(srcs+f):
                shutil.copy(srcs+f, generation_dir)
        return BPGridSearch(deepobs_problem,
                            optim_cls,
                            tune_curv,
                            tune_damping,
                            output_dir=output_dir,
                            generation_dir=generation_dir)

    def get_tunings(self, curv_str, damping_str):
        tune_curv = self._get_curv_tuning(curv_str)
        tune_damping = self._get_damping_tuning(damping_str)

        # no tuning of damping parameter for constant damping and no curvature
        if curv_str == self.Zero and damping_str == self.CONSTANT:
            tune_damping = TuningConstantDampingNoCurvature()

        return tune_curv, tune_damping

    def get_curvature_and_damping(self, optim_cls):
        """Return (curvature, damping) from the optimizer class."""
        for (curv, damp), cls in self.DAMPED_OPTIMS.items():
            if optim_cls == cls:
                return (curv, damp)
        raise ValueError(
            "No (curvature, damping) found for {}".format(optim_cls))

    def get_all_optim_classes(self):
        return [optim_cls for (_, optim_cls) in self.DAMPED_OPTIMS.items()]

    def _get_damped_optimizer(self, curv_str, damping_str):
        key = self._check_damped_optim_exists(curv_str, damping_str)
        return self.DAMPED_OPTIMS[key]

    def _get_damping_tuning(self, damping_str):
        key = self._check_damping_tuning_exists(damping_str)
        return self.DAMPINGS_TUNING[key]()

    def _get_curv_tuning(self, curv_str):
        key = self._check_curv_tuning_exists(curv_str)
        return self.CURVATURES_TUNING[key]()

    def _check_curv_tuning_exists(self, curv_str):
        if curv_str not in self.CURVATURES_TUNING.keys():
            raise ValueError(
                "Curvature tuning {} not registered. Supported: {}.".format(
                    curv_str, pprint.pformat(self.CURVATURES_TUNING.keys())))
        return curv_str

    def _check_damping_tuning_exists(self, damping_str):
        if damping_str not in self.DAMPINGS_TUNING.keys():
            raise ValueError(
                "Damping tuning {} not registered. Supported: {}.".format(
                    damping_str, pprint.pformat(self.DAMPINGS_TUNING.keys())))
        return damping_str

    def _check_damped_optim_exists(self, curv_str, damping_str):
        key = (curv_str, damping_str)
        if key not in self.DAMPED_OPTIMS.keys():
            raise ValueError(
                "Damped optimizer {} not registered. Supported: {}.".format(
                    key, pprint.pformat(self.DAMPED_OPTIMS.keys())))
        return key
