import pprint

import bpoptim
import shutil, os
from .grid_search import BPGridSearch
from .tuning import (TuningBaseDamping, TuningConstantDampingNoCurvature,
                     NoTuning)

PROBLEMS = [
    # 'cifar10_3c3d_tanh',
    # 'cifar10_3c3d_relu',
    # 'cifar100_3c3d_tanh',
    # 'cifar100_3c3d_relu',
    # 'cifar100_allcnnc_tanh',
    # 'cifar100_allcnnc_relu',
    # 'fmnist_2c2d_tanh',
    # 'fmnist_2c2d_relu',
    # 'fmnist_mlp_tanh',
    # 'fmnist_mlp_relu',
    # 'mnist_2c2d_tanh',
    # 'mnist_2c2d_relu',
    'mnist_mlp_tanh',
    # 'mnist_mlp_relu',
    'mnist_mlp_tanh_obd',
    # 'mnist_logreg_custom',
    # 'mnist_logreg_custom_obd',
    # 'mnist_2c2d_relu_obd',
]

DEFAULT_TEST_PROBLEMS_SETTINGS = {
    "cifar10_3c3d_tanh": {"batch_size": 128, "num_epochs": 100},
    "cifar10_3c3d_relu": {"batch_size": 128, "num_epochs": 100},
    'cifar100_3c3d_tanh': {"batch_size": 128, "num_epochs": 200},
    'cifar100_3c3d_relu': {"batch_size": 128, "num_epochs": 200},
    "cifar100_allcnnc_tanh": {"batch_size": 256, "num_epochs": 350},
    "cifar100_allcnnc_relu": {"batch_size": 256, "num_epochs": 350},
    "mnist_2c2d_tanh": {"batch_size": 128, "num_epochs": 100},
    "mnist_2c2d_relu": {"batch_size": 128, "num_epochs": 100},
    "mnist_2c2d_relu_obd": {"batch_size": 128, "num_epochs": 100},
    "mnist_mlp_tanh": {"batch_size": 128, "num_epochs": 100},
    "mnist_mlp_tanh_obd": {"batch_size": 128, "num_epochs": 100},
    "mnist_mlp_relu": {"batch_size": 128, "num_epochs": 100},
    "mnist_mlp_relu_obd": {"batch_size": 128, "num_epochs": 100},
    "mnist_logreg_custom": {"batch_size": 128, "num_epochs": 50},
    "mnist_logreg_custom_obd": {"batch_size": 128, "num_epochs": 50},
    "fmnist_mlp_tanh": {"batch_size": 128, "num_epochs": 100},
    "fmnist_mlp_relu": {"batch_size": 128, "num_epochs": 100},
    "fmnist_2c2d_tanh": {"batch_size": 128, "num_epochs": 100},
    "fmnist_2c2d_relu": {"batch_size": 128, "num_epochs": 100},
}

class GridSearchFactory():
    DiagGGNExact = "DiagGGN"
    DiagGGNMC = "DiagGGN_MC"
    KFAC = "KFAC"
    Adam = "Adam"
    Adam2 = "Adam2"
    SGD = "SGD"
    SGD2 = "SGD2"
    AdaHessian = "AdaHessian"
    HesScaleLMAdamStyle = "HesScaleLMAdamStyle"
    HesScaleMax = "HesScaleMax"
    HesScaleRaw = "HesScaleRaw"
    HesScaleAbs = "HesScaleAbs"
    HesScaleAdamStyle = "HesScaleAdamStyle"
    HesScaleNoHessianUpdate = "HesScaleNoHessianUpdate"
    HesScaleNoGradUpdateMax = "HesScaleNoGradUpdateMax"
    HesScaleNoGradUpdateNoHessianUpdate = "HesScaleNoGradUpdateNoHessianUpdate"
    OBDAdamStyle = "OBDAdamStyle"
    OBDAbsStyle = "OBDAbs"
    HesScaleLMOptimizerAbs = "HesScaleLMOptimizerAbs"

    CURVATURES = [
        Adam,
        SGD,
        HesScaleAbs,
        HesScaleAdamStyle,
        HesScaleLMAdamStyle,
        HesScaleLMOptimizerAbs,
        OBDAbsStyle,
        DiagGGNMC,
        AdaHessian,
        
        OBDAdamStyle,
        # HesScaleMax,
        # DiagGGNExact,
        # HesScaleRaw,
        # KFAC,
        # HesScaleNoHessianUpdate,
        # HesScaleNoGradUpdateMax,
        # HesScaleNoGradUpdateNoHessianUpdate,
    ]

    CURVATURES_TUNING = {
        DiagGGNExact: NoTuning,
        DiagGGNMC: NoTuning,
        KFAC: NoTuning,
        Adam: NoTuning,
        SGD: NoTuning,
        AdaHessian: NoTuning,
        Adam2: NoTuning,
        SGD2: NoTuning,

        HesScaleAdamStyle: NoTuning,
        HesScaleMax: NoTuning,
        HesScaleNoHessianUpdate: NoTuning,
        HesScaleNoGradUpdateMax: NoTuning,
        HesScaleNoGradUpdateNoHessianUpdate: NoTuning,
        OBDAdamStyle: NoTuning,
        OBDAbsStyle: NoTuning,
        HesScaleAbs: NoTuning,
        HesScaleRaw: NoTuning,
        HesScaleLMAdamStyle: NoTuning,
        HesScaleLMOptimizerAbs: NoTuning,
    }

    CONSTANT = "const"

    DAMPINGS = [CONSTANT]

    DAMPINGS_TUNING = {
        CONSTANT: TuningBaseDamping,
    }

    DAMPED_OPTIMS = {
        (DiagGGNExact, CONSTANT): bpoptim.DiagGGNDefaultOptimizer,
        (DiagGGNMC, CONSTANT): bpoptim.DiagGGNMCDefaultOptimizer,
        
        (HesScaleMax, CONSTANT): bpoptim.HesScaleOptimizerMax,
        (HesScaleAbs, CONSTANT): bpoptim.HesScaleOptimizerAbs,
        (HesScaleRaw, CONSTANT): bpoptim.HesScaleOptimizerRaw,
        (HesScaleAdamStyle, CONSTANT): bpoptim.HesScaleOptimizerAdamStyle,
        (HesScaleLMAdamStyle, CONSTANT): bpoptim.HesScaleLMOptimizerAdamStyle,
        (HesScaleLMOptimizerAbs, CONSTANT): bpoptim.HesScaleLMOptimizerAbs,
        (HesScaleNoHessianUpdate, CONSTANT): bpoptim.HesScaleOptimizerZeroHessianUpdate,
        (HesScaleNoGradUpdateMax, CONSTANT): bpoptim.HesScaleOptimizerNoGradUpdateMax,
        (HesScaleNoGradUpdateNoHessianUpdate, CONSTANT): bpoptim.HesScaleOptimizerNoGradUpdateZeroHessianUpdate,

        (Adam, CONSTANT): bpoptim.AdamDefaultOptimizer,
        (Adam2, CONSTANT): bpoptim.Adam2DefaultOptimizer,

        (SGD, CONSTANT): bpoptim.SGDDefaultOptimizer,
        (SGD2, CONSTANT): bpoptim.SGD2DefaultOptimizer,

        (KFAC, CONSTANT): bpoptim.KFACDefaultOptimizer,
        (AdaHessian, CONSTANT): bpoptim.AdaHessDefaultOptimizer,
        (OBDAdamStyle, CONSTANT): bpoptim.OBDAdamStyle,
        (OBDAbsStyle, CONSTANT): bpoptim.OBDAbsStyle,
    }

    def make_grid_search(self,
                         curv_str,
                         damping_str,
                         deepobs_problem,
                         output_dir="../grid_search",
                         generation_dir="../grid_search_command_scripts",
                         copy_files=False):
        optim_cls = self._get_damped_optimizer(curv_str, damping_str)
        tune_curv, tune_damping = self.get_tunings(curv_str, damping_str)
        if copy_files:
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
        if (curv_str == self.SGD or curv_str == self.SGD2) and damping_str == self.CONSTANT:
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
