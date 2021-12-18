"""
Curvature estimators

The role of those classes is to
- compute the curvature at each step.
- maintain a running average of the quantities.
- when needed, compute the inverse of the curvature using some damping.

"""

import math
from torch import einsum, symeig, tensor
from torch import max as torch_max
from backpack import backpack
from backpack.extensions import DiagGGNExact, DiagGGNMC, KFAC, KFLR, KFRA

from bpoptim.utils import multiply_vec_with_kron_facs
from hesscale import HesScale

from .utils import NUMERICAL_STABILITY_CONSTANT
from .inverse_curvature import (
    DiagonalInverseCurvature,
    KroneckerInverseCurvature,
)
from .moving_average import MovingAverage


class CurvatureEstimator:
    def __init__(self, param_groups, beta1, beta2):
        """
        param_groups: the param_groups of the optimizer using the estimator
        """
        self.param_groups = param_groups
        self.beta1 = beta1
        self.beta2 = beta2

    def compute_curvature(self, closure):
        """
        Update the running average of curvature.

        Closure should zero the gradients, do the forward pass
        and return the loss
        """
        raise NotImplementedError

    def compute_inverse(self, damping):
        """
        Returns an instance of InverseCurvature.

        `dampings` should be a list of dampings for each `param_groups`
        """
        raise NotImplementedError

class SgdCurvature:
    def __init__(self, param_groups, beta1=0.0, beta2=0.0):
        self.param_groups = param_groups

    def compute_curvature(self, closure, retain_graph=False):
        loss, output = closure()
        loss.backward(retain_graph=retain_graph)
        return loss, output

    def compute_inverse(self, damping=0.0):
        grad = list(
            [
                list([p.grad for p in group["params"]])
                for group in self.param_groups
            ]
        )
        curv = list(
            [
                list([1.0 for p in group["params"]])
                for group in self.param_groups
            ]
        )
        return DiagonalInverseCurvature(curv, grad)

class AdamCurvature(CurvatureEstimator):
    def __init__(self, param_groups, beta1, beta2):
        super().__init__(param_groups, beta1, beta2)
        self.avg_grad = MovingAverage(param_groups, beta=beta1, use_factors=False)
        self.avg_hessian = MovingAverage(param_groups, beta=beta2, use_factors=False, adam_style=True)

    def compute_curvature(self, closure, retain_graph=False):
        loss, output = closure()
        loss.backward(retain_graph=retain_graph)

        input_to_avg_hessian = list(
            [
                list([p.grad.data ** 2 for p in group["params"]])
                for group in self.param_groups
            ]
        )
        input_to_avg_grad = list(
            [
                list([p.grad.data for p in group["params"]])
                for group in self.param_groups
            ]
        )

        self.avg_hessian.step(input_to_avg_hessian)
        self.avg_grad.step(input_to_avg_grad)

        return loss, output

    def compute_inverse(self, damping):
        curv = self.avg_hessian.get()
        grad = self.avg_grad.get()
            
        inv_curv = []
        for curv_group, damping_group in zip(curv, damping):
            inv_curv.append([])
            for curv_p in curv_group:
                inv_curv[-1].append(1 / (curv_p + damping_group))
        return DiagonalInverseCurvature(inv_curv, grad)


class BackpackCurvatureEstimator(CurvatureEstimator):
    def __init__(self, param_groups, beta1, beta2, bp_extension_cls, use_factors, adam_style=False):
        super().__init__(param_groups, beta1, beta2)
        self.avg_hessian = MovingAverage(param_groups, beta=beta2, use_factors=use_factors, adam_style=adam_style)
        self.avg_grad = MovingAverage(param_groups, beta=beta1, use_factors=False)
        self.bp_extension_cls = bp_extension_cls

    def compute_curvature(self, closure, retain_graph=False):
        """Data structure for moving average is supported by backpack."""
        bp_extension = self.bp_extension_cls()
        bp_savefield = bp_extension.savefield

        with backpack(bp_extension):
            loss, output = closure()
            loss.backward(retain_graph=retain_graph)

            input_to_avg_hessian = list(
                [
                    list([getattr(p, bp_savefield) for p in group["params"]])
                    for group in self.param_groups
                ]
            )
            input_to_avg_grad = list(
                [
                    list([p.grad for p in group["params"]])
                    for group in self.param_groups
                ]
            )

            self.avg_hessian.step(input_to_avg_hessian)
            self.avg_grad.step(input_to_avg_grad)

        return loss, output


class DiagCurvatureBase(BackpackCurvatureEstimator):
    def __init__(self, param_groups, beta1, beta2, bp_extension_cls, adam_style=False):
        use_factors = False
        super().__init__(param_groups, beta1, beta2, bp_extension_cls, use_factors, adam_style=adam_style)

    def compute_inverse(self, damping):
        curv = self.avg_hessian.get()
        grad = self.avg_grad.get()

        inv_curv = []
        for curv_group, damping_group in zip(curv, damping):
            inv_curv.append([])
            for curv_p in curv_group:
                inv_curv[-1].append(1 / (curv_p + damping_group))
        return DiagonalInverseCurvature(inv_curv, grad)

class DiagGGNExactCurvature(DiagCurvatureBase):
    def __init__(self, param_groups, beta1, beta2, adam_style=False):
        super().__init__(param_groups, beta1, beta2, DiagGGNExact, adam_style=False)


class DiagGGNMCCurvature(DiagCurvatureBase):
    def __init__(self, param_groups, beta1, beta2, adam_style=False):
        super().__init__(param_groups, beta1, beta2, DiagGGNMC, adam_style=False)

class HesScaleCurvatureBase(DiagCurvatureBase):
    def __init__(self, param_groups, beta1, beta2, bp_extension_cls, style='max'):
        self.style = style        
        self.adam_style = True if style == 'adam' else False
        super().__init__(param_groups, beta1, beta2, bp_extension_cls, adam_style=self.adam_style)

    def compute_curvature(self, closure, retain_graph=False):
        """Data structure for moving average is supported by backpack."""
        bp_extension = self.bp_extension_cls()
        bp_savefield = bp_extension.savefield

        with backpack(bp_extension):
            loss, output = closure()
            loss.backward(retain_graph=retain_graph)

            input_to_avg_grad = list(
                [
                    list([p.grad for p in group["params"]])
                    for group in self.param_groups
                ]
            )
            self.avg_grad.step(input_to_avg_grad)
            if self.style == "max":
                input_to_avg_hessian = list(
                    [
                        list([torch_max(getattr(p, bp_savefield), tensor([0.0], device=getattr(p, bp_savefield).device)) for p in group["params"]])
                        for group in self.param_groups
                    ]
                )
            elif self.adam_style:
                input_to_avg_hessian = list(
                    [
                        list([getattr(p, bp_savefield) ** 2 for p in group["params"]])
                        for group in self.param_groups
                    ]
                )
            else:
                input_to_avg_hessian = list(
                    [
                        list([getattr(p, bp_savefield) for p in group["params"]])
                        for group in self.param_groups
                    ]
                )
            if self.style == "no_h_update":
                if self.avg_hessian.step_counter == 0:
                    old_estimate = self.avg_hessian.initialize(input_to_avg_hessian)
                else:
                    old_estimate = self.avg_hessian.get()
                    
                self.avg_hessian.step(input_to_avg_hessian)
                self.avg_hessian.zero_decay(old_estimate, input_to_avg_hessian)
            else:
                self.avg_hessian.step(input_to_avg_hessian)

        return loss, output, input_to_avg_hessian
class HesScaleCurvatureMax(HesScaleCurvatureBase):
    def __init__(self, param_groups, beta1, beta2):
        super().__init__(param_groups, beta1, beta2, HesScale, style='max')

class HesScaleCurvatureAdamStyle(HesScaleCurvatureBase):
    def __init__(self, param_groups, beta1, beta2):
        super().__init__(param_groups, beta1, beta2, HesScale, style='adam')
        
class HesScaleCurvatureZeroHessianUpdate(HesScaleCurvatureBase):
    def __init__(self, param_groups, beta1, beta2):
        super().__init__(param_groups, beta1, beta2, HesScale, style='no_h_update')
        
class KroneckerFactoredCurvature(BackpackCurvatureEstimator):
    def __init__(self, param_groups, beta1, beta2, bp_extension_cls):
        use_factors = True
        super().__init__(param_groups, beta1, beta2, bp_extension_cls, use_factors)

    def compute_inverse(self, damping):
        curv = self.avg_hessian.get()
        grad = self.avg_grad.get()

        inv_curv = []
        for curv_group, damping_group in zip(curv, damping):
            inv_curv.append([])
            for curv_p in curv_group:
                if len(curv_p) == 1:
                    # TODO: Not defined by KFAC
                    # HOTFIX: Just invert, shift by damping, no Tikhonov
                    shift = damping_group
                    kfac = curv_p[0]
                    inv_kfac = self.__inverse(kfac, shift=shift)

                    inv_curv[-1].append([inv_kfac])

                elif len(curv_p) == 2:
                    # G, A in Martens' notation (different order due to row-major)
                    kfac2, kfac1 = curv_p

                    # Tikhonov
                    pi = self.__compute_tikhonov_factor(kfac1, kfac2)
                    # shift for factor 1: pi * sqrt(gamma  + eta) = pi * sqrt(gamma)
                    shift1 = pi * math.sqrt(damping_group)
                    # factor 2: 1 / pi * sqrt(gamma  + eta) = 1 / pi * sqrt(gamma)
                    shift2 = 1. / pi * math.sqrt(damping_group)

                    # invert, take into account the diagonal term
                    inv_kfac1 = self.__inverse(kfac1, shift=shift1)
                    inv_kfac2 = self.__inverse(kfac2, shift=shift2)

                    inv_curv[-1].append([inv_kfac2, inv_kfac1])
                else:
                    raise ValueError(
                        "Got {} Kronecker factors, can only handle <= 2".
                        format(len(curv_p)))

        return KroneckerInverseCurvature(inv_curv, grad)

    def __compute_tikhonov_factor(self, kfac1, kfac2):
        """Scalar pi from trace norm for factored Tikhonov regularization.

        For details, see Section 6.3 of the KFAC paper.

        TODO: Allow for choices other than trace norm.
        """
        (dim1, _), (dim2, _) = kfac1.shape, kfac2.shape
        trace1, trace2 = kfac1.trace(), kfac2.trace()
        pi_squared = (trace1 / dim1) / (trace2 / dim2)
        return pi_squared.sqrt()

    def __inverse(self, sym_mat, shift):
        """Invert sym_mat + shift * I"""
        eigvals, eigvecs = self.__eigen(sym_mat)
        # account for diagonal term added to the matrix
        eigvals.add_(shift)
        return self.__inv_from_eigen(eigvals, eigvecs)

    def __eigen(self, sym_mat):
        """Return eigenvalues and eigenvectors from eigendecomposition."""
        eigvals, eigvecs = symeig(sym_mat, eigenvectors=True)
        return eigvals, eigvecs

    def __inv_from_eigen(self, eigvals, eigvecs, truncate=NUMERICAL_STABILITY_CONSTANT):
        inv_eigvals = 1. / eigvals
        inv_eigvals.clamp_(min=0., max=1. / truncate)
        # return inv_eigvals, eigvecs
        return einsum('ij,j,kj->ik', (eigvecs, inv_eigvals, eigvecs))


class KFACCurvature(KroneckerFactoredCurvature):
    """Kronecker factorization by Martens."""
    def __init__(self, param_groups, beta1, beta2):
        bp_extension_cls = KFAC
        super().__init__(param_groups, beta1, beta2, bp_extension_cls)

