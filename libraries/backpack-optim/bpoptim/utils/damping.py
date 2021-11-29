from typing import Callable


class DampingScheme:
    """
    A base class for damping schemes
    """

    def update(self, loss_before, loss_after_func: Callable, predicted_improvement):
        """
        The update might not be needed. `loss_after_func` should be a function
        that computes the loss after the step only if needed.
        """
        raise NotImplementedError

    def get(self):
        raise NotImplementedError


class ConstantDamping(DampingScheme):
    """ """

    def __init__(self, damping=1.0):
        self.damping = damping
        self.__validate_parameters()

    def __validate_parameters(self):
        damping_is_positive = self.damping > 0
        assert damping_is_positive

    def update(
        self,
        loss_before=None,
        loss_after_func: Callable = None,
        predicted_improvement=None,
    ):
        """
        Constant damping does not update the damping parameter.
        Parameters passed to this function are not used.
        """
        return

    def get(self):
        return self.damping
