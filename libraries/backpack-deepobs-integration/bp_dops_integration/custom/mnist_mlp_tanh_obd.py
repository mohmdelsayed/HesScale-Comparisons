

# -*- coding: utf-8 -*-
"""A vanilla MLP architecture for MNIST."""

from torch import nn
import warnings
from testproblems_modules import net_mlp_obd
from mnist import mnist
from deepobs.pytorch.testproblems.testproblem import UnregularizedTestproblem

class mnist_mlp_tanh_obd(UnregularizedTestproblem):
    def __init__(self, batch_size, weight_decay=None):
        """
        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(mnist_mlp_tanh_obd, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            warnings.warn(
                "Weight decay is non-zero but no weight decay is used for this model.",
                RuntimeWarning
            )


    def set_up(self):
        """Sets up the vanilla CNN test problem on MNIST."""
        self.data = mnist(self._batch_size)
        self.loss_function = nn.NLLLoss
        self.net = net_mlp_obd(num_outputs=10, use_tanh=True)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

