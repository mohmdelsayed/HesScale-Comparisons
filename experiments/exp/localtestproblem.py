from deepobs.pytorch.testproblems.testproblem import UnregularizedTestproblem

# -*- coding: utf-8 -*-
"""A vanilla MLP architecture for MNIST."""

from torch import nn
from deepobs.pytorch.datasets.mnist import mnist
import warnings
    
    

class net_mlp(nn.Sequential):
    """  A basic MLP architecture. The network is build as follows:

    - Four fully-connected layers with ``1000``, ``500``,``100`` and ``num_outputs``
      units per layer, where ``num_outputs`` is the number of ouputs (i.e. class labels).
    - The first three layers use Tanh activation, and the last one a softmax
      activation.
    - The biases are initialized to ``0.0`` and the weight matrices with
      truncated normal (standard deviation of ``3e-2``)"""
    def __init__(self, num_outputs):
        super(net_mlp, self).__init__()

        self.add_module('flatten', nn.Flatten())
        self.add_module('dense1', nn.Linear(784, 1000))
        self.add_module('tanh1', nn.Tanh())
        self.add_module('dense2', nn.Linear(1000, 500))
        self.add_module('tanh2', nn.Tanh())
        self.add_module('dense3', nn.Linear(500, 100))
        self.add_module('tanh3', nn.Tanh())
        self.add_module('dense4', nn.Linear(100, num_outputs))

        # for module in self.modules():
        #     if isinstance(module, nn.Linear):
        #         nn.init.constant_(module.bias, 0.0)
        #         module.weight.data = _truncated_normal_init(module.weight.data, mean = 0, stddev=3e-2)
                
class localtestproblem(UnregularizedTestproblem):
    """DeepOBS test problem class for a multi-layer perceptron neural network\
    on Fashion-MNIST.

  The network is build as follows:

    - Four fully-connected layers with ``1000``, ``500``, ``100`` and ``10``
      units per layer.
    - The first three layers use ReLU activation, and the last one a softmax
      activation.
    - The biases are initialized to ``0.0`` and the weight matrices with
      truncated normal (standard deviation of ``3e-2``)
    - The model uses a cross entropy loss.
    - No regularization is used.

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): No weight decay (L2-regularization) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.

      Attributes:
        data: The DeepOBS data set class for Fashion-MNIST.
        loss_function: The loss function for this testproblem is torch.nn.CrossEntropyLoss()
        net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_mlp).
        """
    def __init__(self, batch_size, weight_decay=None):
        """Create a new multi-layer perceptron test problem instance on \
        Fashion-MNIST.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(localtestproblem, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            warnings.warn(
                "Weight decay is non-zero but no weight decay is used for this model.",
                RuntimeWarning
            )

    def set_up(self):
        """Sets up the vanilla CNN test problem on MNIST."""
        self.data = mnist(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_mlp(num_outputs=10)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()



#OR
# """Inform BackPACK extensions how to deal with DeepOBS-specific layers."""
# from backpack.extensions import DiagGGN
# from backpack.extensions.secondorder.diag_ggn.flatten import DiagGGNFlatten
# from deepobs.pytorch.testproblems.testproblems_utils import flatten


# def extend_deepobs_flatten():
#     """Inform BackPACK how to deal with DeepOBS flatten layer."""
#     print("[DEBUG] BackPACK: Extend DeepOBS layer flatten")
#     DiagGGN.add_module_extension(flatten, DiagGGNFlatten())
