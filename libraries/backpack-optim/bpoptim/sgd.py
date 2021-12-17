from torch.optim.optimizer import Optimizer
import torch, math


class SGDOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr=0.15,
        beta1=0.9,
        beta2=0.999,
        damping=1e-8,
        weight_decay=0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= damping:
            raise ValueError("Invalid damping value: {}".format(damping))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(beta2))

        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, damping=damping, weight_decay=weight_decay
        )

        super(SGDOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):

        self.zero_grad()
        loss, output = closure()
        loss.backward()

        for group in self.param_groups:
            for p in group["params"]:
                p.data.add_(p.grad, alpha=-group["lr"])

        return loss
