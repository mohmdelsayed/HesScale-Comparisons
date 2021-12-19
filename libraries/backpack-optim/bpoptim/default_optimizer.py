from .base_optimizer import BaseOptimizer


class DefaultOptimizer(BaseOptimizer):
    def __init__(self,
                 params,
                 curvature,
                 lr=1.,
                 damping=.1,
                 beta1=0.9,
                 beta2=0.9):

        self.damping = damping
        self.step_counter = 0

        super().__init__(params,
                         defaults=dict(lr=lr,
                                       beta1=beta1,
                                       beta2=beta2,
                                       damping=damping),
                         curvature=curvature,
                         beta1=beta1,
                         beta2=beta2)
        self.group_dampings = list([
            group['damping'] + group['weight_decay']
            for group in self.param_groups
        ])


    def step(self, closure=None):
        if closure is None:
            raise ValueError("Need a closure")

        self.clear_step_info()
        self.log_step_info("damping", self.damping)
        self.zero_grad()

        # backpropagation step and update estimators
        loss, _ = self.curv.compute_curvature(closure)
        self.log_step_info("batch_loss_before_step", loss.item())

        # create curvature object
        self.inv_curv = self.curv.compute_inverse(self.group_dampings)

        # multiply dW with step size
        step_proposal = self.inv_curv.multiply([ [-group["lr"] for p in group["params"]] for group in self.param_groups])

        self.apply_step(step_proposal)
        
        return loss
