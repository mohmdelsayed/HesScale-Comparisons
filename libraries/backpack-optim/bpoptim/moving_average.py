import math
class MovingAverage():
    def __init__(self, param_groups, beta=0.9, use_factors=True, adam_style=False):
        """
        Bias-corrected moving average (Adam-style)

        If `use_factors` is `False`, the `step` function will expect
        observations of the form
        ```
        new_obs = [        # for each group
            [              # for each parameter in the group
                curv_p_11, # the curvature for that parameter
                ...
            ],
            ...
        ]
        ```
        If `use_factors` is `True`, then it will expect
        ```
        new_obs = [              # for each group
            [                    # for each parameter in the group
                [                # for each Kronecker factor for the parameter
                    kfac_1_p_11, # the Kronecker factor
                    ...
                ],
                ...
            ],
            ...
        ]
        ```
        The `get` function will return the current estimate of the curvature
        in the same format.
        """
        self.beta = beta
        self.use_factors = use_factors
        self.step_counter = 0
        self.adam_style = adam_style
        # estimate initialization:
        self.estimate = None

    def zero_decay(self, old_estimate, hess_example):
        for i, (curr_est_group, old_est_group, hess_example_group) in enumerate(zip(self.estimate, old_estimate, hess_example)):
            for j, (curr_est, old_est, hess_ex) in enumerate(zip(curr_est_group, old_est_group, hess_example_group)):
                if self.use_factors:
                    for k, (curr_est_factor, old_est_factor, hess_ex_factor) in zip(curr_est, old_est, hess_ex):
                        self.estimate[i][j][k][hess_ex_factor<0.0] =  old_est_factor[hess_ex_factor<0.0]
                else:
                    self.estimate[i][j][hess_ex<0.0] = old_est[hess_ex<0.0]
    def get(self):
        if self.adam_style:
            if self.use_factors:
                return [[[m.sqrt() / math.sqrt(self.bias_correction) for m in i] for i in x] for x in self.estimate]
            else: 
                return [[i.sqrt() / math.sqrt(self.bias_correction) for i in x] for x in self.estimate]
        else:
            if self.use_factors:
                return [[[m / self.bias_correction for m in i] for i in x] for x in self.estimate]
            else: 
                return [[i / self.bias_correction for i in x] for x in self.estimate]

    def initialize(self, new_obs):
        if self.use_factors:
            return [[[m * 0.0 for m in i] for i in x] for x in new_obs]
        else: return [[i * 0.0 for i in x] for x in new_obs]

    def __update(self, old_est, new_obs, beta):

        if self.use_factors:
            for old_est_factor, new_obs_factors in zip(old_est, new_obs):
                old_est_factor.mul_(beta).add_(new_obs_factors, alpha=1-beta)
        else:
                old_est.mul_(beta).add_(new_obs, alpha=1-beta)

    def step(self, new_obs):
        """
        If `use_factors` is `False`, the `step` function expects
        ```
        new_obs = [        # for each group
            [              # for each parameter in the group
                curv_p_11, # the curvature for that parameter
                ...
            ],
            ...
        ]
        ```
        If `use_factors` is `True`, then it expects
        ```
        new_obs = [              # for each group
            [                    # for each parameter in the group
                [                # for each Kronecker factor for the parameter
                    kfac_1_p_11, # the Kronecker factor
                    ...
                ],
                ...
            ],
            ...
        ]
        ```
        """
        if self.step_counter == 0:
            self.estimate = self.initialize(new_obs)

        self.step_counter += 1
                    
        self.bias_correction = 1 - self.beta ** self.step_counter

        for old_group, new_group in zip(self.estimate, new_obs):
            for old_est, new_obs in zip(old_group, new_group):
                self.__update(old_est, new_obs, self.beta)
        
        return self.get()
