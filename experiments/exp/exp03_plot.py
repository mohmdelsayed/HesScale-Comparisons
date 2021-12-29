import os
from bp_dops_integration.plotting import plot_optimizer_performance, plot_hyperparameter_sensitivity
import matplotlib.pyplot as plt

path = '../grid_search/mnist_mlp_tanh/'
my_list = os.listdir(path)
metric = "valid_accuracies"

fig, ax = plt.subplots(4, 1, sharex="col"); fig.set_size_inches(20, 16); ax[2].set_ylim(0.8, 1.0); ax[3].set_ylim(0.8, 1.0)
[plot_optimizer_performance(path+algo, fig=fig, ax=ax, reference_path=None, show=False, metric=metric, mode='area') for algo in my_list]
fig.savefig('performance.png', dpi=300)

fig, ax = plt.subplots(3, 3, sharex="col");fig.set_size_inches(14, 16)
[plot_hyperparameter_sensitivity(path+algo, metric=metric, show=False, fig=fig, ax=ax) for algo in my_list]
fig.savefig('sensitivity.png', dpi=300)

