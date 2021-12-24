from deepobs.analyzer.analyze import plot_optimizer_performance, plot_hyperparameter_sensitivity
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
from matplotlib.pyplot import figure

path = '../grid_search/mnist_mlp_tanh/'
my_list = os.listdir(path)
fig, ax = plt.subplots(4, 1, sharex="col")
fig.set_size_inches(20, 16)

[plot_optimizer_performance(path+algo, fig=fig, ax=ax, reference_path=None, show=False, metric='test_accuracies', mode='final') for algo in my_list]
fig.savefig('performance.png', dpi=100)


# fig, ax = plt.subplots(1, 1, sharex="col")   
# fig.set_size_inches(20, 16)
# [plot_hyperparameter_sensitivity(path+algo, hyperparam='lr', xscale='log', plot_std=True, metric="valid_accuracies", show=False, fig=fig, ax=ax) for algo in my_list]
# fig.savefig('sensitivity.png', dpi=100)

