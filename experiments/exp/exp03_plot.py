from deepobs.analyzer.analyze import plot_optimizer_performance, plot_hyperparameter_sensitivity
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

path = '../grid_search/mnist-logreg-custom/'
my_list = os.listdir(path)[:-7]

fig, ax = plt.subplots(4, 1, sharex="col")
[plot_optimizer_performance(path+algo, fig=fig, ax=ax, reference_path=None, show=False, metric='test_accuracies') for algo in my_list]
plt.show()

fig, ax = plt.subplots(1, 1, sharex="col")   
[plot_hyperparameter_sensitivity(path+algo, hyperparam='lr', xscale='log', plot_std=True, metric="test_accuracies", show=False, fig=fig, ax=ax) for algo in my_list]
plt.show()
