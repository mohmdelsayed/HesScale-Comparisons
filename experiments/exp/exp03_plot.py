import os
from bp_dops_integration.plotting import plot_optimizer_performance, plot_hyperparameter_sensitivity
import matplotlib.pyplot as plt

path = '../grid_search/cifar10_3c3d_relu/'
my_list = os.listdir(path)
colors = ['tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
        'tab:gray',
        'tab:olive',
        'tab:cyan'
        ]

fig, ax = plt.subplots(4, 1, sharex="col"); fig.set_size_inches(20, 16); ax[2].set_ylim(0.55, 0.90); ax[3].set_ylim(0.55, 0.90)
[plot_optimizer_performance(path+algo, fig=fig, ax=ax, reference_path=None, show=False, metric="valid_losses", mode="area") for algo in my_list]
fig.savefig('performance.png', dpi=300)

fig, ax = plt.subplots(3, 3, sharex="col");fig.set_size_inches(14, 16)
for i in range(3):
        for j in range(3):
                ax[i][j].set_ylim(1.0, 5.0)
        
[plot_hyperparameter_sensitivity(path+algo, color=color, metric="test_losses", show=False, fig=fig, ax=ax, mode="final") for algo, color in zip(my_list, colors)]
fig.savefig('sensitivity.png', dpi=300)

