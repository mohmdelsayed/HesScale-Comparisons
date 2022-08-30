# HesScale Comparisions

## Installation:
First, you need to have environemnt with python 3.7:
``` sh
python3.7 -m venv .hesscale_comp
source .hesscale_comp/bin/activate
# install dependencies
pip install --upgrade pip
pip install -r requirements.txt 
pip install -e libraries/backpack-optim/
pip install -e libraries/backpack-deepobs-integration/
pip install -e HesScale/
```

## Reproduce results:
### 1. Create Grid Search for all methods
``` sh
source .hesscale_comp/bin/activate
# generate the grid search values for each method
cd experiments/exp/
python exp01_grid_search.py --problem mnist_logreg_custom
# create scripts for each method to run on SLURM
cd ../grid_search_command_scripts/
./create_scripts.bash 
# schedule all experiments
./run_all_scripts.bash # Submitted batch job 64778755
# list your running jobs
sq
```

### 2. Plot the best combination of parameters for each method:
```sh
source .hesscale_comp/bin/activate
cd experiments/exp/
python exp03_plot.py
```