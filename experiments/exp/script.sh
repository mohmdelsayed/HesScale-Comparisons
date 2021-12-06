#!/bin/bash
#SBATCH --job-name="Plot"                       # single job name for the array
#SBATCH --time=00:15:00                         # maximum wall time per job in d-hh:mm or hh:mm:ss
#SBATCH --mem=2G                               # maximum memory 100M per job
#SBATCH --account=def-ashique
#SBATCH --output=plot.out                 # standard output (%A is replaced by jobID and %a with the array index)
#SBATCH --error=plot.err                  # standard error


python3 exp05_plot_different_curvatures.py --dobs_problem mnist_logreg