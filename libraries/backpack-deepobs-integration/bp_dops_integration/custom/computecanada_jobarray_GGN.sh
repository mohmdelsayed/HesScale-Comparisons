#!/bin/bash
#SBATCH --array=1-30
#SBATCH --job-name="GGN_LocalTestProblem"                       # single job name for the array
#SBATCH --time=02:00:00                         # maximum wall time per job in d-hh:mm or hh:mm:ss
#SBATCH --mem=4G                               # maximum memory 100M per job
#SBATCH --account=def-ashique
#SBATCH --output=%x%A%a.out                 # standard output (%A is replaced by jobID and %a with the array index)
#SBATCH --error=%x%A%a.err                  # standard error


FILE="$SCRATCH/HesScale-Comparisons/experiments/grid_search_command_scripts/jobs_DiagGGNConstantDampingOptimizer_grid_search_localtestproblem.txt"
SCRIPT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $FILE)
module load python/3.7.9
source $SCRATCH/HesScale-Comparisons/env/bin/activate
$SCRIPT