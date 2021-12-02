#!/bin/bash
#SBATCH --array=1-30
#SBATCH --job-name="myprog"                       # single job name for the array
#SBATCH --time=02:00:00                         # maximum wall time per job in d-hh:mm or hh:mm:ss
#SBATCH --mem=100                               # maximum memory 100M per job
#SBATCH --account=def-ashique
#SBATCH --output=myprog%A%a.out                 # standard output (%A is replaced by jobID and %a with the array index)
#SBATCH --error=myprog%A%a.err                  # standard error
FILE="jobs_DiagGGNMCConstantDampingOptimizer_grid_search_localtestproblem.txt"
SCRIPT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $FILE)
$SCRIPT