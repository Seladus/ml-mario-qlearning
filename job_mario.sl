#!/bin/bash

# Created from: slurm submission script, serial job
# support@criann.fr

# Max time the script will run (here 48 hours)
#SBATCH --time 48:00:00

# RAM to use (Mo)
#SBATCH --mem 30000

# Number of cpu core to use
#SBATCH --cpus-per-task=1

# Enable the mailing for the start of the experiments
#SBATCH --mail-type ALL
#SBATCH --mail-user thomas.halipre@insa-rouen.fr

# Which partition to use
#SBATCH --partition insa

# Number of gpu(s) to use
#SBATCH --gres gpu:1

# Number of nodes to use
#SBATCH --nodes 1

# Log files (%J is a variable for the job id)
#SBATCH --output %J.out
#SBATCH --error %J.err

#Loading the module
module load python3-DL/3.8.5

pip install gym-super-mario-bros --user
mkdir saves

# Start the calculation
srun --cpu-freq=highm1 python3 mario_conv.py

