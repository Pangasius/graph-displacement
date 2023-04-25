#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=cell_mov_pred            # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=cell_mov_pred.log   # Log-file (important!)
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:2                # Number of GPU's
#SBATCH --time=09:00:00              # Max execution time
#

# Activate your Anaconda environment
conda activate geom_ff  

# Run your Python script
cd /home/lpirenne/graph-displacement/
python cell_trainer_job.py --number_of_messages 5 --size_of_messages 128 --epochs 200