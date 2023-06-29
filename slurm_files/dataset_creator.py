import os

slurm_script = "\
#!/usr/bin/env bash\n\
#\n\
# Slurm arguments\n\
#\n\
#SBATCH --job-name=cell_dataset   # Name of the job\n\
#SBATCH --export=ALL                # Export all environment variables\n\
#SBATCH --output=cell_dataset.log   # Log-file (important!)\n\
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate\n\
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core\n\
#SBATCH --gres=gpu:0                # Number of GPU's\n\
#SBATCH --time=02:00:00              # Max execution time\n\
#\n\
\n\
# Activate your Anaconda environment\n\
conda activate geom_real  \n\
\n\
# Run your Python script\n\
cd /home/lpirenne/graph-displacement/\n\
python generate_synth_datasets.py\n\
"

with open("slurm_files/dataset.sbatch", "w") as f:
    f.write(slurm_script)

os.system("sbatch slurm_files/dataset.sbatch")