import os

extensions = ["_open_ht_hv", "_open_ht_lv", "_open_lt_hv", "_open_lt_lv"]
number_of_messages = [2]
size_of_messages = [64]
epochs = [51]
distribs = ["laplace", "normal"]

slurm_script = "\
#!/usr/bin/env bash\n\
#\n\
# Slurm arguments\n\
#\n\
#SBATCH --job-name=cell_mov_pred{extension}_{distrib}            # Name of the job\n\
#SBATCH --export=ALL                # Export all environment variables\n\
#SBATCH --output=cell_mov_pred{extension}_{distrib}.log   # Log-file (important!)\n\
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate\n\
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core\n\
#SBATCH --gres=gpu:1                # Number of GPU's\n\
#SBATCH --time=8:00:00              # Max execution time\n\
#\n\
\n\
# Activate your Anaconda environment\n\
conda activate geom_ff  \n\
\n\
# Run your Python script\n\
cd /home/lpirenne/graph-displacement/\n\
python train_and_stats.py --extension {extension} --number_of_message {number_of_messages} --size_of_messages {size_of_messages} --epochs {epochs} --distrib {distrib}\n\
"

for extension in extensions:
    for number_of_message in number_of_messages:
        for size_of_message in size_of_messages:
            for epoch in epochs:
                for distrib in distribs:
                    with open("slurm_files/cell_mov_pred" + extension + ".sbatch", "w") as f:
                        f.write(slurm_script.format(extension=extension, number_of_messages=number_of_message, size_of_messages=size_of_message, epochs=epoch, distrib=distrib))
                    
                    os.system("sbatch slurm_files/cell_mov_pred" + extension + ".sbatch")
