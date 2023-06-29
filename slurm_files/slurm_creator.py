import os

extensions = ["_open_ht_hv", "_open_lt_hv", "_open_lt_lv", "_open_ht_lv"]
number_of_messages = [1,2,3,4]
size_of_messages = [64, 128]
epochs = [51]
distribs = ["laplace", "normal"]
out_channels = [4,8]
horizons = [1,5]

slurm_script = "\
#!/usr/bin/env bash\n\
#\n\
# Slurm arguments\n\
#\n\
#SBATCH --job-name=cell_pred_{number_of_message}_{size_of_message}_{distrib}_{out}_{horizon}   # Name of the job\n\
#SBATCH --export=ALL                # Export all environment variables\n\
#SBATCH --output=cell_pred_{number_of_message}_{size_of_message}_{distrib}_{out}_{horizon}.log   # Log-file (important!)\n\
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate\n\
#SBATCH --mem-per-cpu=10G            # Memory to allocate per allocated CPU core\n\
#SBATCH --gres=gpu:1                # Number of GPU's\n\
#SBATCH --time=8:00:00              # Max execution time\n\
#\n\
\n\
# Activate your Anaconda environment\n\
conda activate geom_real  \n\
\n\
# Run your Python script\n\
cd /home/lpirenne/graph-displacement/\n\
python train_and_stats.py --extension {extension} --number_of_message {number_of_message} --size_of_messages {size_of_message} --epochs {epoch} --distrib {distrib} --out {out} --horizon {horizon}\n\
"

for extension in extensions:
    for number_of_message in number_of_messages:
        for size_of_message in size_of_messages:
            for epoch in epochs:
                for distrib in distribs:
                    for out in out_channels:
                        for horizon in horizons:
                            with open("slurm_files/cell_pred.sbatch", "w") as f:
                                f.write(slurm_script.format(extension=extension, number_of_message=number_of_message, size_of_message=size_of_message, epoch=epoch, distrib=distrib, out=out, horizon=horizon))
                        
                            os.system("sbatch slurm_files/cell_pred.sbatch")
