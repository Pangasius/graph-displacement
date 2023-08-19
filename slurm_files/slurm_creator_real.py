import os

epoch = 1601

number_of_messages = [1,2,3,4]
size_of_messages = [32, 64, 128, 256, 512, 1024]
distribs = ["laplace", "normal"]
horizons = [1,2,3,4,5]
leaves = [0,1,2,3]

base_number_of_message = 2
base_size_of_message = 64
base_distrib = "normal"
base_horizon = 4
base_leave = 0

slurm_script = "\
#!/usr/bin/env bash\n\
#\n\
# Slurm arguments\n\
#\n\
#SBATCH --job-name=cell_pred_{number_of_message}_{size_of_message}_{distrib}_{horizon}_{leave}   # Name of the job\n\
#SBATCH --export=ALL                # Export all environment variables\n\
#SBATCH --output=cell_pred_{number_of_message}_{size_of_message}_{distrib}_{horizon}_{leave}.log   # Log-file (important!)\n\
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate\n\
#SBATCH --mem-per-cpu=16G            # Memory to allocate per allocated CPU core\n\
#SBATCH --gres=gpu:1                # Number of GPU's\n\
#SBATCH --time=24:00:00              # Max execution time\n\
#\n\
\n\
# Activate your Anaconda environment\n\
conda activate geom_real  \n\
\n\
# Run your Python script\n\
cd /home/lpirenne/graph-displacement/\n\
python train_and_stats_real.py --number_of_message {number_of_message} --size_of_messages {size_of_message} --epochs {epoch} --distrib {distrib} --horizon {horizon} --leave {leave}\n\
"

combination_done = []

def launch(number_of_message=base_number_of_message,
           size_of_message=base_size_of_message,
           distrib=base_distrib,
           horizon=base_horizon,
           leave=base_leave):
    
    if (number_of_message, size_of_message, distrib, horizon, leave) in combination_done :
        print("Skipped", number_of_message, size_of_message, distrib, horizon, leave)
        return
    else :
        combination_done.append((number_of_message, size_of_message, distrib, horizon, leave))
    
    with open("slurm_files/cell_pred.sbatch", "w") as f:
        f.write(slurm_script.format(number_of_message=number_of_message, size_of_message=size_of_message, epoch=epoch, distrib=distrib, horizon=horizon, leave=leave))

    os.system("sbatch slurm_files/cell_pred.sbatch")

"""
for number_of_message in number_of_messages:
    launch(number_of_message=number_of_message)
for size_of_message in size_of_messages:
    launch(size_of_message=size_of_message)
for distrib in distribs:
    launch(distrib=distrib)
for horizon in horizons:
    launch(horizon=horizon)
"""
    
for leave in leaves:
    launch(leave=leave)