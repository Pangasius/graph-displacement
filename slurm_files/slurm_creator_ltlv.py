import os

epoch = 51

number_of_messages = [1,2,3,4]
size_of_messages = [64, 128, 256]
horizons = [1,2,3,4,5]

base_extension = "_open_lt_lv"
base_number_of_message = 2
base_size_of_message = 256
base_distrib = "laplace"
base_out = 8
base_horizon = 2

slurm_script = "\
#!/usr/bin/env bash\n\
#\n\
# Slurm arguments\n\
#\n\
#SBATCH --job-name=cell_pred_{extension}_{number_of_message}_{size_of_message}_{distrib}_{out}_{horizon}   # Name of the job\n\
#SBATCH --export=ALL                # Export all environment variables\n\
#SBATCH --output=cell_pred_{extension}_{number_of_message}_{size_of_message}_{distrib}_{out}_{horizon}.log   # Log-file (important!)\n\
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate\n\
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core\n\
#SBATCH --gres=gpu:1                # Number of GPU's\n\
#SBATCH --time=10:00:00              # Max execution time\n\
#\n\
\n\
# Activate your Anaconda environment\n\
conda activate geom_real  \n\
\n\
# Run your Python script\n\
cd /home/lpirenne/graph-displacement/\n\
python train_and_stats.py --extension {extension} --number_of_message {number_of_message} --size_of_messages {size_of_message} --epochs {epoch} --distrib {distrib} --out {out} --horizon {horizon}\n\
"

combination_done = []

def launch(extension=base_extension, 
           number_of_message=base_number_of_message,
           size_of_message=base_size_of_message,
           distrib=base_distrib,
           out=base_out, 
           horizon=base_horizon):
    
    if (extension, number_of_message, size_of_message, distrib, out, horizon) in combination_done :
        print("Skipped", extension, number_of_message, size_of_message, distrib, out, horizon)
        return
    else :
        combination_done.append((extension, number_of_message, size_of_message, distrib, out, horizon))
    
    with open("slurm_files/cell_pred.sbatch", "w") as f:
        f.write(slurm_script.format(extension=extension, number_of_message=number_of_message, size_of_message=size_of_message, epoch=epoch, distrib=distrib, out=out, horizon=horizon))

    os.system("sbatch slurm_files/cell_pred.sbatch")

for number_of_message in number_of_messages:
    launch(number_of_message=number_of_message)
for size_of_message in size_of_messages:
    launch(size_of_message=size_of_message)
for horizon in horizons:
    launch(horizon=horizon)