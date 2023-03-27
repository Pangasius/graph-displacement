import os

""" First create a list of all the jobs you want to run. This is a list of dictionaries. Each dictionary contains the parameters for one job. """
jobs = []
for number_of_messages in [1,2,3,4,5]:
    for size_of_messages in [64,128]:
        for epochs in [150]:
            jobs.append({'number_of_messages':number_of_messages,'size_of_messages':size_of_messages,'epochs':epochs})

""" Now create a slurm file for each job. """
for job in jobs:
    job_name = 'cell_mov_pred' + '_' + str(job['number_of_messages']) + '_' + str(job['size_of_messages'])


    with open('slurm_PATTERN.pat','r') as f:
        slurm_file = f.read()
    slurm_file = slurm_file.replace('cell_mov_pred',job_name)
    slurm_file = slurm_file.replace('5',str(job['number_of_messages']))
    slurm_file = slurm_file.replace('128',str(job['size_of_messages']))
    slurm_file = slurm_file.replace('200',str(job['epochs']))
    
    
    if job['size_of_messages'] == 64:
        slurm_file = slurm_file.replace('--gres=gpu:2', '--gres=gpu:1')
    
    with open(job_name+'.sbatch','w') as f:
        f.write(slurm_file)
        
    os.system('sbatch '+job_name+'.sbatch')
    os.system('rm '+job_name+'.sbatch')

os.system('squeue --me')