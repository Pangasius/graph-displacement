"""This script is used to download the data from the server. """


import os

import paramiko
from scp import SCPClient

import getpass

import glob

password = getpass.getpass("Entering password: ")

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

ssh = createSSHClient("master.alan.priv", 22, "lpirenne", password)
if ssh is None:
    print("Connection failed")
    raise SystemExit

scp = SCPClient(ssh.get_transport()) # type: ignore
 
#create a directory to store the downloaded files
if not os.path.exists('data'):
    os.makedirs('data')
    
if not os.path.exists('data/train'):
    os.makedirs('data/train')
    
if not os.path.exists('data/test'):
    os.makedirs('data/test')
    
if not os.path.exists('data/valid'):
    os.makedirs('data/valid')
    
#download the training data, the paths of the files to download are in the train.txt file
#there is a lot more than needed so we will count the size of the files, and once we reach one gigabyte, we will stop

def download_files(which_file = 'train'):
    max_size = 1e9 #1 gigabyte
    
    if which_file == 'train':
        max_size = 1e9
    elif which_file == 'test':
        max_size = 1e9 / 3
    elif which_file == 'valid':
        max_size = 1e9 / 4
    else:
        print('Wrong file name')
        return
     
    #files already present in the directory
    files = glob.glob('data/' + which_file + '/*')
    files = [file.replace('\\', '/') for file in files]

    #open the file
    with open(which_file + '.txt') as f:
        #read the lines
        lines = f.readlines()
        #initialize the counter for the size of the files
        counter = 0
        #loop over the lines
        for line in lines:
            #get the path of the file
            path = line.split(' ')[0]
            #download the file using "scp -C lpirenne@alan:path data/train"
            
            if not ('data/' + which_file + '/' + path.split('/')[-1].replace('\n', '') in files):
                scp.get(path.replace('\n', ''), 'data/' + which_file + '/' + path.split('/')[-1].replace('\n', ''))
            else :
                print('File already present' + path.split('/')[-1].replace('\n', ''))
            
            #get the size of the file
            size = os.path.getsize('data/' + which_file + '/' + path.split('/')[-1].replace('\n', ''))
            
            #add the size to the counter
            counter += size
            
            print(counter, 'bytes       ', end='\r')
            
            #if the counter is greater than the maximum size, we stop
            if counter > max_size:
                break
            
download_files('train')
download_files('test')
download_files('valid')

scp.close()
         