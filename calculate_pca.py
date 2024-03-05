from argparse import ArgumentParser
import torch
from pathlib import Path
import numpy as np
import os

def get_embeddings(eval_path, embedding):
    emb_dir = Path(eval_path) / "embeddings" / embedding
    eval_files = list(emb_dir.glob("*.npy"))
    cur_embds = None
    for file in eval_files:
        if cur_embds is None:
            cur_embds = np.load(file)
            cur_embds_shape = cur_embds.shape
        else:
            cur_embds = np.concatenate((cur_embds, np.load(file)), axis=0)
    cur_embds = torch.from_numpy(cur_embds)
    cur_embds = cur_embds.float()
    print(cur_embds)
    return(cur_embds, cur_embds_shape)

embeddings = ['panns-wavegram-logmel', 'clap-2023']
systems = ['TBSys09', 'TBSys18', 'TBSys14', 'TBSys24', 'TASys08', 'TASys02', 'TASys03', 'TASys11', 'Baseline', 'eval']
categories = ['dog_bark', 'footstep', 'gunshot', 'keyboard', 'moving_motor_vehicle', 'rain', 'sneeze_cough']

for embedding in embeddings:
    embds = None
    for system in systems:
        for category in categories:
            #get folder name
            if system == 'eval':
                eval_path = './DCASE_2023_Challenge_Task_7_Dataset/eval/' + category + '/'
            elif system == 'Baseline':
                eval_path = './DCASE_2023_Challenge_Task_7_Baseline/' + category + '/'
            else:
                track = system[1]
                eval_path = './DCASE_2023_Challenge_Task_7_Submission/AudioFiles/Submissions/' + track + '/' + system + '/' + category + '/'

            #concatenate
            if embds is None:
                embds, embds_shape = get_embeddings(eval_path, embedding)
            else:
                embds = np.concatenate((embds, get_embeddings(eval_path, embedding)[0]), axis=0)

    count = 0
    max_count = embds.shape[0]
    embds = torch.from_numpy(embds)
    embds = embds.float()
    (U, S, V) = torch.pca_lowrank(embds, q=128)

    for system in systems:
        for category in categories:
            #get folder name
            if system == 'eval':
                eval_path = './DCASE_2023_Challenge_Task_7_Dataset/eval/' + category + '/'
            elif system == 'Baseline':
                eval_path = './DCASE_2023_Challenge_Task_7_Baseline/' + category + '/'
            else:
                track = system[1]
                eval_path = './DCASE_2023_Challenge_Task_7_Submission/AudioFiles/Submissions/' + track + '/' + system + '/' + category + '/'

            emb_dir = Path(eval_path) / "embeddings" / embedding
            eval_files = list(emb_dir.glob("*.npy"))
            emb128_dir = Path(eval_path) / "embeddings" / (embedding + '-emb128')
            for file in eval_files:
                fname = file.name
                cur_embds = np.load(file)
                cur_embds = torch.from_numpy(cur_embds)
                cur_embds = cur_embds.float()
                new_embds = torch.matmul(cur_embds, V)

                if not os.path.exists(emb128_dir):
                    os.makedirs(emb128_dir)
                np.save(emb128_dir / fname, new_embds.numpy().astype('float16'))
                print(f'COUNT:{count}/{max_count}')
                count+=1
