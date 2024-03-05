import numpy as np
from pathlib import Path
import time
import torch
import os
import sys
from doce.setting import Setting
import doce
from fadtk.fadtk.fad_calculation import calculate_fad
import pickle
torch.manual_seed(0)
np.random.seed(42)

if torch.cuda.is_available():
    # Set the random seed for GPU (if available)
    torch.cuda.manual_seed(0)

# define the experiment
experiment = doce.Experiment(
#   name = "exp_with_pann_resnet38",
  name = "fad_embeddings",
  purpose = 'Finding the best embedding for FAD calculation for DCASE',
  author = 'Modan Tailleur',
  address = 'modan.tailleur@ls2n.fr',
)

exp_path = './experiment/'

if not os.path.exists(exp_path):
    os.makedirs(exp_path)

experiment.set_path('fad', exp_path+experiment.name+'/fad/', force=True)
experiment.set_path('correlation', exp_path+experiment.name+'/correlation/', force=True)

experiment.add_plan('fad',
  category = ['dog_bark', 'footstep', 'gunshot', 'keyboard', 'moving_motor_vehicle', 'rain', 'sneeze_cough'],
  # embedding = ['vggish', 'clap-2023', 'clap-laion-audio', 'clap-laion-music', 'MERT-v1-95M-1', 
  #             'MERT-v1-95M-11', 'MERT-v1-95M', 'encodec-emb', 'encodec-emb-48k'],
  embedding = ['clap-2023', 'clap-2023-emb128', 'clap-laion-audio', 'clap-laion-music', 'vggish', 'MERT-v1-95M', 'encodec-emb', 'encodec-emb-48k', 
               'dac-44kHz', 'cdpam-acoustic', 'cdpam-content', 'panns-cnn14-32k', 'panns-cnn14-16k', 'panns-wavegram-logmel', 'panns-wavegram-logmel-1s', 'panns-wavegram-logmel-emb128',
               #panns-1s, 'hubert-base', 'hubert-large', 'whisper-tiny', 'whisper-small', 'whisper-base', 
               #'whisper-medium', 'whisper-large',
               'w2v2-base', 'w2v2-large', 'wavlm-base-plus', 'wavlm-base', 'wavlm-large'],
  # embedding = ['clap-2023', 'clap-laion-audio', 'clap-laion-music', 'vggish', 'MERT-v1-95M', 'encodec-emb', 'encodec-emb-48k', 
  #             'dac-44kHz', 'cdpam-acoustic', 'cdpam-content', 'panns-cnn14-32k', 'panns-wavegram-logmel'],
  # system = ['TASys02', 'TASys03', 'TASys04', 'TASys05', 'TASys06', 'TASys07', 'TASys08', 'TASys10', 'TASys11',\
  #           'TBSys01', 'TBSys02', 'TBSys03', 'TBSys04', 'TBSys05', 'TBSys07', 'TBSys08', 'TBSys09', 'TBSys11', \
  #             'TBSys14', 'TBSys15', 'TBSys16', 'TBSys17', 'TBSys18', 'TBSys19', 'TBSys20', 'TBSys21', 'TBSys22', \
  #               'TBSys23', 'TBSys24', 'TBSys25', 'TBSys26', 'TBSys27', 'TBSys28', 'TBSys29', 'TBSys30', 'TBSys31', 'Baseline'],
  system = ['TBSys09', 'TBSys18', 'TBSys14', 'TBSys24', 'TASys08', 'TASys02', 'TASys03', 'TASys11', 'Baseline'],
  reference = ['dev', 'eval'],
)

experiment.set_metric(
  name = 'fad',
  path = 'output',
  output = 'fad',
  func = np.mean,
  significance = True,
  percent=False,
  lower_the_better=True,
  precision=2
  )

def step(setting, experiment):

    print('XXXXXXXX ONGOING SETTING XXXXXXXX')
    print(setting.identifier())

    force_emb_calc = False
    force_cov_calc = False
    audio_len = 4
    system = setting.system
    category = setting.category

    if setting.reference == 'eval':
        eval_path = './DCASE_2023_Challenge_Task_7_Dataset/eval/' + category + '/'
        # eval_path = './DCASE_2023_Challenge_Task_7_EvalDataset/' + category + '/'
    elif setting.reference == 'dev':
        eval_path = './DCASE_2023_Challenge_Task_7_Dataset/dev/' + category + '/'
        # eval_path = './DCASE_2023_Challenge_Task_7_DevDataset/' + category + '/'

    if system == 'Baseline':
      audio_path = './DCASE_2023_Challenge_Task_7_Baseline/' + category + '/'
    else:
      track = system[1]
      audio_path = './DCASE_2023_Challenge_Task_7_Submission/AudioFiles/Submissions/' + track + '/' + system + '/' + category + '/'

    fad = calculate_fad(model_type=setting.embedding, baseline=eval_path, eval=audio_path, workers=1, force_emb_calc=force_emb_calc, force_cov_calc=force_cov_calc, audio_len=audio_len)
    
    print(f'FAD SCORE: {fad}')

    file_path = experiment.path.fad + setting.identifier() + '_fad'
    np.save(file_path, fad)
    # with open(file_path, 'wb') as file:
    #     pickle.dump(fad, file)
        
# invoke the command line management of the doce package
if __name__ == "__main__":
  doce.cli.main(experiment = experiment, func=step)
