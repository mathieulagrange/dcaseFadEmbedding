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
from itertools import combinations
import pandas as pd
from openpyxl import Workbook

if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(42)

    # Create the directory if it doesn't exist
    output_directory = './dcase_isomap_data'
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    embedding = ['clap-2023', 'panns-wavegram-logmel', 'vggish']
    category = ['dog_bark', 'footstep', 'gunshot', 'keyboard', 'moving_motor_vehicle', 'rain', 'sneeze_cough']
    force_emb_calc = False
    audio_len = 4

    category_couples = list(combinations(category, 2))

    # Create a Pandas DataFrame to store FAD values
    fad_data = pd.DataFrame(columns=category, index=category)

    for emb in embedding:
        
        for category_couple in category_couples:
            baseline = './DCASE_2023_Challenge_Task_7_Dataset/merged/' + category_couple[0] + '/'
            eval_path = './DCASE_2023_Challenge_Task_7_Dataset/merged/' + category_couple[1] + '/'

            fad = calculate_fad(model_type=emb, baseline=baseline, eval=eval_path, workers=1, force_emb_calc=force_emb_calc, audio_len=audio_len)
            
            print(f'FAD:{fad}')
            # Store FAD value in DataFrame
            fad_data.at[category_couple[0], category_couple[1]] = fad
            fad_data.at[category_couple[1], category_couple[0]] = fad  # FAD is symmetric

        fad_data.to_excel(f'{output_directory}/dcase_{emb}_full.xlsx', sheet_name=emb)

        print(f'Excel file {emb}_full.xlsx created successfully.')