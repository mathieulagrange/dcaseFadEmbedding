import main_doce
import pandas as pd
from textwrap import wrap
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import argparse

def main(config):
  reference = 'eval'
  embedding_list = config.emb

  excel_file = './excel_files/fadScores.xlsx'

  for embedding in embedding_list:
    (data_pred, settings_pred_total, header_pred_total) = main_doce.experiment.get_output(
      output = 'fad',
      selector = {"reference":reference, "embedding":embedding},
      path = "fad",
      )
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
      score_list = []
      for idx, (pred, settings_pred) in enumerate(zip(data_pred, settings_pred_total)):
          pairs = settings_pred.split(', ')
          dict_system_info = dict(pair.split('=') for pair in pairs)
          dict_system_info ['fad'] = pred.item()
          dict_system_info ['reference'] = reference
          dict_system_info ['embedding'] = embedding
          score_list.append(dict_system_info)

      df = pd.DataFrame(score_list)
      df = df.rename(columns={'system': 'alg_code'})

      df = df.pivot(index='alg_code', columns='category', values='fad')

      df['avg_category_FAD'] = df.mean(axis=1)
      df.to_excel(writer, index=True, sheet_name=embedding)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modifies fadScores.xlsx to add the FAD for the new specified embeddings')
    parser.add_argument('--emb', type=list, default=['vggish', 'clap-2023', 'clap-laion-audio', 'MERT-v1-95M-1', 'encodec-emb', 'encodec-emb-48k'],
                        help="The list of embeddings to calculate - default: ['vggish', 'clap-2023', 'clap-laion-audio', 'MERT-v1-95M-1', 'encodec-emb', 'encodec-emb-48k']")
    config = parser.parse_args()
    main(config)