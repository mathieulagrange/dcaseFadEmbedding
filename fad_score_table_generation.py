import main_doce
import pandas as pd
import pandas as pd

reference = 'eval'

excel_file = './excel_files/fadScores.xlsx'

embedding_list = main_doce.experiment.fad.embedding

with pd.ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:

  for embedding in embedding_list:
    (data_pred, settings_pred_total, header_pred_total) = main_doce.experiment.get_output(
      output = 'fad',
      selector = {"reference":reference, "embedding":embedding},
      path = "fad",
      )
    if len(data_pred) == 0:
      continue

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
    df.to_excel(writer, index=True, sheet_name=dict_system_info['embedding'])
