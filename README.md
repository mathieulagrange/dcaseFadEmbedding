# dcaseFadEmbedding

## Setup

Make sure you have downloaded the two datasets which should be named "DCASE_2023_Challenge_Task_7_Submission", "DCASE_2023_Challenge_Task_7_Dataset" and "DCASE_2023_Challenge_Task_7_Baseline" which should be placed in the main folder. 

Install 'requirements.txt' before running the code.
Launch 'get_doce.py' to get the latest version of doce (see doce documentation here: [DOCE](https://doce.readthedocs.io/en/latest/))
```bash
git clone https://github.com/mathieulagrange/dcaseFadEmbedding.git
cd dcaseFadEmbedding
python3 get_doce.py
pip install -r requirements.txt
```

Launch 'create_merged_dcase_dataset.py' to create a merged dataset from the evaluation set and the developpment set.
```bash
python3 create_merged_dcase_dataset.py
```

Launch 'download_audioset.py' to download audioset in .ogg format, and launch 'convert_audioset.py' to convert this dataset into a 10-s segments audio dataset sampled at 32kHz.

## Correlations between perceptual scores and FAD on Dcase Task 7 2023 dataset

### FAD calculation

To launch the calculation of the embeddings and of the FAD, use:

```
python3 main_doce.py -s reference=eval -c
```

This will create subfolders for every category folder in "DCASE_2023_Challenge_Task_7_Submission" and "DCASE_2023_Challenge_Task_7_Dataset" which will contain the embeddings. The FAD calculation results are stored in experiment/fad_embeddings/fad as npy files.

Use this bash command to remove every folder linked the the calculation of a specific embedding my-emb:

```
find . -type d -name "my-emb" -exec rm -r {} +
```

### Correlation calculation

Launch the following code for modifying "fadScores.xlsx" with your new results

```
python3 fad_scores_table_generation.py
```

Launch the following code to create a new "correlation.xlsx" file:

```
python3 correlation_calculation.py
```

Launch the following code to perform a t-test between the mean correlations of 'wavegram-logmel' and the other embeddings:

```
python3 ttest_on_correlation_calculation.py
```

Launch the following code to reproduce the paper figures:

```
python3 plot_correlation_scores.py
python3 plot_all_categories_correlation_scores.py

```

The figure are stored in the "figures" folder.

## Inter-category FAD calculation and clustering

Launch the following code to generate the similarity matrix for each Dcase Task 7 2023 category into an xlsx format:
```
python3 intercategory_dcase_fad.py
```

Only 3 embeddings are evaluated here: clap-2023 (MS-CLAP), vggish and panns-wavegram-logmel. The matrices are automatically stored in the 'dcase_isomap_data' folder.


Then launch the following code to plot the isomap found in the paper:

```
python3 plot_dcase_isomap.py
```

The figure is stored in the "figures" folder.