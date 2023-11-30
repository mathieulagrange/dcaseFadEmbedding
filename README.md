# dcaseFadEmbedding

## Setup

Make sure you have downloaded the two datasets which should be named "DCASE_2023_Challenge_Task_7_Submission" and "DCASE_2023_Challenge_Task_7_Dataset" and which should be placed in the main folder. 

Install 'requirements.txt' before running the code.

## FAD calculation

To launch the calculation of the embeddings and of the FAD, use:

```
python3 main_doce.py -s reference=eval -c
```

This will create subfolders for every category folder in "DCASE_2023_Challenge_Task_7_Submission" and "DCASE_2023_Challenge_Task_7_Dataset" which will contain the embeddings. The FAD calculation results are stored in experiment/fad_embeddings/fad as npy files.

## Correlation calculation

Launch the following code for modifying "fadScores.xlsx" with your new results

```
python3 fad_scores_table_generation.py
```

Launch the following code to create a new "correlation.xlsx" file:

```
python3 correlation_calculation.py
```