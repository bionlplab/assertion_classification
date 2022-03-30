from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
import json
from utils import common_args
import os

def load_formated_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Format preprocessed source data into dataframe
    '''
        
    dfs = []
    for idx, row in df.iterrows():
        dfs.append({
            'id': row['sentence_id'],
            'text': row['sentence'],
            'concept': row['concept'],
            'label': row['label'],
            'concept_start': row['concept_start'],
            'concept_end': row['concept_end']
        })

    result_df = pd.DataFrame(dfs)
    return result_df

def load_formated_chia_dataframe(df):
    
    dfs = []
    for idx, row in df.iterrows():
        if row['concept_start'] is not None and row['concept_end'] is not None:
            dfs.append({
                'id': row['sentence_id'],
                'text': row['sentence'],
                'concept': row['concept'],
                'label': row['label'],
                'concept_start': int(row['concept_start']),
                'concept_end': int(row['concept_end'])
            })
        else:
            start_end_list = row['concept_start_end_list']
            start = start_end_list.split('(')[1]
            start = int(start.split(',')[0])
            end = start_end_list.split(',')[-1]
            end = int(end.split(')')[0])
            dfs.append({
                'id': row['sentence_id'],
                'text': row['sentence'],
                'concept': row['concept'],
                'label': row['label'],
                'concept_start': start,
                'concept_end': end
            })

    result_df = pd.DataFrame(dfs)
    return result_df

def load_train_val_dataframe(df, fold: int, fold_path, bioinfer=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Split data into train and val set given the fold index.
    Datasplit indices is preprocessed.
    bioinfer: the naming of id is a bit different.

    fold_path: the path to a folder that stores all folds' indices.
    '''
    train_indices = []
    val_indices = []
    for dirname, _, filenames in os.walk(fold_path):
        filenames.sort()
        for filename in filenames:
            if str(fold) not in filename:
                with open(os.path.join(dirname, filename), 'r') as file:
                    data = file.readlines()
                    train_indices.extend([d.strip() for d in data])
            else:
                with open(os.path.join(dirname, filename), 'r') as file:
                    data = file.readlines()
                    val_indices = [d.strip() for d in data]

    if bioinfer:
        df['id'] = df['id'].apply(lambda x: ".".join(x.split(".")[:-1]))   

    train_df = df[df['id'].isin(train_indices)]
    val_df = df[df['id'].isin(val_indices)]

    assert len(train_df) + len(val_df) == len(df)
    return train_df, val_df


if __name__ == '__main__':
    parser= common_args()
    args = parser.parse_args()
