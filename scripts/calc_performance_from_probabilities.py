import os
import sys
import pandas as pd
import numpy as np
from sklearn import metrics


# Thresholds for determining unlikely/possible/likely prediction
LOWER_THRESHOLD = 0.4
UPPER_THRESHOLD = 0.6

def get_prob_files_from_ensemble_path(ensemble_path, extension):
    prob_files_paths = []
    for i in range(1, 6):
        path = os.path.join(ensemble_path,
                extension + str(i) + '.csv')
        prob_files_paths.append(path)

    return prob_files_paths

prob_suffix = '_prob'
target_suffix = '_target'

def calc_metrics(df, task_name, threshold=0.5):
    probabilities, targets = [], []
    for index, row in df.iterrows():
        subjID = row['subjectID']
        task_prob = row[task_name + prob_suffix]
        task_target = row[task_name + target_suffix]
        probabilities.append(task_prob)
        targets.append(task_target)

    predictions = [elem > threshold for elem in probabilities]
    auc = metrics.roc_auc_score(targets, probabilities)
    accuracy = metrics.accuracy_score(targets, predictions)
    return auc, accuracy

def get_tasks(df):
    tasks = []
    for column_name in df:
        if column_name != 'subjectID':
            if column_name.endswith(prob_suffix):
                task = column_name[:-len(prob_suffix)]
                tasks.append(task)
            elif column_name.endswith(target_suffix):
                task = column_name[:-len(target_suffix)]
                tasks.append(task)
    return set(tasks)

def add_predictions(df, tasks):
    for task in tasks:
        pred_col = task + '_pred'
        prob_col = task + '_prob'
        df[pred_col] = df[prob_col]
        df.loc[df[prob_col] > UPPER_THRESHOLD, pred_col] = 'Likely'
        df.loc[df[prob_col] < LOWER_THRESHOLD, pred_col] = 'Unlikely'
        df.loc[(df[prob_col] > LOWER_THRESHOLD) & (df[prob_col] < UPPER_THRESHOLD), pred_col] = 'Possible'
    return df

def merge_dfs(dfs, key='subjectID'):
    concat_df = pd.concat(dfs)
    avg_df = concat_df.groupby(key).mean().reset_index()
    return avg_df

if __name__ == '__main__':
    ensemble_path = sys.argv[1]
    extension = sys.argv[2] # 'probabilities_test_calibrated'
    prob_files_paths = get_prob_files_from_ensemble_path(
            ensemble_path, extension)
    try:
        dfs = [pd.read_csv(prob_file_path) for prob_file_path in
               prob_files_paths]
    except:
        path = os.path.join(ensemble_path,
                    extension + '.csv')
        dfs = [pd.read_csv(path)]

    df = merge_dfs(dfs)
    tasks = get_tasks(df)
    df = add_predictions(df, tasks)

    df.to_csv(os.path.join(ensemble_path, extension + '_ensemble.csv'))

    for task in tasks:
        auc, accuracy = calc_metrics(df, task)
        print(f'Task: {task}, AUC: {auc}, Accuracy: {accuracy}')

    df['dataset'] = np.where(df['subjectID'].str.startswith('TBPOC'), 'niel', 'tom')
    df_p = df[df['dataset'] == 'niel']
    df_h = df[df['dataset'] == 'tom']
    for task in tasks:
        auc, accuracy = calc_metrics(df_p, task)
        print(f'Niel: Task: {task}, AUC: {auc}, Accuracy: {accuracy}')
    for task in tasks:
        auc, accuracy = calc_metrics(df_h, task)
        print(f'Tom: Task: {task}, AUC: {auc}, Accuracy: {accuracy}')
