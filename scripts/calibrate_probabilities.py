import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
import IPython

def load_data(df, probs_cols, target_col, is_test=False):
    probs = df[probs_cols]
    probs = logit(probs)
    if is_test is True:
        return probs
    targets = df[target_col]
    return probs, targets


def train_logreg(probs, targets):
    """Train a logreg."""
    clf = LogisticRegression(C=1e8, penalty='l2')
    clf.fit(probs, targets)
    return clf


def predict_logreg(probs, clf):
    calibrated_probs = clf.predict_proba(probs)[:,1]
    return calibrated_probs

def calibrate_individual(valid_path, test_path):
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    probs_cols = ['pulm_tb_prob', 'xr_nod_prob', 'xr_gg_prob',
            'xr_cav_prob', 'xr_effus_prob', 'xr_consol_prob',
            'xr_micro_prob']
    target_cols = ['pulm_tb_target', 'xr_nod_target', 'xr_gg_target',
            'xr_cav_target', 'xr_effus_target', 'xr_consol_target',
            'xr_micro_target']

    calibrated_probs = {}
    for target_col in target_cols:
        train_probs, targets = load_data(valid_df,
                # probs_cols,
                [target_col.replace('target', 'prob')],
                target_col)
        clf = train_logreg(train_probs, targets)
        test_probs = load_data(test_df,
                # probs_cols,
                [target_col.replace('target', 'prob')],
                target_col, is_test=True)
        calibrated_probs[target_col] = predict_logreg(test_probs, clf)

    for target_col, probs in calibrated_probs.items():
        test_df[target_col.replace('target', 'prob')] = probs

    test_df.to_csv(test_path.replace('test', 'test_calibrated'))


def calibrate_ensemble(folder_name):
    for i in range(1, 6):
        valid_path = os.path.join(folder_name, 'probabilities_valid'
                + str(i) + '.csv')
        test_path = os.path.join(folder_name, 'probabilities_test'
                + str(i) + '.csv')
        calibrate_individual(valid_path, test_path)

import sys
calibrate_ensemble(sys.argv[1])


