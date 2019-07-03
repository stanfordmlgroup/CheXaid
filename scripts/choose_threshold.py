from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

prob_files_paths = [
    '/deep/u/pranavsr/aihc-fall18-2dimaging/probabilities_valid1.csv',
    '/deep/u/pranavsr/aihc-fall18-2dimaging/probabilities_valid2.csv',
    '/deep/u/pranavsr/aihc-fall18-2dimaging/probabilities_valid3.csv',
    '/deep/u/pranavsr/aihc-fall18-2dimaging/probabilities_valid4.csv',
    '/deep/u/pranavsr/aihc-fall18-2dimaging/probabilities_valid5.csv'
]

for model_output in prob_files_paths:
    df = pd.read_csv(model_output)
    tb_prob = df['pulm_tb_prob']
    gt = df['pulm_tb_target']
    scores = []
    thresholds = np.linspace(0.5, 0.5, 40)
    for threshold in thresholds:
        tb_decision = tb_prob > threshold
        score = accuracy_score(tb_decision, gt)
        scores.append(score)
    scores = np.array(scores)
    max_score = np.max(scores)
    max_index = np.argmax(scores)
    optimal_threshold = thresholds[max_index]
    print(optimal_threshold, max_score)
