import pandas as pd
import numpy as np

cols = ['Unnamed: 0', 'Unnamed: 0.1', 'subj_id', 'sex', 'age', 'prev_tb',
        'art_status', 'temp', 'oxy_sat', 'hgb', 'cd4', 'wbc', 'cough',
        'cough_dur', 'xr_lymph', 'xr_nod', 'xr_cav', 'xr_consol', 'xr_gg',
        'xr_micro', 'xr_effus', 'label', 'dataset', 'img_name', 'Path',
        'subjID', 'pulm_tb']

if __name__ == '__main__':
    df = pd.DataFrame()
    df['subj_id'] = np.arange(128)
    df['subjID'] = np.arange(128)
    img_paths = []
    img_names = []
    for i in range(128):
        img_paths.append('./dummy/dummy' + str(i) + '.jpeg')
        img_names.append('dummy' + str(i) + '.jpeg')
    df['img_name'] = img_names
    df['Path'] = img_paths
    for col in ['sex', 'prev_tb', 'art_status', 'oxy_sat', 'cough', 'cough_dur',
                'xr_lymph', 'xr_nod', 'xr_cav', 'xr_consol', 'xr_gg',
                'xr_micro', 'xr_effus', 'label']:
        df[col] = np.random.binomial(1, 0.5, 128)
    df['pulm_tb'] = df['label']
    for col in ['age', 'temp', 'hgb', 'cd4', 'wbc']:
        df[col] = np.random.random(128)
    df['dataset'] = 'niel'
    # print(df)
    train_df = df.iloc[:64]
    dev_df = df.iloc[64:96]
    test_df = df.iloc[96:]
    train_df.to_csv("train_data_0.csv")
    dev_df.to_csv("dev_data_0.csv")
    test_df.to_csv("test_data.csv")
