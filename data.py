import os
import pandas as pd


DATA_DIR=os.path.join('..','sample')
def load_data():
    ppt_df = pd.read_csv(os.path.join(DATA_DIR, 'participants.tsv'), delimiter='\t')
    feat_df = pd.read_csv(os.path.join(DATA_DIR, 'features.tsv'), delimiter='\t')
    comb_df = ppt_df.merge(feat_df, on='pid')
    return comb_df

def prepare_data():
    df = load_data()
    indep_features = ['baseline_sbp', 'age', 'weight', 'height', 'delta_hr_ekg', 'delta_rpat_pressure']
    target_feature = 'delta_sbp'

    ambulatory_df = df[df['phase'] == 'ambulatory']
    clean_df = ambulatory_df.dropna(subset=indep_features + [target_feature])

    X = clean_df[indep_features].values.astype('float32')
    y = clean_df[target_feature].values.astype('float32').reshape(-1, 1)
    pids = clean_df['pid'].values

    return X, y, pids