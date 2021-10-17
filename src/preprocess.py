import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from utils import init_logger, seed_everything
from conf import *
import gc

# ====================================================
# FE
# ====================================================
def add_feature(df):
    # breath_time
    # df['breath_time'] = df['time_step'] - df['time_step'].shift(1)
    # df.loc[df['time_step'] == 0, 'breath_time'] = 0
    # # u_in_time
    # df['u_in_time'] = df['u_in'] - df['u_in'].shift(1)
    # df.loc[df['time_step'] == 0, 'u_in_time'] = 0
    # return df

    # From https://www.kaggle.com/tenffe/finetune-of-tensorflow-bidirectional-lstm
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    # df['R'] = df['R'].astype(str)
    # df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    # df = pd.get_dummies(df)

    # org fe
    df['breath_time'] = df['time_step'] - df['time_step'].shift(1)
    df.loc[df['time_step'] == 0, 'breath_time'] = 0
    # u_in_time
    df['u_in_time'] = df['u_in'] - df['u_in'].shift(1)
    df.loc[df['time_step'] == 0, 'u_in_time'] = 0
    return df


def preprocess_df(df):
    for c in ['u_in']:
        df[c] = np.log1p(df[c])
    r_map = {5: 0, 20: 1, 50: 2}
    c_map = {10: 0, 20: 1, 50: 2}
    df['R'] = df['R'].map(r_map)
    df['C'] = df['C'].map(c_map)
    return df


def main():
    seed_everything()
    # ====================================================
    # Data Loading
    # ====================================================
    train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
    train = preprocess_df(train)
    # for c in ['u_in']:
    #     train[c] = np.log1p(train[c])
    #     test[c] = np.log1p(test[c])
        
    # r_map = {5: 0, 20: 1, 50: 2}
    # c_map = {10: 0, 20: 1, 50: 2}
    # train['R'] = train['R'].map(r_map)
    # test['R'] = test['R'].map(r_map)
    # train['C'] = train['C'].map(c_map)
    # test['C'] = test['C'].map(c_map)

    train = add_feature(train)

    # ====================================================
    # CV split
    # ====================================================
    print('Applying Group Fold')
    Fold = GroupKFold(n_splits=5)
    groups = train['breath_id'].values
    for n, (train_index, val_index) in enumerate(Fold.split(train, train['pressure'], groups)):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    print(train.groupby('fold').size())

    out_path = f'../input/features/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    train_fold_path = os.path.join(out_path, f'train_fold_v{cfg.feature_version}.csv')
    test_fe_path = os.path.join(out_path, f'test_fe_v{cfg.feature_version}.csv')
    train.to_csv(train_fold_path, index=False)
    del train
    gc.collect()
    print(f'Added features saving: {train_fold_path}')

    # ====================================================
    # test
    # ====================================================
    test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
    test = preprocess_df(test)
    test = add_feature(test)
    test.to_csv(test_fe_path, index=False)
    del test
    print(f'Added features saving: {test_fe_path}')


if __name__ == '__main__':
    main()