#!/usr/bin/env python
# encoding: utf-8

import csv
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.utils import shuffle

def func(x):
    x = float(x)
    if x < 0:
        x = 0.0
    return x

def generate(times=1.5):
    # nsrmc, sb_seq, fp_seq, type, hy_dm, hymc
    data = []
    with open('../tax-data/nsrmc_dual_sequence_samples','r') as csvfile:
        reader = csv.reader(csvfile, delimiter='#')
        for row in reader:
            data.append(row)

    df = pd.DataFrame(data, columns=['nsrmc', 'sb_time_seq', 'fp_time_seq', 'type', 'hy_dm', 'hymc'])
    df['hy_dm_new'] = df['hy_dm'].map(lambda x : x[0:2])
    df = df[df['hy_dm_new'].isin(['51', '52'])]
    df_xk = df[df['type'] == 'xk']
    df_zc = df[df['type'] == 'zc'].sample(n=int(df_xk.shape[0] * times))
    df = pd.concat([df_xk,df_zc], axis=0)

    xk = df_xk['type'].reset_index()
    xk.rename(columns={'index':'id'}, inplace=True)
    zc = df_zc['type'].reset_index()
    zc.rename(columns={'index':'id'}, inplace=True)
    xk_zc = pd.concat([xk,zc], axis=0)
    #print(xk_zc)

    df_sb_time_seq = df['sb_time_seq'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).to_frame('sb_time_seq')
    df_sb_time_seq = df_sb_time_seq['sb_time_seq'].str.split(':', expand=True)
    df_sb_time_seq.columns = ["sb_time", "sb_val"]
    df_sb_time_seq['sb_val'] = df_sb_time_seq['sb_val'].map(lambda x : func(x))
    sb_qcut = pd.qcut(df_sb_time_seq['sb_val'], 2000, labels=False, duplicates='drop').to_frame('sb_val_qcut')
    sb_qcut = sb_qcut.reset_index()
    sb_qcut.rename(columns={'index':'id'}, inplace=True)
    sb_qcut = sb_qcut.groupby(by='id')['sb_val_qcut'].apply(list).to_frame('sb_val_qcut')
    #print(sb_qcut)

    df_fp_time_seq = df['fp_time_seq'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).to_frame('fp_time_seq')
    df_fp_time_seq = df_fp_time_seq['fp_time_seq'].str.split(':', expand=True)
    df_fp_time_seq.columns = ["fp_time", "fp_val"]
    df_fp_time_seq['fp_val'] = df_fp_time_seq['fp_val'].map(lambda x : func(x))
    fp_qcut = pd.qcut(df_fp_time_seq['fp_val'], 2000, labels=False, duplicates='drop').to_frame('fp_val_qcut')
    fp_qcut = fp_qcut.reset_index()
    fp_qcut.rename(columns={'index':'id'}, inplace=True)
    fp_qcut = fp_qcut.groupby(by='id')['fp_val_qcut'].apply(list).to_frame('fp_val_qcut')
    #print(fp_qcut)

    sb_data = pd.merge(xk_zc, sb_qcut, on='id')
    sb_fp_data = pd.merge(sb_data, fp_qcut, on='id')
    for _ in range(10):
        sb_fp_data = shuffle(sb_fp_data)

    # time
    filepath = '../tax-data/records.csv'
    sb_fp_data.to_csv(filepath, index=False)

    return filepath
