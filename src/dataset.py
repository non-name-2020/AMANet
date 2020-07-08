#!/usr/bin/env python
# encoding: utf-8

import dill
import ast
import numpy as np
import pandas as pd


def load_dictionary(filepath="../medical-data/voc_final.pkl"):
    # load dictionary
    voc = dill.load(open(filepath, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    str2tok_diag = diag_voc.idx2word
    str2tok_proc = pro_voc.idx2word
    str2tok_med = med_voc.idx2word
    # dict(zip(m.values(), m.keys()))
    input_size1 = len(str2tok_diag)
    input_size2 = len(str2tok_proc)
    output_size = len(str2tok_med)
    print('size of input1: {}, input2: {}'.format(len(str2tok_diag), len(str2tok_proc)))
    print('size of output: {}'.format(len(str2tok_med)))
    return input_size1, input_size2, output_size


def split_train_valid_set(filepath="../medical-data/records_final.pkl", save_dir="../medical-data/"):
    patient_records = dill.load(open(filepath, "rb"))
    # split dataset
    all_index = list(range(len(patient_records)))
    # 2/3 as train set
    train_index = all_index[:int(len(patient_records) * 2 / 3)]
    # 1/6 as valid set
    valid_index = all_index[int(len(patient_records) * 2 / 3):int(len(patient_records) * 5 / 6)]
    # 1/6 as test set
    test_index = all_index[int(len(patient_records) * 5 / 6):int(len(patient_records) * 1)]
    patient_list_train = [patient_records[i] for i in train_index]
    patient_list_valid = [patient_records[i] for i in valid_index]
    patient_list_test = [patient_records[i] for i in test_index]

    print('num_patient: {}'.format(len(patient_records)))
    print('num train: {}'.format(len(patient_list_train)))
    print('num valid: {}'.format(len(patient_list_valid)))
    print('num test: {}'.format(len(patient_list_test)))

    with open(save_dir+'/patient_train.txt',"w+") as wp:
        for i in range(len(patient_list_train)):
            wp.write(str(patient_list_train[i])+"\n")

    with open(save_dir+'/patient_valid.txt',"w+") as wp:
        for i in range(len(patient_list_valid)):
            wp.write(str(patient_list_valid[i])+"\n")

    with open(save_dir+'/patient_test.txt',"w+") as wp:
        for i in range(len(patient_list_test)):
            wp.write(str(patient_list_test[i])+"\n")

    return patient_list_train, patient_list_valid, patient_list_test


def load_tax_data(data_path='../tax-data/records.csv'):
    df = pd.read_csv(data_path)

    split_point = int(df.shape[0] * 2 / 3)
    data_train = df[:split_point]
    eval_len = int((df.shape[0] - data_train.shape[0]) / 2)
    data_test = df[split_point:split_point + eval_len]
    data_eval = df[split_point + eval_len:]
    return data_train, data_test, data_eval, [2000, 2000, 1]


def load_data(data_path= '../medical-data/records_final.pkl',voc_path='../medical-data/voc_final.pkl'):
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    return data_train, data_test, data_eval, voc, voc_size

def statistic_tax(data_path= '../tax-data/records.csv'):
    import re
    df = pd.read_csv(data_path)
    df['sb_val_qcut_len'] = df['sb_val_qcut'].apply(lambda x : len([int(v) for v in re.sub(r'[\[\]\s]+', "", x).split(',')]))
    df['fp_val_qcut_len'] = df['fp_val_qcut'].apply(lambda x : len([int(v) for v in re.sub(r'[\[\]\s]+', "", x).split(',')]))
    print('fp min len: %d, max len: %d, mean len: %f' %(df['fp_val_qcut_len'].min(),df['fp_val_qcut_len'].max(),df['fp_val_qcut_len'].mean()))
    print('sb min len: %d, max len: %d, mean len: %f' %(df['sb_val_qcut_len'].min(),df['sb_val_qcut_len'].max(),df['sb_val_qcut_len'].mean()))


def statistic(data_path= '../medical-data/records_final.pkl',voc_path='../medical-data/voc_final.pkl'):
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    stats = [[0,0,0,0,0],[0,0,0,0],[0,0,0,0],[100000000,100000000,100000000,100000000]]
    stats[0][0] = len(data)
    stats[0][1] = sum([len(p) for p in data])
    stats[0][2] = voc_size[0]
    stats[0][3] = voc_size[1]
    stats[0][4] = voc_size[2]
    stats[1][0] = stats[0][1]/stats[0][0]
    diagnosis = []
    procedure = []
    medication = []
    for p in data:
        len_p = len(p)
        if stats[2][0] < len_p:
            stats[2][0] = len_p
        if stats[3][0] > len_p:
            stats[3][0] = len_p
        for v in p:
            len_v0 = len(v[0])
            if stats[2][1] < len_v0:
                stats[2][1] = len_v0
            if stats[3][1] > len_v0:
                stats[3][1] = len_v0
            len_v1 = len(v[1])
            if stats[2][2] < len_v1:
                stats[2][2] = len_v1
            if stats[3][2] > len_v1:
                stats[3][2] = len_v1
            len_v2 = len(v[2])
            if stats[2][3] < len_v2:
                stats[2][3] = len_v2
            if stats[3][3] > len_v2:
                stats[3][3] = len_v2
            diagnosis.append(len_v0)
            procedure.append(len_v1)
            medication.append(len_v2)
    stats[1][1] = sum(diagnosis)/len(diagnosis)
    stats[1][2] = sum(procedure)/len(procedure)
    stats[1][3] = sum(medication)/len(medication)
    return stats

def prepare_mimic_sample_dual(icd_list, med_list, output_size, index=-1):
    # 随机选择一个样本
    if index < 0:
        index = int(np.random.choice(len(icd_list), 1))
    input_seq = icd_list[index]
    i1 = []
    i2 = []
    isi1 = True
    for c in input_seq:
        if c == 0:
            isi1 = False
        else:
            if isi1:
                i1.append(c)
            else:
                i2.append(c)

    o = med_list[index]
    output = [med - 2 for med in o]

    if i2 is []:
        i2 = [0]

    input_vec1 = [i1]
    input_vec2 = [i2]

    output_vec = [[0] * output_size]

    for med in o:
        output_vec[0][med-2] = 1
    return input_vec1, input_vec2, output_vec, output


def prepare_mimic_sample_dual_persist(patient_list, output_size, index=-1):
    # select a patient
    if index < 0:
        index = int(np.random.choice(len(patient_list), 1))

    patient = patient_list[index]
    adms = []
    # select a visit of the patient
    for adm in patient:
        if len(adm) > 2:
            # input seq1
            input_seq1 = adm[0]
            # inout seq2
            input_seq2 = adm[1]
            # output seq
            output_seq = adm[2]
        else:
            # input seq1
            input_seq1 = adm[0]
            # input seq1 reverse as input seq2
            input_seq2 = adm[0][::-1]
            # output seq
            output_seq = adm[1]
        adms.append(prepare_mimic_sample_dual([input_seq1 + [0] + input_seq2],
                                              [output_seq],
                                              output_size, 0))
    return adms


def prepare_tax_dual(tax_df, index=-1):
    # select a company
    if index < 0:
        index = int(np.random.choice(tax_df.shape[0], 1))
    qy = tax_df[index:index+1]

    # input seq1
    input_seq1 = [eval(qy["sb_val_qcut"].values[0])]
    # inout seq2
    input_seq2 = [eval(qy["fp_val_qcut"].values[0])]
    # output seq
    output = qy["type"].values[0]

    if output == 'xk':
        outpur_vec = [[0]]
        o = [0]
    else:
        outpur_vec = [[1]]
        o = [1]

    return input_seq1, input_seq2, outpur_vec, o


def load_dataset(filepath):
    patients = []
    with open(filepath, "r") as rp:
        for line in rp.readlines():
            patient = ast.literal_eval(line)
            patients.append(patient)

    return patients

def load_drgs_data(data_path='../drgs-data/records.csv'):
    df = pd.read_csv(data_path, sep='\t')
    split_point = int(df.shape[0] * 3 / 5)
    data_train = df[:split_point]
    eval_len = int((df.shape[0] - data_train.shape[0]) / 2)
    data_test = df[split_point:split_point + eval_len]
    data_eval = df[split_point + eval_len:]

    min_diagnosis_len = 100000
    max_diagnosis_len = 0
    mean_diagnosis_len = 0
    for i in range(df.shape[0]):
        length = len(eval(df.iloc[[i]]["diagnosis"].values[0]))
        if max_diagnosis_len < length:
            max_diagnosis_len = length
        if min_diagnosis_len > length:
            min_diagnosis_len = length
        mean_diagnosis_len += length
    print("max: %d, min:%d, mean: %f" %(max_diagnosis_len, min_diagnosis_len, mean_diagnosis_len/df.shape[0]))

    min_operation_len = 100000
    max_operation_len = 0
    mean_operation_len = 0
    for i in range(df.shape[0]):
        length = len(eval(df.iloc[[i]]["operation"].values[0]))
        if max_operation_len < length:
            max_operation_len = length
        if min_operation_len > length:
            min_operation_len = length
        mean_operation_len += length
    print("max: %d, min:%d, mean: %f" %(max_operation_len, min_operation_len, mean_operation_len/df.shape[0]))

    import os
    dirpath = os.path.dirname(data_path)
    diagnosis_size = pd.read_csv(os.path.join(dirpath, "diagnosis_id_code_dict.csv")).shape[0]
    operation_size = pd.read_csv(os.path.join(dirpath, "operation_id_code_dict.csv")).shape[0]
    drgs_size = pd.read_csv(os.path.join(dirpath, "drgs_id_code_dict.csv")).shape[0]
    return data_train, data_test, data_eval, [diagnosis_size, operation_size, drgs_size]


def prepare_drgs_dual(drgs_df, drgs_size, index=-1):
    # select a sample
    if index < 0:
        index = int(np.random.choice(drgs_df.shape[0], 1))
    sample = drgs_df[index:index+1]

    # input seq1
    input_seq1 = [eval(sample["diagnosis"].values[0])]
    # inout seq2
    input_seq2 = [eval(sample["operation"].values[0])]
    # output seq
    output = sample["drgs"].values[0]

    output_vec = [[0]*drgs_size]
    output_vec[0][output] = 1
    output_vec = np.array(output_vec).reshape((1,drgs_size))
    o = [0] * drgs_size
    o[output] = 1
    #print(o)
    return input_seq1, input_seq2, output_vec, o



