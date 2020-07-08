#!/usr/bin/env python
# encoding: utf-8

import csv
from random import shuffle
from os import listdir
from os.path import isfile, join

def load_data(dirname="../drgs-data"):
    files = [join(dirname, f) for f in listdir(dirname) if isfile(join(dirname, f))]
    diagnosis_code_name_dict = {}
    diagnosis_id_code_dict = {}
    diagnosis_code_id_dict = {}

    operation_code_name_dict = {}
    operation_id_code_dict = {}
    operation_code_id_dict = {}

    drgs_code_name_dict = {}
    drgs_id_code_dict = {}
    drgs_code_id_dict = {}
    drgs_samples = {}
    for filename in files:
        if 'jiande' not in filename:
            continue
        print(filename)
        csvFile = open(filename, "r")
        reader = csv.reader(csvFile)
        for item in reader:
            if reader.line_num == 1:
                continue
            drgs_code = item[9]
            drgs_name = item[10]
            drgs_code_name_dict[drgs_code] = drgs_name
            diagnosis = []
            if drgs_code not in drgs_code_id_dict:
                drgs_id = len(drgs_code_id_dict) + 1
                drgs_code_id_dict[drgs_code] = drgs_id
                drgs_id_code_dict[drgs_id] = drgs_code

            for i in range(0, 22, 2):
                diagnosis_code = item[13+i]
                diagnosis_name = item[14+i]
                if diagnosis_code is None or diagnosis_name is None:
                    break
                diagnosis_name = diagnosis_name.strip()
                diagnosis_code = diagnosis_code.strip()
                if diagnosis_code == '' or diagnosis_name == '':
                    break
                if diagnosis_code not in diagnosis_code_id_dict:
                    diagnosis_id = len(diagnosis_code_id_dict)
                    diagnosis_code_id_dict[diagnosis_code] = diagnosis_id
                    diagnosis_id_code_dict[diagnosis_id] = diagnosis_code

                diagnosis.append(diagnosis_code_id_dict[diagnosis_code])
                diagnosis_code_name_dict[diagnosis_code] = diagnosis_name

            operation = []
            for i in range(0, 10, 2):
                operation_code = item[35 + i]
                operation_name = item[36 + i]
                if operation_code is None and operation_name is None:
                    break
                operation_code = operation_code.strip()
                operation_name = operation_name.strip()
                if operation_code == '' or operation_name == '':
                    break
                if operation_code not in operation_code_id_dict:
                    operation_id = len(operation_code_id_dict) + 1
                    operation_code_id_dict[operation_code] = operation_id
                    operation_id_code_dict[operation_id] = operation_code

                operation.append(operation_code_id_dict[operation_code])
                operation_code_name_dict[operation_code] = operation_name
            if len(operation) == 0:
                operation = [0]
            drgs_samples[item[0]] = [diagnosis, operation, drgs_code_id_dict[drgs_code]]
        csvFile.close()
    drgs_sample_stats = {}
    for item in drgs_samples.items():
        drgs_code = item[1][2]
        if drgs_code not in drgs_sample_stats:
            drgs_sample_stats[drgs_code] = 0
        drgs_sample_stats[drgs_code] += 1
    print('--------samples stats--------')
    print(drgs_sample_stats)

    # 过滤掉类别次数
    samples = [item[1] for item in drgs_samples.items() if drgs_sample_stats[item[1][2]] >= 10]

    return drgs_code_id_dict, drgs_id_code_dict, drgs_code_name_dict, diagnosis_code_id_dict, diagnosis_id_code_dict, diagnosis_code_name_dict, operation_code_id_dict, operation_id_code_dict, operation_code_name_dict, samples

def re_load_data(drgs_code_id_dict, drgs_id_code_dict, drgs_code_name_dict, diagnosis_code_id_dict, diagnosis_id_code_dict, diagnosis_code_name_dict, operation_code_id_dict, operation_id_code_dict, operation_code_name_dict, samples):
    new_drgs_code_id_dict = {}
    new_drgs_id_code_dict = {}
    new_drgs_code_name_dict = {}
    new_diagnosis_code_id_dict = {}
    new_diagnosis_id_code_dict = {}
    new_diagnosis_code_name_dict = {}
    new_operation_code_id_dict = {}
    new_operation_id_code_dict = {}
    new_operation_code_name_dict = {}
    new_samples = []
    for sample in samples:
        diagnosis_lst = sample[0]
        new_diagnosis_lst = []
        for diagnosis_id in diagnosis_lst:
            diagnosis_code = diagnosis_id_code_dict[diagnosis_id]
            diagnosis_name = diagnosis_code_name_dict[diagnosis_code]
            if diagnosis_code not in new_diagnosis_code_name_dict:
                new_diagnosis_code_name_dict[diagnosis_code] = diagnosis_name
                new_diagnosis_id = len(new_diagnosis_id_code_dict) + 1
                new_diagnosis_id_code_dict[new_diagnosis_id] = diagnosis_code
                new_diagnosis_code_id_dict[diagnosis_code] = new_diagnosis_id
            new_diagnosis_lst.append(new_diagnosis_code_id_dict[diagnosis_code])

        operation_lst = sample[1]
        new_operation_lst = []
        for operation_id in operation_lst:
            if operation_id == 0:
                continue
            operation_code = operation_id_code_dict[operation_id]
            operation_name = operation_code_name_dict[operation_code]
            if operation_code not in new_operation_code_name_dict:
                new_operation_code_name_dict[operation_code] = operation_name
                new_operation_id = len(new_operation_id_code_dict) + 1
                new_operation_id_code_dict[new_operation_id] = operation_code
                new_operation_code_id_dict[operation_code] = new_operation_id
            new_operation_lst.append(new_operation_code_id_dict[operation_code])
        if len(new_operation_lst) == 0:
            new_operation_lst = [0]

        drgs_id = sample[2]
        drgs_code = drgs_id_code_dict[drgs_id]
        drgs_name = drgs_code_name_dict[drgs_code]
        if drgs_code not in new_drgs_code_name_dict:
            new_drgs_code_name_dict[drgs_code] = drgs_name
            new_drgs_id = len(new_drgs_id_code_dict)
            new_drgs_id_code_dict[new_drgs_id] = drgs_code
            new_drgs_code_id_dict[drgs_code] = new_drgs_id
        new_drgs_id = new_drgs_code_id_dict[drgs_code]

        new_samples.append([new_diagnosis_lst, new_operation_lst, new_drgs_id])



    return new_drgs_code_id_dict, new_drgs_id_code_dict, new_drgs_code_name_dict, new_diagnosis_code_id_dict, new_diagnosis_id_code_dict, new_diagnosis_code_name_dict, new_operation_code_id_dict, new_operation_id_code_dict, new_operation_code_name_dict, new_samples


def data_stats(drgs_dict, diagnosis_dict, operation_dict, samples):
    print(drgs_dict)
    print("------drgs num------")
    print(len(drgs_dict))

    print("------diagnosis num------")
    print(len(diagnosis_dict))

    print("------operation num------")
    print(len(operation_dict))

    print("------samples num------")
    print(len(samples))

    drgs_samples_stats = {}
    for item in samples:
        drgs_code = item[2]
        if drgs_code not in drgs_samples_stats:
            drgs_samples_stats[drgs_code] = 0
        drgs_samples_stats[drgs_code] += 1
    print('------samples num of per class-------')
    print(drgs_samples_stats)


def dataset_split(dataset, traning_rate=0.6):
    new_dataset = {}
    for item in dataset:
        label = item[2]
        if label not in new_dataset:
            new_dataset[label] = []
        new_dataset[label].append(item)
    traning_set = []
    valid_set = []
    test_set = []
    for key, value in new_dataset.items():
        lst = new_dataset[key]
        for _ in range(5):
            shuffle(lst)
        traning_size = int(len(lst) * traning_rate)
        valid_test_size = int(len(lst) * (1-traning_rate))
        traning_set.extend(lst[0:traning_size])
        l1 = lst[traning_size:traning_size + int(valid_test_size/2)]
        valid_set.extend(l1)
        if len(l1) <= 0:
            print('-------l1-----------')
            print(key)
        l2 = lst[traning_size + int(valid_test_size/2):]
        test_set.extend(l2)
        if len(l2) <= 0:
            print('-------l2-----------')
            print(key)
    print('------training size-------')
    print(len(traning_set))
    for _ in range(5):
        shuffle(traning_set)
    print('------valid size-------')
    print(len(valid_set))
    for _ in range(5):
        shuffle(valid_set)
    print('------test size-------')
    print(len(test_set))
    for _ in range(5):
        shuffle(test_set)
    return traning_set, valid_set, test_set

def process():
    drgs_code_id_dict, drgs_id_code_dict, drgs_code_name_dict, diagnosis_code_id_dict, diagnosis_id_code_dict, diagnosis_code_name_dict, operation_code_id_dict, operation_id_code_dict, operation_code_name_dict, dataset = load_data()
    data_stats(drgs_code_id_dict, diagnosis_id_code_dict, operation_code_id_dict, dataset)
    drgs_code_id_dict, drgs_id_code_dict, drgs_code_name_dict, diagnosis_code_id_dict, diagnosis_id_code_dict, diagnosis_code_name_dict, operation_code_id_dict, operation_id_code_dict, operation_code_name_dict, dataset = re_load_data(drgs_code_id_dict, drgs_id_code_dict, drgs_code_name_dict, diagnosis_code_id_dict, diagnosis_id_code_dict, diagnosis_code_name_dict, operation_code_id_dict, operation_id_code_dict, operation_code_name_dict, dataset)
    data_stats(drgs_code_id_dict, diagnosis_id_code_dict, operation_code_id_dict, dataset)
    traning_set, valid_set, test_set = dataset_split(dataset)

    with open('../drgs-data/drgs_code_id_dict.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(['code', 'id'])
        for record in drgs_code_id_dict.items():
            spamwriter.writerow([record[0],record[1]])


    with open('../drgs-data/drgs_id_code_dict.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(['id', 'code'])
        for record in drgs_id_code_dict.items():
            spamwriter.writerow([record[0],record[1]])

    with open('../drgs-data/drgs_code_name_dict.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(['code', 'name'])
        for record in drgs_code_name_dict.items():
            spamwriter.writerow([record[0],record[1]])

    with open('../drgs-data/diagnosis_code_id_dict.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(['code', 'id'])
        spamwriter.writerow(['unknow',0])
        for record in diagnosis_code_id_dict.items():
            spamwriter.writerow([record[0],record[1]])

    with open('../drgs-data/diagnosis_id_code_dict.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(['id', 'code'])
        spamwriter.writerow(['0', 'unknow'])
        for record in diagnosis_id_code_dict.items():
            spamwriter.writerow([record[0],record[1]])

    with open('../drgs-data/diagnosis_code_name_dict.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(['code', 'name'])
        spamwriter.writerow(['unknow','unknow'])
        for record in diagnosis_code_name_dict.items():
            spamwriter.writerow([record[0],record[1]])

    with open('../drgs-data/operation_code_id_dict.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(['code', 'id'])
        spamwriter.writerow(['unknow',0])
        for record in operation_code_id_dict.items():
            spamwriter.writerow([record[0],record[1]])

    with open('../drgs-data/operation_id_code_dict.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(['id', 'code'])
        spamwriter.writerow([0,'unknow'])
        for record in operation_id_code_dict.items():
            spamwriter.writerow([record[0],record[1]])

    with open('../drgs-data/operation_code_name_dict.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(['code', 'name'])
        spamwriter.writerow(['unknow','unknow'])
        for record in operation_code_name_dict.items():
            spamwriter.writerow([record[0],record[1]])

    dataset = []
    dataset.extend(traning_set)
    dataset.extend(test_set)
    dataset.extend(valid_set)
    with open('../drgs-data/records.csv', 'w', newline='') as csvfile:
        #spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(['diagnosis', 'operation', 'drgs'])
        for record in dataset:
            spamwriter.writerow(record)
