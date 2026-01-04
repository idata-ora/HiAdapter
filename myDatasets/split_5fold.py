import os
import random
import numpy as np
import pandas as pd

excel_file = "/data/lqy/OSPatch/allfiles.csv"  ######################################################
print('[dataset] loading dataset from %s' % (excel_file))
rows = pd.read_csv(excel_file)
print('[dataset] dataset from %s, number of cases=%d' % (excel_file, len(rows)))

file_list_folder = "/data/lqy/OSPatch/split" ######################################################
for fold in range(1):
    print('[fold] {}'.format(fold))
    train_file = os.path.join(file_list_folder, 'train_fold_'+str(fold)+'.txt')
    val_file = os.path.join(file_list_folder, 'val_fold_'+str(fold)+'.txt')
    test_file = os.path.join(file_list_folder, 'test_fold_'+str(fold)+'.txt')

    random.seed(1)
    ratio_val = 0.2  # 20% for validation
    ratio_test = 0.2  # 20% for test
    assert 0 <= fold <= 4, 'fold should be in 0 ~ 4'
    sample_index = random.sample(range(len(rows)), len(rows)) # random
    num_val = round((len(rows) - 1) * ratio_val)
    num_test = round((len(rows) - 1) * ratio_test)
    if fold < 1 / ratio_val - 1:
        val_split = sample_index[fold * num_val: (fold + 1) * num_val]
        test_split = sample_index[(fold * num_val) + num_val: (fold * num_val) + num_val + num_test]
    else:
        val_split = sample_index[fold * num_val:]
        test_split = sample_index[0: num_test]  # Take the remaining samples for testing
    train_split = [i for i in sample_index if i not in val_split and i not in test_split]
    print("[dataset] train split: {}, val split: {}, test split: {}".format(len(train_split), len(val_split), len(test_split)))
    # train_split = [i for i in sample_index if i not in val_split]  
    # print("[dataset] train split: {}, val split: {}".format(len(train_split), len(val_split)))

    # save val
    val_split_array = np.array(val_split)
    val_data = []
    for idx in val_split_array:
        val_data.append(rows.iloc[idx, :].to_numpy())
    val_data_array = np.array(val_data)
    np.savetxt(val_file, val_data_array, fmt='%s', delimiter=',')
    print("val dataset save to {}".format(val_file))  

    # save train
    train_split_array = np.array(train_split)
    train_data = []
    for idx in train_split_array:
        train_data.append(rows.iloc[idx, :].to_numpy())
    train_data_array = np.array(train_data)
    np.savetxt(train_file, train_data_array, fmt='%s', delimiter=',')
    print("train dataset save to {}".format(train_file))

    # # save test
    test_split_array = np.array(test_split)
    test_data = []
    for idx in test_split_array:
        test_data.append(rows.iloc[idx, :].to_numpy())
    test_data_array = np.array(test_data)
    np.savetxt(test_file, test_data_array, fmt='%s', delimiter=',')
    print("test dataset save to {}".format(test_file))