from sklearn.model_selection import train_test_split
import random

class SubjectDataset:
    def __init__(self, data, folds):
        self.data = data
        self.folds = folds

    # def __len__(self):
    #     return self.data.shape[0]

    def split_dataset_according_2_subject(self, fold: object) -> object:
        # 提取数据中的 subject_id
        subject_ids = [entry['subject_id'] for entry in self.data]

        # 获取不同 subject_id 的个数
        unique_subjects = list(set(subject_ids))
        seed = 21
        random.seed(seed)
        random.shuffle(unique_subjects)
        num_unique_subjects = len(unique_subjects)

        n_per=int(num_unique_subjects/self.folds)

        trll = 0
        trlr = fold * n_per
        # trlr = 6 * n_per
        vall = trlr
        valr = fold * n_per + n_per
        #valr = 6 * n_per + 4* n_per
        trrl = valr
        trrr = num_unique_subjects

        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))

        # train_indices = [unique_subjects[i] for i in train_left_indices + train_right_indices]
        val_indices = [unique_subjects[i] for i in list(range(vall,valr))]
        train_indices = list(set(unique_subjects) - set(val_indices))
        print('train_subject:',train_indices,'test_subject:',val_indices)
        # 根据划分好的 subject_id 获取对应的数据
        train_data = [entry for entry in self.data if entry['subject_id'] in train_indices]
        test_data = [entry for entry in self.data if entry['subject_id'] in val_indices]

        return train_data, test_data

    def split_dataset_according_2_subject_seed(self, seed, fold: object) -> object:
        # 提取数据中的 subject_id
        subject_ids = [entry['subject_id'] for entry in self.data]

        # 获取不同 subject_id 的个数
        unique_subjects = list(set(subject_ids))
        random.seed(seed)
        random.shuffle(unique_subjects)
        num_unique_subjects = len(unique_subjects)

        n_per = int(num_unique_subjects / self.folds)

        trll = 0
        trlr = fold * n_per
        # trlr = 6 * n_per
        vall = trlr
        valr = fold * n_per + n_per
        # valr = 6 * n_per + 4* n_per
        trrl = valr
        trrr = num_unique_subjects

        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))

        # train_indices = [unique_subjects[i] for i in train_left_indices + train_right_indices]
        val_indices = [unique_subjects[i] for i in list(range(vall, valr))]
        train_indices = list(set(unique_subjects) - set(val_indices))
        print('train_subject:', train_indices, 'test_subject:', val_indices)
        # 根据划分好的 subject_id 获取对应的数据
        train_data = [entry for entry in self.data if entry['subject_id'] in train_indices]
        test_data = [entry for entry in self.data if entry['subject_id'] in val_indices]

        return train_data, test_data, train_indices, val_indices

    def split_dataset_according_2_subject_v1(self, fold: object) -> object:
        # 提取数据中的 subject_id
        subject_ids = [entry['subject_id'] for entry in self.data]
        # 获取不同 subject_id 的个数
        unique_subjects = list(set(subject_ids))
        seed = 131
        random.seed(seed)
        random.shuffle(unique_subjects)
        num_unique_subjects = len(unique_subjects)

        # 计算该fold中的训练集和测试集的subject范围
        train_start = 0
        train_end = train_start + int(0.7 * num_unique_subjects)

        train_indices = [unique_subjects[i] for i in range(train_start, train_end)]
        test_indices = list(set(unique_subjects) - set(train_indices))
        print('train_subject:', train_indices, 'test_subject:', test_indices)

        # 根据划分好的subject_id获取对应的数据
        train_data = [entry for entry in self.data if entry['subject_id'] in train_indices]
        test_data = [entry for entry in self.data if entry['subject_id'] in test_indices]

        return train_data, test_data



    def split_dataset_according_2_all_data_v1(self,fold):
        subject_ids = [entry['subject_id'] for entry in self.data]
        num_data = len(subject_ids)
        list_ = [i for i in range(num_data)]
        random.shuffle(list_)
        # train data
        train_start = 0
        train_end = train_start + int(0.7 * num_data)

        train_indices = [list_[i] for i in range(train_start, train_end)]
        test_indices = list(set(list_) - set(train_indices))

        train_data = [self.data[i] for i in train_indices]
        test_data = [self.data[i] for i in test_indices]
        return train_data, test_data

    def split_dataset_according_2_all_data(self, fold):
        # 提取数据中的 subject_id
        subject_ids = [entry['subject_id'] for entry in self.data]
        num_data = len(subject_ids)
        list_ = [i for i in range(num_data)]
        seed=25
        random.seed(seed)
        random.shuffle(list_)
        n_per=int(num_data/self.folds)

        trll = 0
        trlr = fold * n_per
        # trlr = 7 * n_per
        vall = trlr
        valr = fold * n_per + n_per
        # valr = 7 * n_per + 3* n_per
        trrl = valr
        trrr = num_data

        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))

        # train_indices = [list_[i] for i in train_left_indices + train_right_indices]
        # val_indices = [list_[i] for i in list(range(vall,valr))]
        val_indices = [list_[i] for i in list(range(vall, valr))]
        train_indices = list(set(list_) - set(val_indices))
        # print('train_index:',train_indices,'test_index:',val_indices)
        # 根据划分好的 subject_id 获取对应的数据
        train_data = [self.data[i] for i in train_indices] #self.data[train_indices]#[entry for entry in self.data if entry['subject_id'] in train_indices]
        test_data =[self.data[i] for i in val_indices]#[entry for entry in self.data if entry['subject_id'] in val_indices]

        # train_save_path = f"{save_path}/train_data_fold_{fold}.pkl"
        # test_save_path = f"{save_path}/test_data_fold_{fold}.pkl"

        # with open(train_save_path, 'wb') as train_file:
        #     pickle.dump(train_data, train_file)

        # with open(test_save_path, 'wb') as test_file:
        #     pickle.dump(test_data, test_file)
        return train_data, test_data


    def split_dataset_according_2_all_data_seed(self, seed, fold):
        # 提取数据中的 subject_id
        subject_ids = [entry['subject_id'] for entry in self.data]
        num_data = len(subject_ids)
        list_ = [i for i in range(num_data)]
        random.seed(seed)
        random.shuffle(list_)
        n_per=int(num_data/self.folds)
        trlr = fold * n_per
        vall = trlr
        valr = fold * n_per + n_per

        val_indices = [list_[i] for i in list(range(vall, valr))]
        train_indices = list(set(list_) - set(val_indices))
        train_data = [self.data[i] for i in train_indices] #self.data[train_indices]#[entry for entry in self.data if entry['subject_id'] in train_indices]
        test_data =[self.data[i] for i in val_indices]#[entry for entry in self.data if entry['subject_id'] in val_indices]
        return train_data, test_data, train_indices, val_indices,seed
        
    # def __getitem__(self, data_id):
    #     return self.data[data_id, :, :, :], self.label[data_id, ]




from torch.utils.data import Dataset
import numpy as np
import torch
class CustomDataset(Dataset):
    def __init__(self, data, label):
        # print(data.shape)
        self.data = torch.reshape(torch.from_numpy(data),(data.shape[0],1,data.shape[1],data.shape[2])).type(torch.float32)
        # print(self.data.shape)
        self.label = torch.from_numpy(label).type(torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, data_id):
        return self.data[data_id,:,:,:],self.label[data_id,]


