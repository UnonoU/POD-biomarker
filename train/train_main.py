import sys
# sys.path.append('/mnt/dataset0/zhuyan/narcosis')
# sys.path.append('/mnt/dataset0/zhuyan/narcosis/thu_ep')
from SubjectDataset import SubjectDataset, CustomDataset
import pickle
import argparse
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from model import ConvNet_baseNonlinearHead, SpatialTemporalConv,ConvNet_baseNonlinearHead_new
from train_utils import train_earlyStopping
import random
from sklearn.preprocessing import StandardScaler
import os


parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')
parser.add_argument('--epochs-finetune', default=200, type=int, metavar='N',
                    help='number of total epochs to run in finetuning')
parser.add_argument('--max-tol', default=30, type=int, metavar='N',
                    help='number of max tolerence for epochs with no val loss decrease in finetuning')
parser.add_argument('--batch-size-finetune', default=270, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-finetune', default=0.0005, type=float, metavar='LR',
                    help='learning rate in finetuning')#0.0005

parser.add_argument('--gpu-index', default=1, type=int, help='Gpu index.')

parser.add_argument('--epochs-pretrain', default=200, type=int, metavar='N',
                    help='number of total epochs to run in pretraining')
parser.add_argument('--restart_times', default=2, type=int, metavar='N',
                    help='number of total epochs to run in pretraining')
parser.add_argument('--max-tol-pretrain', default=20, type=int, metavar='N',
                    help='number of max tolerence for epochs with no val loss decrease in pretraining')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help=' n views in contrastive learning')
parser.add_argument('--batch-size-pretrain', default=32, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR',
                    help='learning rate')#0.007
parser.add_argument('--weight-decay', default=0.01, type=float,
                    metavar='W', help='weight decay (default: 0.05)',
                    dest='weight_decay')#0.005
parser.add_argument('--temperature', default=0.007, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-times', default=1, type=int,
                    help='number of sampling times for one sub pair (in one session)')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--sample-method', default='cross', type=str,
                    help='how to sample pretrain data')
parser.add_argument('--tuneMode', default='linear', type=str,
                    help='how to finetune the parameters')
parser.add_argument('--hidden-dim', default=30, type=int,
                    help='number of hidden units')
parser.add_argument('--timeLen', default=20, type=int,
                    help='time length in seconds')
parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')

parser.add_argument('--timeFilterLen', default=125, type=int,
                    help='time filter length')
parser.add_argument('--n_spatialFilters', default=10, type=int,
                    help='time filter length')
parser.add_argument('--n_timeFilters', default=10, type=int,
                    help='time filter length')
parser.add_argument('--multiFact', default=2, type=int,
                    help='time filter length')

args = parser.parse_args()

random.seed(args.randSeed)
np.random.seed(args.randSeed)
torch.manual_seed(args.randSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(8)
sample_method = args.sample_method
pretrain = True
finetune = False
randomInit = True
fixFirstLayers = False
tuneMode = args.tuneMode
stratified = ['initial', 'middle1', 'middle2']
channel_norm = False
time_norm = False
if args.batch_size_pretrain == 28:
    label_type = 'cls9'
elif args.batch_size_pretrain == 24:
    label_type = 'cls2'

args.device = torch.device('cuda:0')

n_spatialFilters = args.n_spatialFilters
n_timeFilters = args.n_timeFilters
timeFilterLen = args.timeFilterLen
multiFact = 2
hidden_dim = args.hidden_dim
out_dim = 30
fs = 250
timeLen = args.timeLen

timeStep = 2
print(args)

segment_length = 20
overlap_ratio = 2


def normalize_per_subject(data):
    normalized_data = []
    scaler_dict = {}
    for subject_id in np.unique([entry['subject_id'] for entry in data]):
        subject_data = [entry for entry in data if entry['subject_id'] == subject_id]
        segment_data = np.array([entry['segment_data'] for entry in subject_data])

        scaler = StandardScaler()
        normalized_segment_data = scaler.fit_transform(segment_data)

        scaler_dict[subject_id] = scaler

        for i, entry in enumerate(subject_data):
            entry['segment_data'] = normalized_segment_data[i]
            normalized_data.append(entry)

    return normalized_data, scaler_dict

# 读取data_A.pkl
with open('/mnt/dataset1/UnonoU/POD-biomarker/dataset/data_A_result.pkl', 'rb') as file_a:
    data_A = pickle.load(file_a)

# 读取data_B.pkl
with open('/mnt/dataset1/UnonoU/POD-biomarker/dataset/data_B_result.pkl', 'rb') as file_b:
    data_B = pickle.load(file_b)
# 标准化函数
def normalize_segment(segment):
    scaler = StandardScaler()
    return scaler.fit_transform(segment.T).T  # 对每个通道进行标准化


base_dir = '/mnt/dataset1/UnonoU/POD-biomarker/result/model_train/'

for _ in range(20):
    seed =random.randint(0, 1000)
    save_dir = os.path.join(base_dir, 'seed_' + str(seed))

    train_save_sub_root = os.path.join(save_dir, 'train_dataset')
    test_save_sub_root = os.path.join(save_dir, 'test_dataset')
    train_result_sub_root = os.path.join(save_dir, 'result', 'train_result')
    val_result_sub_root = os.path.join(save_dir, 'result', 'val_result')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(train_save_sub_root):
        os.makedirs(train_save_sub_root)

    if not os.path.exists(test_save_sub_root):
        os.makedirs(test_save_sub_root)

    if not os.path.exists(train_result_sub_root):
        os.makedirs(train_result_sub_root)

    if not os.path.exists(val_result_sub_root):
        os.makedirs(val_result_sub_root)

    n_folds = 4
    total_size = len(data_A) + len(data_B)  # .shape[0]
    n_per = int(total_size / n_folds)

    if pretrain:
        results_pretrain = {}
        results_pretrain['train_top1_history'], results_pretrain['val_top1_history'] = np.zeros(
            (n_folds, args.epochs_pretrain)), np.zeros((n_folds, args.epochs_pretrain))
        results_pretrain['train_top5_history'], results_pretrain['val_top5_history'] = np.zeros(
            (n_folds, args.epochs_pretrain)), np.zeros((n_folds, args.epochs_pretrain))
        results_pretrain['train_loss_history'], results_pretrain['val_loss_history'] = np.zeros(
            (n_folds, args.epochs_pretrain)), np.zeros((n_folds, args.epochs_pretrain))
        results_pretrain['best_val_top1'], results_pretrain['best_val_top5'] = np.zeros(n_folds), np.zeros(n_folds)
        results_pretrain['best_val_loss'], results_pretrain['best_epoch'] = np.zeros(n_folds), np.zeros(n_folds)

        for fold in range(n_folds):
            train_save_sub_path = os.path.join(train_save_sub_root, str(fold + 1))
            test_save_sub_path = os.path.join(test_save_sub_root, str(fold + 1))
            train_result_sub_path = os.path.join(train_result_sub_root, str(fold + 1))
            val_result_sub_path = os.path.join(val_result_sub_root, str(fold + 1))

            if not os.path.exists(train_save_sub_path):
                os.makedirs(train_save_sub_path)
            if not os.path.exists(test_save_sub_path):
                os.makedirs(test_save_sub_path)
            if not os.path.exists(train_result_sub_path):
                os.makedirs(train_result_sub_path)
            if not os.path.exists(val_result_sub_path):
                os.makedirs(val_result_sub_path)

        for fold in range(n_folds):
            print('fold:', fold + 1, '/', n_folds)

            dataset_A = SubjectDataset(data_A, n_folds)
            train_data_A, test_data_A,train_index_A,test_index_A,seed_A = dataset_A.split_dataset_according_2_all_data_seed(seed, fold)
            dataset_B = SubjectDataset(data_B, n_folds)
            train_data_B, test_data_B,train_index_B,test_index_B,seed_B = dataset_B.split_dataset_according_2_all_data_seed(seed, fold)

            train_data = train_data_A + train_data_B
            test_data = test_data_A + test_data_B
            train_data_to_save = {
                'train_index_A': train_index_A,
                'train_index_B': train_index_B,
                'seed': seed}
            test_data_to_save = {
                'test_index_A': test_index_A,
                'test_index_B': test_index_B,
                'seed': seed}
            np.save(os.path.join(train_save_sub_root, str(fold + 1),'train_data.npy'),train_data_to_save)
            np.save(os.path.join(test_save_sub_root, str(fold + 1), 'test_data.npy'),test_data_to_save)

            # train_data_ = np.array([entry['segment_data'] for entry in train_data])
            # test_data_ = np.array([entry['segment_data'] for entry in test_data])
            # train_label_ = np.array([entry['label_id'] for entry in train_data])
            # test_label_ = np.array([entry['label_id'] for entry in test_data])
            #
            # train_dataset = CustomDataset(train_data_, train_label_)
            # test_dataset = CustomDataset(test_data_, test_label_)

            # 对训练和测试数据进行标准化
            train_data_normalized = np.array([normalize_segment(entry['segment_data']) for entry in train_data])
            test_data_normalized = np.array([normalize_segment(entry['segment_data']) for entry in test_data])
            train_label_ = np.array([entry['label_id'] for entry in train_data])
            test_label_ = np.array([entry['label_id'] for entry in test_data])
            train_dataset = CustomDataset(train_data_normalized, train_label_)
            test_dataset = CustomDataset(test_data_normalized, test_label_)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                                       shuffle=True, num_workers=4)
            val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                                     shuffle=True, num_workers=4)
            # print(train_data_.shape)
            model = ConvNet_baseNonlinearHead_new(n_spatialFilters,
            #model = ConvNet_baseNonlinearHead(n_spatialFilters,
                                                  n_timeFilters,
                                                  timeFilterLen,
                                                  29,
                                                  stratified=stratified,
                                                  multiFact=multiFact,
                                                  isMaxPool=False,
                                                  out_dim=2,
                                                  args=args,#).to(args.device)
                                                  sequence_length=2500).to(args.device)

            args.save_dir_ft = os.path.join(save_dir, str(fold + 1))

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005,
                                         weight_decay=0.007)  # weight_decay 1e-5, 1e-4, 1e-3, 1e-2, 1e-1

            # print(save_dir_ft)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs_finetune, gamma=0.8, last_epoch=-1,
                                                        verbose=False)
            #  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)
            criterion = torch.nn.CrossEntropyLoss().to(args.device)

            best_epoch, train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_confusion, best_confusion_train, confusionMat, train_confusion = train_earlyStopping(
                args, train_loader, val_loader, model, criterion, optimizer, scheduler, True)

            results_pretrain['train_loss_history'][fold, :], results_pretrain['val_loss_history'][fold, :] = train_loss_history, val_loss_history
            results_pretrain['best_val_top1'][fold] = results_pretrain['val_top1_history'][fold, best_epoch]
            results_pretrain['best_val_top5'][fold] = results_pretrain['val_top5_history'][fold, best_epoch]
            results_pretrain['best_val_loss'][fold] = results_pretrain['val_loss_history'][fold, best_epoch]
            results_pretrain['best_epoch'][fold] = best_epoch

            print("Size of best_confusion:", best_confusion.shape)

            np.save(os.path.join(train_result_sub_root, str(fold + 1), 'train_loss_history.npy'), train_loss_history)
            np.save(os.path.join(val_result_sub_root, str(fold + 1), 'val_loss_history.npy'), val_loss_history)
            np.save(os.path.join(train_result_sub_root, str(fold + 1), 'train_acc_history.npy'), train_acc_history)
            np.save(os.path.join(val_result_sub_root, str(fold + 1), 'val_acc_history.npy'), val_acc_history)
            np.save(os.path.join(val_result_sub_root, str(fold + 1), 'confusion_matrix.npy'), best_confusion)
            np.save(os.path.join(val_result_sub_root, str(fold + 1), 'confusion_train_matrix.npy'), best_confusion_train)
            np.save(os.path.join(val_result_sub_root, str(fold + 1), 'last_confusion_matrix.npy'), confusionMat)
            np.save(os.path.join(val_result_sub_root, str(fold + 1), 'last_confusion_train_matrix.npy'), train_confusion)

        with open(os.path.join(save_dir, 'results_pretrain.pkl'), 'wb') as f:
            pickle.dump(results_pretrain, f)
        print(save_dir)
        print('val loss mean: %.3f, std: %.3f; val acc top1 mean: %.3f, std: %.3f; val acc top5 mean: %.3f, std: %.3f' % (
            np.mean(results_pretrain['best_val_loss']), np.std(results_pretrain['best_val_loss']),
            np.mean(results_pretrain['best_val_top1']), np.std(results_pretrain['best_val_top1']),
            np.mean(results_pretrain['best_val_top5']), np.std(results_pretrain['best_val_top5'])))