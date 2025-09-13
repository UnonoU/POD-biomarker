import sys
# sys.path.append("/mnt/dataset0/zhuyan/narcosis/thu_ep")
# sys.path.append("/mnt/dataset0/zhuyan/narcosis")
import argparse
import torch
import pickle
import re
import os
import numpy as np
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from itertools import combinations
from matplotlib import pyplot as plt
from itertools import combinations
import pickle
import itertools
from model import ConvNet_baseNonlinearHead, ConvNet_baseNonlinearHead_new, ConvNet_baseNonlinearHead_window
from SubjectDataset import SubjectDataset, CustomDataset
import mne
from sklearn.preprocessing import StandardScaler   

# Create the model instance
parser = argparse.ArgumentParser(description='Finetune the pretrained model for EEG emotion recognition')
parser.add_argument('--epochs-finetune', default=100, type=int, metavar='N',
                    help='number of total epochs to run in finetuning')
parser.add_argument('--max-tol', default=30, type=int, metavar='N',
                    help='number of max tolerence for epochs with no val loss decrease in finetuning')
parser.add_argument('--batch-size-finetune', default=270, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-finetune', default=0.0005, type=float, metavar='LR',
                    help='learning rate in finetuning')

parser.add_argument('--gpu-index', default=1, type=int, help='Gpu index.')

parser.add_argument('--epochs-pretrain', default=100, type=int, metavar='N',
                    help='number of total epochs to run in pretraining')
parser.add_argument('--restart_times', default=3, type=int, metavar='N',
                    help='number of total epochs to run in pretraining')
parser.add_argument('--max-tol-pretrain', default=30, type=int, metavar='N',
                    help='number of max tolerence for epochs with no val loss decrease in pretraining')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help=' n views in contrastive learning')
parser.add_argument('--batch-size-pretrain', default=24, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.0007, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', default=0.015, type=float,
                    metavar='W', help='weight decay (default: 0.05)',
                    dest='weight_decay')
parser.add_argument('--temperature', default=0.07, type=float,
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
parser.add_argument('--timeLen', default=5, type=int,
                    help='time length in seconds')
parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')

parser.add_argument('--timeFilterLen', default=60, type=int,
                    help='time filter length')
parser.add_argument('--n_spatialFilters', default=16, type=int,
                    help='time filter length')
parser.add_argument('--n_timeFilters', default=16, type=int,
                    help='time filter length')
parser.add_argument('--multiFact', default=2, type=int,
                    help='time filter length')

args = parser.parse_args()
args.device = torch.device('cuda:0')
stratified = ['initial', 'middle1', 'middle2']
# model = ConvNet_baseNonlinearHead(n_spatialFilters=10, n_timeFilters=10, timeFilterLen=125, n_channs=29, stratified=stratified, multiFact=2,
#                                   isMaxPool=False, out_dim=2, args=args).to(args.device)
model = ConvNet_baseNonlinearHead_new(n_spatialFilters=10, n_timeFilters=10, timeFilterLen=125, n_channs=29, stratified=stratified, multiFact=2,
                                  isMaxPool=False, out_dim=2, args=args, sequence_length=2500).to(args.device)
# model = ConvNet_baseNonlinearHead_window(n_spatialFilters=10, n_timeFilters=10, timeFilterLen=188, n_channs=29, stratified=stratified, multiFact=2,
#                                   isMaxPool=False, out_dim=2, args=args, sequence_length=2500).to(args.device)

with open('/mnt/dataset1/UnonoU/POD-biomarker/dataset/data_A_result.pkl', 'rb') as file_a:
    data_A = pickle.load(file_a)

with open('/mnt/dataset1/UnonoU/POD-biomarker/dataset/data_B_result.pkl', 'rb') as file_b:
    data_B = pickle.load(file_b)


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy
def evaluate_and_mask_spatial_conv(model, data_loader, filter_index, original_weights):
    # 屏蔽特定filter
    with torch.no_grad():
        # 在这里进行深复制以保护原始权重
        current_weights = original_weights.clone()
        current_weights[filter_index] = torch.zeros_like(current_weights[filter_index])
        model.spatialConv.weight.data = current_weights
    # 评估修改后的模型性能
    masked_accuracy = evaluate_model(model, data_loader)
    # 恢复权重
    model.spatialConv.weight.data = original_weights
    return masked_accuracy

def evaluate_and_mask_temporal_conv(model, data_loader, filter_index, original_weights):
    with torch.no_grad():
        current_weights = original_weights.clone()
        current_weights[filter_index] = torch.zeros_like(current_weights[filter_index])
        model.timeConv.weight.data = current_weights
    masked_accuracy = evaluate_model(model, data_loader)
    model.timeConv.weight.data = original_weights
    return masked_accuracy


def evaluate_and_mask_both_conv(model, data_loader, original_weights_spatial, original_weights_temporal, original_accuracy, n):
    max_impact = 0
    max_group = {'spatial': None, 'temporal': None}
    top_impacts = {'spatial': [], 'temporal': []}
    top_impacts = {'spatial': [], 'temporal': []}
    impacts = []

    spatial_groups = [list(comb) for comb in itertools.combinations(range(len(original_weights_spatial)), n)]
    temporal_groups = [list(comb) for comb in itertools.combinations(range(len(original_weights_temporal)), n)]

    for spatial_group in spatial_groups:
        for temporal_group in temporal_groups:
            with torch.no_grad():
                # Mask spatial filters
                current_weights_spatial = original_weights_spatial.clone()
                for filter_index in spatial_group:
                    current_weights_spatial[filter_index] = torch.zeros_like(current_weights_spatial[filter_index])
                model.spatialConv.weight.data = current_weights_spatial

                # Mask temporal filters
                current_weights_temporal = original_weights_temporal.clone()
                for filter_index in temporal_group:
                    current_weights_temporal[filter_index] = torch.zeros_like(current_weights_temporal[filter_index])
                model.timeConv.weight.data = current_weights_temporal

            # Evaluate the model with masked filters
            masked_accuracy = evaluate_model(model, data_loader)
            impact = (original_accuracy - masked_accuracy) / original_accuracy
            impacts.append((impact, spatial_group, temporal_group))

            # Update maximum impact and group if current impact is greater
            if max_impact <= impact:
                max_impact = impact
                max_group = {'spatial': spatial_group, 'temporal': temporal_group}

    # Restore original weights
    model.spatialConv.weight.data = original_weights_spatial
    model.timeConv.weight.data = original_weights_temporal

    # Sort the impacts and get top 5
    top_impacts = sorted(impacts, key=lambda x: x[0], reverse=True)[:5]
    top_groups = [(impact[1], impact[2]) for impact in top_impacts]

    return max_impact, max_group, top_impacts, top_groups

def save_contributions_and_weights(model, contribution_rates, mask_type, save_path):
    filter_weights = getattr(model, mask_type).weight.data.clone().cpu().numpy()
    contributions = np.array(contribution_rates)
    data_to_save = []
    for idx, (contribution, weights) in enumerate(zip(contributions, filter_weights)):
        data_to_save.append((idx, contribution, weights))
    # 将数据转换为结构化数组
    dtype = [('filter_index', int), ('contribution_rate', float), ('weights', np.float32, weights.shape)]
    structured_array = np.array(data_to_save, dtype=dtype)   
    # 保存到.npy文件
    np.save(save_path, structured_array)

def check_file_not_empty(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def normalize_segment(segment):
    scaler = StandardScaler()
    return scaler.fit_transform(segment.T).T  # 对每个通道进行标准化

n_folds=4
directory="/mnt/dataset0/zhuyan/narcosis/run_in_much_time_0612_lr_0.0015_wd_0.007_widow_188"
pattern = 'seed_(\d+)'
channel_file_path = '/mnt/dataset0/zhuyan/narcosis/channel_inf.txt'
with open(channel_file_path, 'r', encoding='utf-16') as file:
    lines = file.readlines()
ch_pos = {}
for line in lines[1:]:  # Skip the first line (header)
    parts = line.strip().split('\t')
    channel_number = int(parts[0])
    coordinates = list(map(float, parts[4:6]))
    if channel_number not in [18, 31, 32]:
        ch_pos[channel_number] = coordinates

for root, dirs, files in os.walk(directory):
    files.sort()
    if len(files) >= 2:
        count_pth_tar = 0
        for file in files:
            if file.endswith('.pth.tar'):
                count_pth_tar += 1
                if count_pth_tar == 2:
                    match = re.search(pattern, root)
                    seed = int(match.group(1))
                    #seed=130
                    checkpoint_path = os.path.join(root, file)
                    checkpoint_path_base=root
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                    save_root = os.path.join(checkpoint_path_base, 'contrbution_test_data')
                    save_results_root = os.path.join(checkpoint_path_base,  'contribution_test_results')
                    print('path:', checkpoint_path_base)
                    print('check_pth',checkpoint_path)
                    print('seed:',seed)

                    #检查 Temporal_contributions_all.npy 是否存在且不为空
                    # temporal_contributions_path = os.path.join(save_results_root, 'mask_filter',
                    #                                            'Temporal_contributions_all.npy')
                    # if not check_file_not_empty(temporal_contributions_path):
                    #     print(f'Temporal contributions file is missing or empty: {temporal_contributions_path}')
                    #     continue

                    sp_len = 188
                    fs = 125
                    time_conv_weights = checkpoint['state_dict']['timeConv.weight']
                    time_conv_weights_cpu = time_conv_weights.cpu().detach()
                    spatial_conv_weights = checkpoint['state_dict']['spatialConv.weight']
                    spatial_conv_weights_cpu = spatial_conv_weights.cpu().detach()
                    plt.figure(figsize=(26, 20))
                    for i in range(time_conv_weights_cpu.shape[0]):
                        plt.subplot(3, 4, i + 1)
                        time = np.arange(sp_len) / fs 
                        plt.plot(time,time_conv_weights_cpu[i, 0, 0, :].numpy())
                        plt.title(f'Time Filter {i + 1}')
                        plt.xlabel('Time(s)')
                        plt.ylabel('Weight')
                    save_path = os.path.join(checkpoint_path_base, 'time_filters.png')
                    plt.savefig(save_path)

                    all_data=[]
                    time_weights = time_conv_weights_cpu[:, 0, 0, :].numpy()
                    
                    freq = np.fft.fftfreq(sp_len * 2, d=1 / fs)
                    for i in range(10):
                        print(i)
                        plt.figure(figsize=(15, 3))
                        time_weight = time_weights[i, :][::-1]
                        tw_freq = abs(np.fft.fft(time_weight, sp_len * 2))
                        all_data.append({'time_weight': time_weight, 'tw_freq': tw_freq})
                        time = np.arange(sp_len) / fs
                        plt.subplot(121)
                        plt.plot(time, time_weights[i, :])
                        plt.title(f'Time Filter {i + 1}')
                        plt.xlabel('Time(s)')
                        plt.ylabel('Weight')
                        plt.subplot(122)
                        plt.plot(freq[:60], tw_freq[:60])
                        plt.title(f'Frequency Spectrum - Time Filter {i + 1}')
                        plt.xlabel('Frequency (Hz)')
                        plt.ylabel('Amplitude')
                        save_path = os.path.join(checkpoint_path_base, 'time_frequnency_filters_{}.png'.format(i + 1))
                        plt.savefig(save_path)
                    save_path_timecov= os.path.join(checkpoint_path_base,'time_conv.npy')
                    np.save(save_path_timecov, all_data)

                    spatial_data=[]
                    for i in range(spatial_conv_weights_cpu.shape[0]):
                        filter_index = i
                        specific_filter_weights = spatial_conv_weights_cpu[filter_index, 0, :, 0].numpy()
                        spatial_data.append({'spatial_weight':specific_filter_weights})
                        ch_pos_values = np.array(list(ch_pos.values()))
                        fig, ax = plt.subplots(figsize=(20, 20))
                        mne.viz.plot_topomap(specific_filter_weights, ch_pos_values, ch_type='eeg', res=100, axes=ax)
                        save_path = os.path.join(checkpoint_path_base, 'spatial_filters_with_ch_pos_filter_{}.png'.format(i + 1))
                        fig.savefig(save_path)
                    save_path_spatialcov = os.path.join(checkpoint_path_base, 'spatial_conv.npy')
                    np.save(save_path_spatialcov, spatial_data)

                    all_data = data_A + data_B
                    # 转换为 numpy 数组

                    all_data_ = np.array([normalize_segment(entry['segment_data']) for entry in all_data])
                    # all_data_ = np.array([entry['segment_data'] for entry in all_data])
                    all_label_ = np.array([entry['label_id'] for entry in all_data])
                    all_dataset = CustomDataset(all_data_, all_label_)

                    output_loader = torch.utils.data.DataLoader(all_dataset, batch_size=32, shuffle=True, num_workers=0)
                    # 获取原始权重
                    original_spatial_weights = model.spatialConv.weight.data.clone()
                    original_temporal_weights = model.timeConv.weight.data.clone()
                    # 计算原始准确率
                    original_accuracy = evaluate_model(model, output_loader)
                    print(f'Original Model Accuracy: {original_accuracy * 100:.2f}%')
                    # 计算空间卷积层的贡献率
                    contribution_rates_spatialConv = []
                    for filter_index in range(10):
                        masked_accuracy = evaluate_and_mask_spatial_conv(model, output_loader, filter_index,
                                                                         original_spatial_weights)
                        impact = (original_accuracy - masked_accuracy) / original_accuracy
                        contribution_rates_spatialConv.append(impact)
                        print(f"Filter: {filter_index}, Impact on Accuracy: {impact * 100:.2f}%")
                    print("SpatialConv Contribution Rates:", contribution_rates_spatialConv)
                    plt.figure(figsize=(10, 5))
                    contribution_rates_spatialConv_tensor = torch.tensor(contribution_rates_spatialConv)
                    plt.bar(range(contribution_rates_spatialConv_tensor.shape[0]),
                            contribution_rates_spatialConv_tensor.numpy())
                    plt.xlabel('Spatial Filter Index')
                    plt.ylabel('Contribution Rate')
                    plt.title('Contribution Rate of Spatial Convolution Filters')
                    # 保存
                    save_path_spatial = os.path.join(save_results_root, 'mask_filter', 'Spatial_contributions_all.png')
                    directory = os.path.dirname(save_path_spatial)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    plt.savefig(save_path_spatial)
                    save_spatial = os.path.join(save_results_root, 'mask_filter', 'Spatial_contributions_all.npy')
                    save_contributions_and_weights(model, contribution_rates_spatialConv, 'spatialConv', save_spatial)

                    # 计算时间卷积层的贡献率
                    contribution_rates_timeConv = []
                    for filter_index in range(10):
                        masked_accuracy = evaluate_and_mask_temporal_conv(model, output_loader, filter_index,
                                                                          original_temporal_weights)
                        impact = (original_accuracy - masked_accuracy) / original_accuracy
                        contribution_rates_timeConv.append(impact)
                        print(f"Filter: {filter_index}, Impact on Accuracy: {impact * 100:.2f}%")
                    print("TimeConv Contribution Rates:", contribution_rates_timeConv)
                    plt.figure(figsize=(10, 5))
                    contribution_rates_timeConv_tensor = torch.tensor(contribution_rates_timeConv)
                    plt.bar(range(contribution_rates_timeConv_tensor.shape[0]),
                            contribution_rates_timeConv_tensor.numpy())
                    plt.xlabel('Temporal Filter Index')
                    plt.ylabel('Contribution Rate')
                    plt.title('Contribution Rate of Temporal Convolution Filters')
                    save_path_temporal = os.path.join(save_results_root, 'mask_filter', 'Temporal_contributions_all.png')
                    directory = os.path.dirname(save_path_temporal)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    plt.savefig(save_path_temporal)
                    save_temporal = os.path.join(save_results_root, 'mask_filter', 'Temporal_contributions_all.npy')
                    save_contributions_and_weights(model, contribution_rates_timeConv, 'timeConv', save_temporal)

"""
                        # spatial_temporal
                        max_impacts = []
                        max_groups = []
                        top_groups = []
                        top_impacts = []
                        for length in range(1,4):
                            max_impact, max_group, top_impacts, top_groups=evaluate_and_mask_both_conv(model=model,data_loader=output_loader,
                                                                                                    original_weights_spatial=original_spatial_weights,
                                                                                                    original_weights_temporal=original_temporal_weights,
                                                                                                    original_accuracy=original_accuracy,n=length)
                            max_impacts.append(max_impact)
                            max_groups.append(max_group)
                            top_groups.append(top_groups)
                            top_impacts.append(top_impacts)
                            print("spatial_temporal_max_impacts", max_impacts)
                            print("spatial_temporal_max_groups", max_groups)
                            print("spatial_temporal_top5_groups", top_groups)
                            print("spatial_temporal_top5_impacts", top_impacts)

                        save_path_Spatial_temporal = os.path.join(save_results_root, 'mask_filter', f'both_spatial_temporal_filters_{fold + 1}.pkl')
                        directory = os.path.dirname(save_path_Spatial_temporal)
                        with open(save_path_Spatial_temporal, 'wb') as f:
                            pickle.dump((max_impacts, max_groups, top_groups, top_impacts), f)
"""



  





    
    


    
   