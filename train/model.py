import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvNet_use(nn.Module):
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, inp_dim, hidden_dim, out_dim):
        super(ConvNet_use, self).__init__()
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.spatialConv = nn.Conv2d(n_timeFilters, n_timeFilters*n_spatialFilters, (n_channs, 1), groups=n_timeFilters)
        self.avgpool = nn.AvgPool2d((1, 30)) # 看一下样本的时间长度，让它在avgpool之后时间维度是1
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, input):
        # print(input.shape)
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, int(input.shape[0]/2))

        out = self.timeConv(input)
        out = self.spatialConv(out)
        print(out.shape)
        out = F.relu(out) 
        out = self.avgpool(out)
        # 可以在这里保存out
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def stratified_norm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_str[n_samples*i: n_samples*(i+1), :] = (out[n_samples*i: n_samples*(i+1), :] - out[n_samples*i: n_samples*(i+1), :].mean(
            dim=0)) / (out[n_samples*i: n_samples*(i+1), :].std(dim=0) + 1e-3)
    return out_str

def batch_norm(out):
    out_str = out.clone()
    out_str = (out - out.mean(dim=0)) / (out.std(dim=0) + 1e-3)
    return out_str

def stratified_layerNorm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_oneSub = out[n_samples*i: n_samples*(i+1)]
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0], -1, out_oneSub.shape[-1]).permute(0,2,1)
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0]*out_oneSub.shape[1], -1)
        out_oneSub_str = out_oneSub.clone()
        # We don't care about the channels with very small activations
        # out_oneSub_str[:, out_oneSub.abs().sum(dim=0) > 1e-4] = (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4] - out_oneSub[
        #     :, out_oneSub.abs().sum(dim=0) > 1e-4].mean(dim=0)) / (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4].std(dim=0) + 1e-3)
        out_oneSub_str = (out_oneSub - out_oneSub.mean(dim=0)) / (out_oneSub.std(dim=0) + 1e-3)
        out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str.reshape(n_samples, -1, out_oneSub_str.shape[1]).permute(
            0,2,1).reshape(n_samples, out.shape[1], out.shape[2], -1)
    return out_str

def batch_layerNorm(out):
    n_samples, chn1, chn2, n_points = out.shape
    out = out.reshape(n_samples, -1, n_points).permute(0,2,1)
    out = out.reshape(n_samples*n_points, -1)
    out_str = (out - out.mean(dim=0)) / (out.std(dim=0) + 1e-3)
    out_str = out_str.reshape(n_samples, n_points, chn1*chn2).permute(
        0,2,1).reshape(n_samples, chn1, chn2, n_points)
    return out_str


class SpatialTemporalConv(nn.Module):
    def __init__(self, num_kernel, timeFilterLen, n_channs, stratified, isMaxPool, out_dim, time_length, args):
        super(SpatialTemporalConv, self).__init__()
        self.spatialConv = nn.Conv2d(1, num_kernel, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, num_kernel, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))

        self.num_kernel = num_kernel
        self.stratified = stratified
        self.isMaxPool = isMaxPool
        self.args = args
        self.fc1 = nn.Linear(21248, 64)  # check input dim
        self.fc2 = nn.Linear(64, out_dim)
        self.mapping = nn.Linear(num_kernel*1*time_length*num_kernel, out_dim)
        self.softmax = torch.nn.Softmax()

    def forward(self, input):        
        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)
        out = F.elu(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc2(F.relu(self.fc1(out)))
        return self.softmax(out)


class ConvNet_baseNonlinearHead_new(nn.Module):
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, stratified,multiFact,  isMaxPool, out_dim, args, sequence_length):
        super(ConvNet_baseNonlinearHead_new, self).__init__()
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.avgpool = nn.AvgPool2d((1, 125))
        self.dropout = nn.Dropout(p=0.5)
        
        pooled_length = sequence_length // 125
        input_features = n_timeFilters *n_spatialFilters* pooled_length
        
        self.fc1 = nn.Linear(input_features, 32)
        self.fc2 = nn.Linear(32, out_dim)
        self.n_spatialFilters = n_spatialFilters
        self.n_timeFilters = n_timeFilters
        self.stratified = stratified
        self.isMaxPool = isMaxPool
        self.args = args

    def forward(self, input):
        out = self.spatialConv(input)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = self.timeConv(out)
        out = F.elu(out)
        out = self.avgpool(out)

        if 'middle1' in self.stratified:
            out = stratified_layerNorm(out, int(out.shape[0]/2))

        if self.isMaxPool:
            # Select the dim with max average values (half of the total dims)
            _, indices = torch.topk(out.mean(dim=3), out.shape[1]//2, dim=1)
            out_pooled = torch.zeros((out.shape[0], out.shape[1]//2, out.shape[2], out.shape[3])).to(self.args.device)
            for i in range(out.shape[0]):
                out_pooled[i,:,:,:] = out[i,indices[i,:,0]]
            out_pooled = out_pooled.reshape(out_pooled.shape[0], -1)
            return out_pooled, indices
        else:
            out = out.reshape(out.shape[0], -1)
            out = F.relu(self.fc1(out))
            out = self.dropout(out)
            out = F.relu(self.fc2(out))
            return out


class ConvNet_baseNonlinearHead_window(nn.Module):
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, stratified, multiFact, isMaxPool, out_dim, args, sequence_length):
        super(ConvNet_baseNonlinearHead_window, self).__init__()
        
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, timeFilterLen // 2))
        
        self.avgpool = nn.AvgPool2d((1, timeFilterLen))
        self.dropout = nn.Dropout(p=0.5)
        
        pooled_length = sequence_length // timeFilterLen
        input_features = n_timeFilters * n_spatialFilters * pooled_length
        
        self.fc1 = nn.Linear(input_features, 32)
        self.fc2 = nn.Linear(32, out_dim)
        
        self.n_spatialFilters = n_spatialFilters
        self.n_timeFilters = n_timeFilters
        self.stratified = stratified
        self.isMaxPool = isMaxPool
        self.args = args

    def forward(self, input):
        out = self.spatialConv(input)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = self.timeConv(out)
        out = F.elu(out)
        out = self.avgpool(out)

        if 'middle1' in self.stratified:
            out = stratified_layerNorm(out, int(out.shape[0] / 2))

        # 如果启用了最大池化
        if self.isMaxPool:
            _, indices = torch.topk(out.mean(dim=3), out.shape[1] // 2, dim=1)
            out_pooled = torch.zeros((out.shape[0], out.shape[1] // 2, out.shape[2], out.shape[3])).to(self.args.device)
            for i in range(out.shape[0]):
                out_pooled[i, :, :, :] = out[i, indices[i, :, 0]]
            out_pooled = out_pooled.reshape(out_pooled.shape[0], -1)
            return out_pooled, indices
        else:
            # 展平并通过全连接层
            out = out.reshape(out.shape[0], -1)
            out = F.relu(self.fc1(out))
            out = self.dropout(out)
            out = F.relu(self.fc2(out))
            return out

class ConvNet_baseNonlinearHead(nn.Module):
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, stratified, isMaxPool, out_dim, args):
        super(ConvNet_baseNonlinearHead, self).__init__()
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.avgpool = nn.AvgPool2d((1, 125))
        
        self.n_spatialFilters = n_spatialFilters
        self.n_timeFilters = n_timeFilters
        self.stratified = stratified
        self.isMaxPool = isMaxPool
        self.args = args
        # pooled_length = 2500 // 30
        # input_features = n_timeFilters * n_spatialFilters * pooled_length
        self.fc1 = nn.Linear(2000, 32)  # check input dim
        self.fc2 = nn.Linear(32, out_dim)

    def forward(self, input):
        # input shape:(N, 1, 29, 2500)
        # print(input.shape)
        # if 'initial' in self.stratified:
        #     input = stratified_layerNorm(input, int(input.shape[0]/2))
        # print(input.shape)
        # activations = {}
        # print('spatialConv input shape:',input.shape)
        out = self.spatialConv(input)
        # ouputshape:(N, n_spatialFilters, 1, 2500)

        # activations['spatialConv'] = out
        # print('spatialConv output shape:',out.shape)
        out = out.permute(0,2,1,3)
        # (N, 1, n_spatialFilters, 2500)

        # print('timeConv input shape:',out.shape)
        out = self.timeConv(out)
        # (N, n_timeFilters, 10, 2500)
        
        # activations['timeConv'] = out
        # print('timeConv output shape:',out.shape)
        out = F.elu(out)
        out = self.avgpool(out)
        # print('Conv output shape:', out.shape)

        if 'middle1' in self.stratified:
            out = stratified_layerNorm(out, int(out.shape[0]/2))

        if self.isMaxPool:
            # Select the dim with max average values (half of the total dims)
            _, indices = torch.topk(out.mean(dim=3), out.shape[1]//2, dim=1)
            out_pooled = torch.zeros((out.shape[0], out.shape[1]//2, out.shape[2], out.shape[3])).to(self.args.device)
            for i in range(out.shape[0]):
                out_pooled[i,:,:,:] = out[i,indices[i,:,0]]
            out_pooled = out_pooled.reshape(out_pooled.shape[0], -1)
            return out_pooled, indices
        else:
            out = out.reshape(out.shape[0], -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            return out
        
    def backward(self, output):
        # 利用梯度的反向传播，求出n_spatialFilter个spatial_Filter的贡献度(contribution)
        spatial_conv_weights = self.spatialConv.weight
        #spatial_conv_weights.requires_grad_()
        spatial_conv_gradients = torch.autograd.grad(output, spatial_conv_weights, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
       # spatial_conv_gradients = torch.autograd.grad(output, spatial_conv_weights, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
        spatial_contributions = spatial_conv_gradients.abs().sum(dim=(2, 3))
        # 使用 sorted 函数将列表从大到小排序，并保留原始索引
        spatial_contributions = sorted(enumerate(spatial_contributions), key=lambda x: x[1], reverse=True)

        # 提取排序后的值和索引
        spatial_sorted_values = [item[1] for item in spatial_contributions]
        spatial_sorted_indices = [item[0] for item in spatial_contributions]

        # 利用梯度的反向传播，求出n_timeFilter个time_Filter的贡献度(contribution)
        time_conv_weights = self.timeConv.weight
       # time_conv_weights.requires_grad_()
        time_conv_gradients = torch.autograd.grad(output, time_conv_weights, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
        time_contributions = time_conv_gradients.abs().sum(dim=(2, 3))
        # 使用 sorted 函数将列表从大到小排序，并保留原始索引
        time_contributions = sorted(enumerate(time_contributions), key=lambda x: x[1], reverse=True)

        # 提取排序后的值和索引
        time_sorted_values = [item[1] for item in time_contributions]
        time_sorted_indices = [item[0] for item in time_contributions]

        return spatial_sorted_values, spatial_sorted_indices, time_sorted_values, time_sorted_indices

class ConvNet_baseNonlinearHead_learnRescale(nn.Module):
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, multiFact):
        super(ConvNet_baseNonlinearHead_learnRescale, self).__init__()
        self.rescaleConv1 = nn.Conv2d(1, 1, (1, timeFilterLen))
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.rescaleConv2 = nn.Conv2d(1, 1, (1, timeFilterLen))
        self.avgpool = nn.AvgPool2d((1, 30))
        # self.bn1 = nn.BatchNorm2d(n_timeFilters)
        self.spatialConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*multiFact, (n_spatialFilters, 1), groups=n_timeFilters)
        self.timeConv2 = nn.Conv2d(n_timeFilters*multiFact, n_timeFilters*multiFact*multiFact, (1, 6), groups=n_timeFilters*multiFact)
        self.rescaleConv3 = nn.Conv2d(1, 1, (1, 6))
        # self.bn2 = nn.BatchNorm2d(n_timeFilters*multiFact*multiFact)
        self.n_spatialFilters = n_spatialFilters
        self.n_timeFilters = n_timeFilters

    def forward(self, input):
        out_tmp = self.rescaleConv1(input)
        out_mean = torch.mean(out_tmp, 3, True)
        out_var = torch.mean(out_tmp**2, 3, True)
        input = (input - out_mean) / torch.sqrt(out_var + 1e-5)

        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)

        out = out.reshape(out.shape[0], 1, out.shape[1]*out.shape[2], out.shape[3])
        out_tmp = self.rescaleConv2(out)
        out_mean = torch.mean(out_tmp, 3, True)
        out_var = torch.mean(out_tmp**2, 3, True)
        out = (out - out_mean) / torch.sqrt(out_var + 1e-5)
        out = out.reshape(out.shape[0], self.n_timeFilters, self.n_spatialFilters, out.shape[3])

        out = F.elu(out)
        out = self.avgpool(out)

        out = F.elu(self.spatialConv2(out))
        out = F.elu(self.timeConv2(out))

        out = out.permute(0,2,1,3)
        out_tmp = self.rescaleConv3(out)
        out_mean = torch.mean(out_tmp, 3, True)
        out_var = torch.mean(out_tmp**2, 3, True)
        out = (out - out_mean) / torch.sqrt(out_var + 1e-5)
        out = out.permute(0,2,1,3)

        out = out.reshape(out.shape[0], -1)
        return out
    


class simpleNN3(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim, n_samples, stratified):
        super(simpleNN3, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.n_samples = n_samples
        self.stratified = stratified
    def forward(self, input):
        # if self.stratified:
        #     input = stratified_norm(input, self.n_samples)
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        # if self.stratified:
        #     out = stratified_norm(out, self.n_samples)
        out = self.fc3(out)
        return out

class LSTM_NN(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim, n_samples, stratified):
        super(LSTM_NN, self).__init__()
        self.lstm = nn.LSTM(inp_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.n_samples = n_samples
        self.stratified = stratified
    def forward(self, input):
        # input: (batch, seq, features)
        input, _ = self.lstm(input)
        input = input.reshape(input.shape[0]*input.shape[1], -1)
        if self.stratified:
            input = stratified_norm(input, self.n_samples)
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        if self.stratified:
            out = stratified_norm(out, self.n_samples)
        out = self.fc3(out)
        return out