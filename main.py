import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import tools.universal as uni
import tools.deformation as deformation
from model.MultiScaleReconstructionNet import MultiScaleReconNet
import argparse


def load_all_data(data_set, number, window_size, not_cut=True):
    if config.artificial_label:
        tr, f_la = deformation.deform(data_set, number)
    else:
        tr = uni.load_data(data_set, number, "test")
        f_la = np.zeros(len(tr))

    te = uni.load_data(data_set, number, "test")
    la = uni.load_data(data_set, number, "labels")
    # tr = np.load("D:\\Projects\\Model Design and Test\\deformation\\SMD\\1-1_train_deformed.npy")
    # f_la = np.load("D:\\Projects\\Model Design and Test\\deformation\\SMD\\1-1_labels_deformed.npy")
    if config.ignore_zeros:
        lis = uni.zero_features(data_set, number)
        tr = np.delete(tr, lis, axis=1)
        te = np.delete(te, lis, axis=1)
    # convert to windows
    # the loss function needs f_la to be windows


    f_la = uni.convert_to_windows(f_la, window_size, not_cut)
    tr = uni.convert_to_windows(tr, window_size, not_cut)
    te = uni.convert_to_windows(te, window_size, not_cut)
    if not not_cut:
        la = la[window_size:]
    return tr, f_la, te, la


def K_distance_between_tensors(tensor_1, tensor_2, p, k):
    """
    对窗口的最后一个点，即当前时间点给予权重k
    :param tensor_1: inputs
    :param tensor_2: outputs
    :param p: 1==曼哈顿距离 2==欧氏距离
    :param k: k倍权重
    :return: k倍权重后的距离
    """
    if tensor_1 is None:
        return torch.tensor(0)
    if k > 1:
        tensor_1[:, -1, :] = tensor_1[:, -1, :] * k
        tensor_2[:, -1, :] = tensor_2[:, -1, :] * k
    tensor_1, tensor_2 = torch.permute(tensor_1, (0, 2, 1)), torch.permute(tensor_2, (0, 2, 1))
    dis = F.pairwise_distance(tensor_1, tensor_2, p)
    return dis


class KLoss(nn.Module):
    def __init__(self, k):
        super(KLoss, self).__init__()
        self.k = k

    def forward(self, _input, _output, _targets):
        # 当targets为1时，认为idea output等于现在的output这样子距离就是0了
        _targets = _targets.unsqueeze(-1)
        idea_output = torch.where(_targets == 0, _input, _output)
        _error = K_distance_between_tensors(_output, idea_output, 1, self.k)
        _loss = torch.mean(_error)
        return _loss


# regular reconstruction based loss
class ReconLoss(nn.Module):
    def __init__(self, k):
        super(ReconLoss, self).__init__()
        self.k = k

    def forward(self, _input, _output, _targets):
        # 当targets为1时，认为idea output等于现在的output这样子距离就是0了
        _targets = _targets.unsqueeze(-1)
        idea_output = torch.where(_targets == 0, _input, _input)
        _error = K_distance_between_tensors(_output, idea_output, 1, self.k)
        _loss = torch.mean(_error)
        return _loss


def contrast_error(_gru0, _gru1, _targets):
    label0, label1 = _targets[:, 0], _targets[:, 1]
    new_label = label0 + label1
    new_label = torch.where(new_label > 1, 1, new_label)
    contrast_dist = F.pairwise_distance(_gru0, _gru1)
    contrast_dist = torch.sigmoid(contrast_dist)
    return contrast_dist, new_label


def _train(_config, _train_loader, _model_name):
    # 模型存储路径
    if _config.ablation:
        model_saved_path = f"ablation models\\{_config.data_set}\\{_model_name}.pth"
    else:
        model_saved_path = f"savedModel\\{_config.data_set}\\{_model_name}.pth"

    # 判断是否训练过
    if os.path.exists(model_saved_path):
        print("model_trained")
    else:
        # 模型初始化
        _model = MultiScaleReconNet(_config).to(device)
        _optimizer = optim.Adam(_model.parameters(), lr=_config.learning_rate)
        _model.to(device)
        _model.train()
        if _config.calibrated_reconstruct and _config.kDistance:
            _criterion = KLoss(_config.k_times)
        else:
            _criterion = ReconLoss(_config.k_times)
            # 早停训练
        _early_stopping = EarlyStopping(patience=3)
        for epoch in range(_config.epochs):
            _epoch_loss = 0.0  # 用于记录每个 epoch 的总损失
            # 使用 tqdm 包装 train_loader，创建进度条
            with tqdm(_train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}",
                      unit="batch") as _tqdm_train_loader:
                for _batch in _tqdm_train_loader:
                    _inputs, _targets = _batch[0], _batch[1]
                    _inputs, _targets = _inputs.to(device).float(), _targets.to(device).float()
                    _optimizer.zero_grad()

                    _outputs = _model(_inputs)
                    _loss = _criterion(_inputs, _outputs, _targets)
                    _loss.backward()
                    _optimizer.step()
                    # 累积每个 batch 的损失
                    _epoch_loss += _loss.item()
                    # 更新进度条的描述信息（显示损失）
                    _tqdm_train_loader.set_postfix(loss=_loss.item())
                # 手动关闭进度条
                _tqdm_train_loader.close()
            # 计算并打印每个 epoch 的平均损失
            _average_epoch_loss = _epoch_loss / len(_train_loader)
            print(f"Epoch {epoch + 1}/{config.epochs}, Average Loss: {_average_epoch_loss}")
            # 检查早停条件
            _early_stopping(_average_epoch_loss)
            if _early_stopping.early_stop:
                print("Early stopping")
                break
        os.makedirs(os.path.dirname(model_saved_path), exist_ok=True)
        torch.save(_model.state_dict(), model_saved_path)


def _test(_config, _test_loader, _test_label, _model_name):
    # load model
    _model = MultiScaleReconNet(_config)
    if _config.ablation:
        model_saved_path = f"ablation models\\{_config.data_set}\\{_model_name}.pth"
    else:
        model_saved_path = f"savedModel\\{_config.data_set}\\{_model_name}.pth"

    _model.load_state_dict(torch.load(model_saved_path))

    # uni.print_model_parameters(model=_model)
    _model.to(device)
    _model.eval()
    _scores, _pred_labels = [], []
    recon = []
    with torch.no_grad():
        for _batch in _test_loader:
            _inputs, _targets = _batch[0], _batch[1]
            _inputs, _targets = _inputs.to(device).float(), _targets.to(device).float()
            # 前向传播
            _outputs = _model(_inputs)
            # record reconstruction
            recon.extend(np.squeeze(_outputs[:, -1, :], 1).tolist())
            temp = np.squeeze(_outputs[:, -1, :].tolist())
            # calculate reconstruction error
            _outputs = K_distance_between_tensors(_inputs, _outputs, _config.pow, _config.k_times)
            _outputs = _outputs.cpu().numpy()
            _scores.extend(_outputs.tolist())
    _scores = np.asarray(_scores).reshape(-1, _config.n_features)
    recon = np.asarray(recon).reshape(-1, _config.n_features)
    half_w = int(_config.window_size / 2)
    _zeros = np.zeros_like(_scores[:half_w])
    _scores = np.concatenate((_scores[half_w:], _zeros), axis=0)

    _test_data = uni.load_data(_config.data_set, _config.number, "test")[_config.window_size:]
    # feature selection
    if _config.feature_selection:
        important_lis = uni.aggregation_features_into_k(_config.data_set, _config.number, 2)
    else:
        important_lis = np.arange(_test_data.shape[1])
    if _config.ignore_zeros:
        f_index_lis = np.zeros(_test_data.shape[1])
        del_lis = uni.zero_features(_config.data_set, _config.number)
        _test_data = np.delete(_test_data, del_lis, axis=1)
        count = 0
        for temp_j in range(len(f_index_lis)):
            if temp_j not in del_lis:
                f_index_lis[temp_j] = temp_j - count
            else:
                f_index_lis[temp_j] = - 1
                count = count + 1

        new_important_lis = []
        for item in important_lis:
            if f_index_lis[item] != -1:
                new_important_lis.append(f_index_lis[item])
        del important_lis
        important_lis = np.asarray(new_important_lis, dtype=int)
    # uni.plot_output_versus_input(_test_data[:, 0], recon[:, 0], horizontal_length=1000)
    anomaly_portion = uni.count_anomaly_percentage_in_test(_test_label)
    if _config.integrate_prediction:
        _pred_labels = np.zeros_like(_scores)
        # 只统计重要的列
        for i in important_lis:
            if _config.feature_selection:
                _pred_labels[:, i] = uni.labeling_pred_with_anomaly_portion(_scores[:, i], config.peak_portion)
            else:
                _pred_labels[:, i] = uni.labeling_pred_with_threshold(_test_label, _scores[:, i])
        _pred_labels = np.sum(_pred_labels, axis=1)
        redundancy = 0
        # redundancy only work in ablation
        if not _config.feature_selection:
            redundancy = uni.get_reduction(_config.data_set)
        _pred_labels = np.where(_pred_labels >= len(important_lis)-redundancy, 1, 0)
    else:
        # average the scores in important list
        temp_scores = _scores[:, important_lis]
        temp_scores = np.average(temp_scores, axis=1)
        _pred_labels = uni.labeling_pred_with_anomaly_portion(temp_scores, anomaly_portion)

    # uni.print_performance(_test_label, _pred_labels)
    _, _pred_labels = uni.point_adjust(_test_label, _pred_labels)
    return _pred_labels


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif ((self.mode == 'min' and score > self.best_score - self.min_delta) or
              (self.mode == 'max' and score < self.best_score + self.min_delta)):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Config:
    def __init__(self, data_set, number, batch_size, window_size, period_length, learning_rate, epochs,
                 n_features, pro_features, k_times, pow, peak_portion,
                 not_cut=False,
                 ignore_zeros=True, ablation=False, artificial_label=True, calibrated_reconstruct=True, kDistance=True,
                 feature_selection=True, integrate_prediction=True):
        # 数据集名称
        self.data_set = data_set
        # 数据集内的序号
        self.number = number
        self.batch_size = batch_size
        self.window_size = window_size
        # 序列可能的周期长度
        self.period_length = period_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        # 数据集的特征数
        self.n_features = n_features
        # 模型维度，降维或者升维
        self.pro_features = pro_features
        self.k_times = k_times
        self.padding_size = 0 if self.window_size % self.period_length == 0 else self.window_size - self.window_size % self.period_length
        self.pow = pow
        self.peak_portion = peak_portion
        # whether cut the front of time series
        self.not_cut = not_cut
        self.ignore_zeros = ignore_zeros
        # the complete edition of our model is to set the following configurations=True
        self.ablation = ablation
        self.artificial_label = artificial_label
        self.calibrated_reconstruct = calibrated_reconstruct
        self.kDistance = kDistance
        self.feature_selection = feature_selection
        self.integrate_prediction = integrate_prediction

def main():

    train_data, false_train_label, test_data, test_label = load_all_data(config.data_set, config.number,
                                                                                     config.window_size, config.not_cut)
    if not config.kDistance:
        config.k_times = 1
    # 转tensor
    train_data, false_train_label = torch.tensor(train_data), torch.tensor(false_train_label)
    # 转tensor Dataset 然后 DataLoader
    train_dataset = TensorDataset(train_data, false_train_label)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    if config.ablation:
        # simplified name
        model_name = f"{config.number}_AL={config.artificial_label}_CR={config.calibrated_reconstruct}_KD={config.kDistance}_FS={config.feature_selection}_IP={config.integrate_prediction}"
    else:
        # 模型命名，保存和加载
        model_name = f"KCR_{config.number}_{config.batch_size}_{config.window_size}_{config.pro_features}_k{config.k_times}"

    # 训练
    _train(config, train_loader, model_name)
    # 测试
    test_data, test_label = torch.tensor(test_data), torch.tensor(test_label)
    test_dataset = TensorDataset(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    # test_loader2 = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    pred_label = _test(config, test_loader, test_label, model_name)
    _, _, f1 = uni.print_performance(test_label, pred_label)
    return f1[1]


if __name__ == "__main__":
    uni.set_all_random_seed(42)
    device = uni.set_device()

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--number', type=str, default='1-1')
    parser.add_argument('--ablation', type=str, default=False)
    parser.add_argument('--artificial_label', type=str, default=True)
    parser.add_argument('--calibrated_reconstruct', type=str, default=True)
    parser.add_argument('--kDistance', type=str, default=True)
    parser.add_argument('--feature_selection', type=str, default=True)
    parser.add_argument('--integrate_prediction', type=str, default=True)
    # 解析命令行参数
    args = parser.parse_args()
    # 检查是否有足够的参数
    if len(sys.argv) < 2:
        print("Usage: python main.py <arg1> <arg2> ...")

    # args = sys.argv[1:]  # sys.argv[0] is script name
    dataset, number = args.dataset, args.number
    print(f"KCR on {dataset} {number} with parameters")
    with open("hyper parameters\\configs.json") as file:
        json_data = json.load(file)
    for item in json_data['data']:
        if item['data_set'] == dataset and item['number'] == number:
            filtered_data = item
    # filtered_data = [item for item in json_data['data'] if item['data_set'] == dataset and item['number'] == number]
    config = Config(filtered_data['data_set'], filtered_data['number'], filtered_data['batch_size'],
                    filtered_data['window_size'], filtered_data['period_length'], filtered_data['learning_rate'],
                    filtered_data['epochs'], filtered_data['n_features'], filtered_data['pro_features'],
                    filtered_data['k_times'], filtered_data['pow'], filtered_data['peak_portion'],
                    filtered_data['not_cut'], filtered_data['ignore_zeros'], filtered_data['ablation'],
                    filtered_data['artificial_label'], filtered_data['calibrated_reconstruct'], filtered_data['kDistance'],
                    filtered_data['feature_selection'], filtered_data['integrate_prediction'])
    if args.ablation:
        config.artificial_label = args.artificial_label
        config.calibrated_reconstruct = args.calibrated_reconstruct
        config.kDistance = args.kDistance
        config.feature_selection = config.feature_selection
        config.integrate_prediction = config.integrate_prediction
    main()



