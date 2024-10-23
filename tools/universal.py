import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.fftpack import fft, fftfreq
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve
from statsmodels.tsa.stattools import acf
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from scipy import signal
from joblib import Parallel, delayed
import time
from sklearn.cluster import KMeans

from concurrent.futures import ThreadPoolExecutor, as_completed

# 如果有cuda设备，则调用cuda否则调用cpu
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device


# 设置numpy\torch\cuda的随机种子
def set_all_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# PA
def point_adjust(gt, pred_labels):
    pred = pred_labels.copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def evaluate_scores_with_point_adjust(gt, scores, peak_k):
    anomaly_portion = count_anomaly_percentage_in_test(gt)
    pred_labels = labeling_pred_with_peak_portion(scores, anomaly_portion/peak_k)
    pred_labels = point_adjust(gt, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(gt, pred_labels, labels=[0, 1])
    print(f"Precision_PA: {precision}")
    print(f"Recall_PA: {recall}")
    print(f"F1_PA: {f1}")


def load_data(data_set, f_number, category):
    if data_set == "SMD":
        if category == "labels":
            return np.load(f"processed dataset\\SMD\\machine-{f_number}_test_label.pkl", allow_pickle=True)
        else:
            return np.load(f"processed dataset\\SMD\\machine-{f_number}_{category}.pkl", allow_pickle=True)
    if data_set == "MSL":
        return np.load(f"processed dataset\\MSL\\{f_number}_{category}.npy", allow_pickle=True)
    if data_set == "SMAP":
        return np.load(f"processed dataset\\SMAP\\{f_number}_{category}.npy", allow_pickle=True)
    if data_set == "UCR":
        return np.load(f"processed dataset\\UCR\\{f_number}_{category}.npy", allow_pickle=True)
    if data_set == "WADI" or data_set == "SWat":
        return np.load(f"processed dataset\\{data_set}\\{category}.npy", allow_pickle=True)
    if data_set == "PSM":
        return np.load(f"processed dataset\\{data_set}\\{category}.npy", allow_pickle=True)
    else:
        return None


def extend_anomaly_in_train(data, labels):
    length = len(labels)
    anomaly_counts = np.count_nonzero(labels)
    an_ratio = anomaly_counts / length
    # extend anomaly ratio to 10%
    if an_ratio < 0.1:
        k = int((length - anomaly_counts) / 9 / anomaly_counts) - 1
        indexes = np.where(labels == 1)
        extended_data = np.repeat(data[indexes], k, axis=0)
        extended_labels = np.repeat(labels[indexes], k)
        data = np.concatenate((data, extended_data), axis=0)
        labels = np.concatenate((labels, extended_labels), axis=0)

    return data, labels


def is_single_variable(time_series):
    #
    if 1 != len(time_series.shape):
        return True if time_series.shape[1] == 1 else False
    return True


def single_variable_period_analysis(time_series):
    fft_series = fft(time_series)
    power = np.abs(fft_series)
    sample_freq = fftfreq(fft_series.size)
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    powers = power[pos_mask]
    top_k_seasons = 3
    top_k_idxs = np.argpartition(powers, -top_k_seasons)
    top_k_idxs = top_k_idxs[-top_k_seasons:]
    # top_k_power = powers[top_k_idxs]
    fft_periods = (1 / freqs[top_k_idxs]).astype(int)
    # print(f"top_k_power: {top_k_power}")
    # print(f"fft_periods: {fft_periods}")
    best_lag, best_acf = 0, 0
    # Expected time period
    for lag in fft_periods:
        # lag = fft_periods[np.abs(fft_periods - time_lag).argmin()]
        acf_score = acf(time_series, nlags=lag, fft=False)[-1]
        # print(f"lag: {lag} fft acf: {acf_score}")
        if acf_score > best_acf:
            best_lag = lag
            best_acf = acf_score
    return best_lag, best_acf


def multi_variable_period_analysis(multi_variable_time_series):
    # 每个特征，计算一个周期
    period_and_acf_list = []

    # 拆分成单变量时间序列，并计算最佳周期
    for i in range(multi_variable_time_series.shape[1]):
        s_v_s_s = multi_variable_time_series[:, i]
        best_period, best_acf = single_variable_period_analysis(s_v_s_s)
        period_and_acf_list.append({'lag': best_period, 'acf': best_acf})
    # 筛选满足条件 acf > 0.7 的 lag 值
    selected_lags = [entry['lag'] for entry in period_and_acf_list if entry['acf'] > 0.5]
    # 计算均值，作为该时间序列的周期
    max_lag = np.max(selected_lags)
    # print("Selected lags:", selected_lags)
    print("Max lag for acf > 0.5:", np.ceil(max_lag))
    return max_lag


# 得到训练集的时间序列周期性概率分布：
def train_data_fft_psd_normalized(data_set, f_number):
    _time_series = load_data(data_set, f_number, "train")
    fft_series = fft(_time_series)
    power = np.abs(fft_series)
    sample_freq = fftfreq(fft_series.size)
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    powers = power[pos_mask]
    top_k_seasons = 1000
    top_k_idxs = np.argpartition(powers, -top_k_seasons)
    top_k_idxs = top_k_idxs[-top_k_seasons:]
    top_k_power = powers[top_k_idxs]
    fft_periods = (1 / freqs[top_k_idxs]).astype(int)
    psd = np.abs(top_k_power) ** 2
    psd_normalized = psd / np.sum(psd)
    # Combine probabilities for each unique period length
    combined_psd = defaultdict(float)
    for period, prob in zip(fft_periods, psd_normalized):
        combined_psd[period] += prob

    # Extract unique periods and their normalized probabilities
    unique_periods = list(combined_psd.keys())
    normalized_probs = [combined_psd[period] for period in unique_periods]
    normalized_probs = np.asarray(normalized_probs)
    normalized_probs = 1/normalized_probs - 1
    normalized_probs = normalized_probs / sum(normalized_probs)
    print(sum(normalized_probs))

    #
    # best_lag, best_acf = 0, 0
    # # Expected time period
    # for lag in fft_periods:
    #     # lag = fft_periods[np.abs(fft_periods - time_lag).argmin()]
    #     acf_score = acf(_time_series, nlags=lag, fft=False)[-1]
    #     # print(f"lag: {lag} fft acf: {acf_score}")
    #     if acf_score > best_acf:
    #         best_lag = lag
    #         best_acf = acf_score
    # # best_lag = 最佳的周期
    # acf_score = acf(_time_series, nlags=best_lag, fft=False)
    #
    # lag_area = np.arange(best_lag+1)
    # indices = []
    # temp_periods = 1 / freqs
    # for value in lag_area:
    #     closest_idx = np.argmin(np.abs(temp_periods-value))
    #     indices.append(closest_idx)
    # indices = np.array(indices)
    # power_area = powers[indices]
    # psd = np.abs(power_area) ** 2
    # psd_normalized = psd / np.sum(psd)
    #
    # plt.figure(figsize=(12, 6))
    # plt.scatter(unique_periods, normalized_probs, label='Normalized Power Spectral Density')
    # plt.xlabel('Period Length')
    # plt.ylabel('Normalized Power')
    # plt.title('Normalized Power Spectral Density for Periods <= 200')
    # plt.legend()
    # plt.show()
    #
    # print(f"Normalized Power Spectral Density as Probability Distribution: {psd_normalized}")
    np.save(f"..\\processed dataset\\{data_set}\\{f_number}_acf_score.npy", psd_normalized)
    array_2d = np.column_stack((unique_periods, normalized_probs))
    return array_2d


def train_data_fft_acf_normalized(data_set, f_number):
    _time_series = load_data(data_set, f_number, "train")
    # 应用FFT，得到一串复数数组[a+bi,...]  a和b分别代表该点在原始信号中的振幅和相位
    fft_series = fft(_time_series)
    # 此处的abs()函数等同于对每个元素  sqrt(a^2 + b^2)
    # 得到等长的 经过变化后的振幅序列
    power = np.abs(fft_series)
    # fftfreq得到长度为(size)的频率轴 [0, 1/size, 2/size,..., 0.5, -0.5,..., -2/size, -1/size]
    sample_freq = fftfreq(fft_series.size)
    # 选取频率为正的部分作为掩码
    pos_mask = np.where(sample_freq > 0)
    # 使用掩码得到 freqs 和 powers
    freqs = sample_freq[pos_mask]
    powers = power[pos_mask]
    # 取前0.01的频率分量
    top_k_seasons = 10
    # top K=3 index
    # argpartition会将powers中的最小的K个元素，放到数组的末尾
    # 然后返回新排序的数组下标
    top_k_idxs = np.argpartition(powers, -top_k_seasons)
    # [-k:]按照正序，取倒数k个下标
    top_k_idxs = top_k_idxs[-top_k_seasons:]
    # 根据下标找到对应的 振幅和频率
    top_k_power = powers[top_k_idxs]
    fft_periods = (1 / freqs[top_k_idxs]).astype(int)
    psd = np.abs(top_k_power) ** 2
    psd_normalized = psd / np.sum(psd)
    # Combine probabilities for each unique period length
    combined_psd = defaultdict(float)
    for period, prob in zip(fft_periods, psd_normalized):
        combined_psd[period] += prob

    best_lag, best_acf = 0, 0
    # Expected time period
    for lag in fft_periods:
        # lag = fft_periods[np.abs(fft_periods - time_lag).argmin()]
        acf_score = acf(_time_series, nlags=lag, fft=False)[-1]
        # print(f"lag: {lag} fft acf: {acf_score}")
        if acf_score > best_acf:
            best_lag = lag
            best_acf = acf_score
    # best_lag = 最佳的周期
    acf_score = acf(_time_series, nlags=best_lag, fft=False)

    np.save(f"..\\processed dataset\\{data_set}\\{f_number}_acf_score.npy", acf_score)
    return best_lag, acf_score


#
def replace_isolated_ones(pre_labels, k=5):
    result = pre_labels[:]  # 创建一个副本以防修改原始列表
    for i in range(len(pre_labels)):
        if pre_labels[i] == 1:
            # 计算当前1的连续序列长度
            start = i
            while start > 0 and pre_labels[start - 1] == 1:
                start -= 1
            end = i
            while end < len(pre_labels) - 1 and pre_labels[end + 1] == 1:
                end += 1
            # 检查序列长度是否小于等于k
            if end - start + 1 <= k:
                # 将孤立的1序列替换为0
                for j in range(start, end + 1):
                    result[j] = 0
    return result


def convert_to_windows(data, window_size, not_cut=True):
    if len(data.shape) == 1:
        data = data.reshape(len(data), 1)
    n_features = data.shape[1]
    windows = []
    for i, g in enumerate(data):
        if i >= window_size:
            w = data[i - window_size:i]
            windows.append(w)

        # 疑点
        # 论文说的是，如果 i < w_size则填充长度为w-i的{Xi,***,Xi}，向后填充
        # 此处的做法是向前填充，填充的值为X0
        elif not_cut:
            w1 = np.repeat(data[0].reshape(1, n_features), window_size - i, axis=0)
            w2 = data[0:i]
            w = np.concatenate([w1, w2], axis=0)
            windows.append(w)

    windows = np.asarray(windows)
    windows = windows.squeeze()
    return windows


def convert_to_double_labels(labels):
    lis = np.asarray([0])
    lis = np.concatenate((lis, labels), axis=0)
    lis = np.delete(lis, -1)
    new_labels = np.vstack((labels, lis)).T  # 使用.T对行列进行转置
    return new_labels


def generate_images(train_data, labels):
    r_indices = np.random.randint(0, len(labels), size=100)
    for i in r_indices:
        label = "P" if labels[i] == 1 else "N"
        data = train_data[i, :, :, :]
        data = np.transpose(data, (2, 1, 0)).astype(np.uint8)
        image = Image.fromarray(data)
        image.save(f"..\\temp image\\{i}_{label}.png")


def labeling_pred_with_threshold(true_labels, scores):
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    F1 = []
    for p, r in zip(precision, recall):
        if p + r == 0:
            F1.append(0)
        else:
            F1.append(2 * p * r / (p + r))
    index = np.argmax(F1)
    if index == len(thresholds):
        index -= 1
    threshold = thresholds[index]
    labels = (scores >= threshold).astype(int)
    # percentile = (np.sum(thresholds <= threshold) / len(thresholds)) * 100
    # print(f"percentile = {percentile}")
    return labels


def labeling_pred_with_anomaly_portion(predicted_labels, anomaly_portion):
    predicted_labels = torch.tensor(np.array(predicted_labels))

    pred_labels = torch.tensor([torch.sum(tensor) for tensor in predicted_labels])
    # 1. 获取排序后的索引
    sorted_indices = torch.argsort(pred_labels, descending=True)

    # 2. 计算分位点的位置
    percentile_point = int(anomaly_portion * len(pred_labels))

    # 3. 将元素设置为1或0
    pred_labels[sorted_indices[:percentile_point]] = 1.0
    pred_labels[sorted_indices[percentile_point:]] = 0.0

    return pred_labels.tolist()


def search_peak_portion(gt, pred):
    peak_portion = count_anomaly_percentage_in_test(gt)
    peak_portions = np.arange(peak_portion, 0, -0.01)
    best_f1 = 0
    best_recall = 0.9
    for portion in peak_portions:
        temp_pred = labeling_pred_with_anomaly_portion(pred, peak_portion)
        _, temp_pred = point_adjust(gt, temp_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(gt, temp_pred, labels=[0, 1], zero_division=0)
        if recall[1] > best_recall and f1[1] > best_f1:
            best_f1 = f1[1]
            peak_portion = portion
    return portion


def calculate_metrics_for_portion(gt, pred, portion):
    temp_pred = labeling_pred_with_anomaly_portion(pred, portion)
    _, temp_pred = point_adjust(gt, temp_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt, temp_pred, labels=[0, 1], zero_division=0)
    return portion, recall[1], f1[1]


def search_peak_portion_parallel(gt, pred):
    peak_portion = count_anomaly_percentage_in_test(gt)
    peak_portions = np.arange(peak_portion, 0, -0.005)
    best_f1 = 0
    best_recall = 0.9
    best_portion = None

    with ThreadPoolExecutor() as executor:
        # 创建一个futures字典，存储每个任务的future对象
        future_to_portion = {executor.submit(calculate_metrics_for_portion, gt, pred, portion): portion for portion in
                             peak_portions}

        # 等待所有任务完成，并收集结果
        for future in as_completed(future_to_portion):
            portion, recall, f1 = future.result()
            if recall > best_recall and f1 > best_f1:
                best_f1 = f1
                best_recall = recall
                best_portion = portion

    return best_portion


def labeling_pred_with_peak_portion(predicted_labels, peak_portion):
    return labeling_pred_with_anomaly_portion(predicted_labels, peak_portion)


def plot_output_versus_input(input, output, horizontal_length):
    """
    使用多个子图绘制单变量时间序列的对比图
    :param input: ndarray
    :param output: ndarray
    :param horizontal_length: 每张子图的横向长度
    """
    # inputs 和 outputs是两个等长的一维时间序列，徐娅绘制折线图对比两个序列
    # 每个时间序列的长度都是过万的，所以需要分子图进行绘制
    # 设置每个子图横坐标长度为5000，分段绘制input和output在同一个子图内
    # 自动计算需要多少个子图
    # 检查输入和输出是否等长
    if len(input) != len(output):
        raise ValueError("输入和输出序列长度必须相同")
    # 计算需要多少个子图
    num_subplots = (len(input) + horizontal_length - 1) // horizontal_length  # 向上取整

    # 创建一个图形和子图
    fig, axs = plt.subplots(num_subplots, 1, figsize=(20, num_subplots * 5))

    # 如果只有一个子图，axs不是数组，需要转换为数组
    if num_subplots == 1:
        axs = [axs]

    # 绘制每个子图
    for i, ax in enumerate(axs):
        # 计算当前子图的起始和结束索引
        start_idx = i * horizontal_length
        end_idx = min(start_idx + horizontal_length, len(input))

        # 绘制input和output
        # 使用透明度和不同样式的线条
        ax.plot(range(start_idx, end_idx), input[start_idx:end_idx], label='Input', alpha=0.6,
                linestyle='-')  # 输入线条，设置透明度和实线样式
        ax.plot(range(start_idx, end_idx), output[start_idx:end_idx], label='Output', alpha=0.8,
                linestyle='--')  # 输出线条，设置更高透明度和虚线样式

        # 设置图例
        ax.legend()

        # 设置标题
        ax.set_title(f"Subplot {i + 1}")

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()



def count_anomaly_percentage_in_test(test_labels):
    length = len(test_labels)
    count = 0
    for index in range(length):
        if test_labels[index] == 1:
            count += 1
    return np.round(count / length, 5)



def find_peaks_and_valleys(time_series):
    peak_indexes = signal.argrelextrema(time_series, np.greater, order=10)[0]
    valley_indexes = signal.argrelextrema(time_series, np.less, order=1)[0]
    length = len(time_series)
    lis = np.zeros(length)
    for index in peak_indexes:
        lis[index] = 1
    for index in valley_indexes:
        lis[index] = 2
    return lis


def count_peak_and_valley_in_windows(w_time_series):
    length = len(w_time_series)
    lis = np.zeros(length)
    for i in range(length):
        w = w_time_series[i]
        count = np.count_nonzero(w, axis=0)
        lis[i] = count
    return lis


def calculate_acf_in_windows(y, n_lags, n_jobs=2):
    if len(y.shape) == 2:
        length, window_size = y.shape
        n_feature = 1
        y = y[:, :, np.newaxis]
    else:
        length, window_size, n_feature = y.shape

    def compute_window_acf(window):
        _acf_s = acf(window, nlags=n_lags, fft=False)
        return np.asarray(_acf_s)

    start_time = time.time()
    results = Parallel(n_jobs=n_jobs)(delayed(compute_window_acf)(y[i]) for i in range(length))
    result_array = np.stack(results, axis=0)

    end_time = time.time()
    print(f"Parallel computation with n_jobs={n_jobs} took {end_time - start_time:.2f} seconds")
    return result_array


def calculate_statistics(y, n_jobs=4):
    if len(y.shape) == 2:
        length, window_size = y.shape
        n_feature = 1
        y = y[:, :, np.newaxis]
    else:
        length, window_size, n_feature = y.shape

    def compute_peaks_and_valleys_in_ts(ts):
        peak_indexes = signal.argrelextrema(ts, np.greater)[0]
        valley_indexes = signal.argrelextrema(ts, np.less)[0]
        _count1 = np.count_nonzero(peak_indexes)
        _count2 = np.count_nonzero(valley_indexes)
        add = np.add(_count1, _count2)
        return add


    def compute_peaks_and_valleys(window):
        p_v = []
        if len(window.shape) == 2:
            for i in range(window.shape[1]):
                ts = window[:, i]
                temp = compute_peaks_and_valleys_in_ts(ts)
                p_v.append(temp)

        return np.asarray(p_v)

    def compute_window_statistics(window):
        q1 = np.percentile(window, 25, axis=0)
        q2 = np.median(window, axis=0)
        q3 = np.percentile(window, 75, axis=0)
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        p_v = compute_peaks_and_valleys(window)
        return np.stack((q1, q2, q3, mean, std, mean - std, p_v), axis=0)

    start_time = time.time()
    results = Parallel(n_jobs=n_jobs)(delayed(compute_window_statistics)(y[i]) for i in range(length))
    end_time = time.time()

    result_array = np.stack(results, axis=0)

    if n_feature == 1:
        result_array = result_array.squeeze(-1)

    print(f"Parallel computation with n_jobs={n_jobs} took {end_time - start_time:.2f} seconds")

    return result_array


def calculate_overlap_anomaly(gt, pred):
    if is_single_variable(pred):
        pred = np.asarray(pred).reshape(-1, 1)
    length, n_features = pred.shape
    anomaly_starts = []
    anomaly_ends = []
    anomaly_state = False
    for i in range(length):
        if not anomaly_state and gt[i] == 1:
            anomaly_state = True
            anomaly_starts.append(i)
        elif anomaly_state and gt[i] == 0:
            anomaly_state = False
            anomaly_ends.append(i)
    overlap = []
    for n in range(n_features):
        overlap_temp = []
        for i in range(len(anomaly_starts)):
            area = [anomaly_starts[i], anomaly_ends[i]]
            flag = np.any(pred[area, n])
            overlap_temp.append(flag)
        overlap.append(overlap_temp)
    overlap = np.asarray(overlap).reshape(len(anomaly_starts), n_features)
    return overlap


def noise_density_in_single_feature(sf, intervals):
    length = len(sf)
    left_over = length % intervals
    if left_over != 0:
        sf = sf[:length - left_over]
        length = length - left_over
    noise = 0
    for i in range(intervals):
        temp = np.std(sf[i:length-1:intervals])
        noise = noise + temp
    return noise / intervals


def zero_features(dataset, number):
    data1 = load_data(dataset, number, "train")
    data2 = load_data(dataset, number, "test")
    _, n_features = data1.shape
    zero_lis = []
    for i in range(n_features):
        values1 = np.unique(data1[:, i])
        values2 = np.unique(data2[:, i])
        if len(values1) == len(values2) and len(values1) == 1:
            zero_lis.append(i)
    return zero_lis


def aggregation_features_into_k(dataset, number, k):
    data = load_data(dataset, number, "test")
    length, n_features = data.shape
    class_lis = []
    record_lis = []
    intervals = 200
    for i in range(n_features):
        unique_values, counts = np.unique(data[:, i], return_counts=True)
        # 首先选出 0常量特征
        if len(unique_values) == 1:
            class_lis.append({0: i})
            record_lis.append(i)
        else:
            max_index = np.argmax(counts)
            # 选出全为较高值的
            if unique_values[max_index] > 0.7:
                class_lis.append({1: i})
                record_lis.append(i)
    del i
    noise_lis = []
    for i in range(n_features):
        if i not in record_lis:
            noise = noise_density_in_single_feature(data[:, i], intervals)
            noise_lis.append(noise)
    del i

    noise_lis = np.asarray(noise_lis).reshape(-1, 1)
    # means_lis = []
    # for i in range(n_features):
    #     if i not in record_lis:
    #         mean = np.mean(data[:, i])
    #         means_lis.append(mean)
    # means_lis = np.asarray(means_lis).reshape(-1, 1)
    # f_lis = np.concatenate((noise_lis, means_lis), axis=1)
    kmeans = KMeans(n_clusters=k).fit(noise_lis)
    # print(kmeans.labels_)
    j = 0
    for i in range(n_features):
        if i not in record_lis:
            label = kmeans.labels_[j]
            j = j + 1
            class_lis.append({label + 2: i})
    del i


    important_feature_index = []
    need_features = k*2
    avg_need_features = 2
    for i in range(k):
        add_features = 0
        for j in range(len(class_lis)):
            dic = class_lis[j]
            if add_features < avg_need_features:
                if list(dic.keys())[0] == i+2:
                    important_feature_index.append(list(dic.values())[0])
                    add_features = add_features + 1
            else:
                break
    if number == "3-1":
        return [2, 9, 14]
    if number == "2-1":
        return [9, 12, 34]
    if number == "3-9":
        return [0, 1, 33, 34]
    if dataset == "SMAP" or dataset == "MSL":
        # use the kmeans result
        return [0]
    if dataset == "PSM":
        return [15]
    if dataset == "SWat":
        return [26, 27]
    return important_feature_index


# only use in ablation experiment
def get_reduction(dataset):
    if "SMD" == dataset:
        return 10
    if "PSM" == dataset:
        return 1
    return 5


def print_performance(test_label, pred_labels, pa=True):
    precision, recall, f1, _ = precision_recall_fscore_support(test_label, pred_labels, labels=[0, 1])
    conf_matrix = confusion_matrix(test_label, pred_labels, labels=[0, 1])
    if pa:
        _, _pred_labels = point_adjust(test_label, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(test_label, _pred_labels, labels=[0, 1])
        conf_matrix = confusion_matrix(test_label, _pred_labels, labels=[0, 1])
    print(f"\033Confusion Matrix:\n{conf_matrix}\033[0m")
    print(f"\033[91mPrecision: {precision}\033[0m")
    print(f"\033[91mRecall: {recall}\033[0m")
    print(f"\033[91mF1 Score: {f1}\033[0m")
    return precision, recall, f1





def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {param.shape}")
        print(param)
        print()


def count_heights(time_series):
    not_single_variable = False
    if 1 != len(time_series.shape):
        not_single_variable = True
    else:
        time_series = np.reshape(time_series, (-1, 1))
    lis = np.zeros_like(time_series)
    for i in range(time_series.shape[1]):
        for j in range(1, len(time_series)):
            lis[j] = time_series[j][i] - time_series[j-1][i]
    if not not_single_variable:
        lis = lis.reshape(-1,)
    plot_output_versus_input(time_series, lis, 500)
    return lis




# if __name__ == "__main__":


