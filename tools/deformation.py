import random

import numpy as np
from scipy import signal
import tools.universal as uni



def count_anomaly_length(labels):
    count, anomaly_state = 0, False
    lis = []
    for i in range(len(labels)):
        if not anomaly_state:
            if labels[i] == 1:
                anomaly_state = True
                count = 1
        else:
            if labels[i] == 1:
                count = count + 1
            else:
                lis.append(count)
                count = 0
                anomaly_state = False
    lis = np.asarray(lis)
    return lis


def generate_slices(time_series, ratio=0.2, avg_anomaly_length=300, border=1000):
    length = len(time_series)
    # 002的平均异常长度
    total_anomaly_length = int(ratio * length)
    # 向下取整  count + 1 = 异常段数量
    count = total_anomaly_length // avg_anomaly_length
    # 异常不能出现在刚开始和结尾
    random_starts = np.random.choice(range(border, length - border), size=count + 1, replace=False)
    # 排序
    random_starts.sort()
    # 为了保证ratio，要控制异常段的长度和
    random_length = np.random.choice(range(int(avg_anomaly_length * 0.8), int(avg_anomaly_length * 1.2)), size=count)
    # 计算最后一个异常段的长度
    last_anomaly_length = total_anomaly_length - np.sum(random_length)
    random_length = np.append(random_length, last_anomaly_length)
    random_ends = random_starts + random_length
    lis = list(zip(random_starts, random_ends))
    return lis


def transverse_stretching(y, ratio):
    """
    :param y: 纵坐标values
    :param ratio: 缩放    比例
    """
    # ratio是超参数
    # UCR 001中横向拉伸6倍
    x = np.arange(len(y))
    new_length = int(len(y) * ratio)
    temp_x = np.linspace(x[0], x[-1], new_length)
    # 按照x和y的关系，进行均匀插值
    new_y = np.interp(temp_x, x, y)
    # 生成新的序列
    # new_x = np.arange(x[0], x[0] + length, 1)
    # 绘制原始数据和插值后的数据
    # plt.plot(x, y, 'o-', label='Original Data')
    # plt.plot(new_x, new_y, 'x-', label='Interpolated Data')
    #
    # plt.legend()
    # plt.show()
    return new_y


def change_height(y, height):
    return y + height


def vertical_stretching(y, ratio=1.5):
    """
    使用 y = -kx^2 + (1+k)x 进行线性变化
    x > 0 是放大 x < 0是缩小
    :param y: 一维的时间序列
    :param ratio: 变换比例 默认为1.5
    """
    new_y = -ratio * pow(y, 2) + (1 + ratio) * y
    # 保证变换后不会超过值域[0, 1]
    new_y[new_y > 1] = 1
    return new_y


def vertical_transpose(y):
    """
    垂直翻转时间序列
    :param y:一维单变量时间序列
    """
    return 1 - y


def horizontal_transpose(y):
    """
    水平翻转时间序列
    :param y:一维单变量时间序列
    """
    # 逆序返回ndarray
    return y[::-1]


def add_peak(y, peak_height=0):
    length = len(y)
    index = np.random.choice(range(0, length - 1), size=1)
    if peak_height == 0:
        y[index] = y[index] + 0.5
    else:
        y[index] = peak_height

    return y


def add_valley(y, valley_depth):
    length = len(y)
    index = np.random.choice(range(0, length - 1), size=1)
    y[index] = y[index] - valley_depth
    return y


# 替换为震动幅度很小的序列
def tiny_fluctuation_replacement(y, amplitude=0.01):
    """
    替换为震动幅度很小的序列
    :param y: 单变量时间序列
    :param amplitude: 振幅
    """
    length = len(y)
    center_value = np.mean(y)
    random_values = np.random.uniform(low=center_value - amplitude, high=center_value + amplitude, size=length)
    return random_values


def uniform_replacement(y, height=0):
    """
    替换为震动幅度很小的序列
    :param height:
    :param y: 单变量时间序列
    """
    center_value = np.mean(y)

    if height == 0:
        y[:] = center_value
    else:
        half_length = int(len(y) / 2)
        y[:half_length] = center_value
        y[half_length:] = center_value + height
    return y


def add_tiny_fluctuation(y, amplitude=0.03):
    length = len(y)
    fluctuations = np.random.uniform(low=-amplitude, high=amplitude, size=length)
    y = y + fluctuations
    return y


# 该方法是一维的, slice_index中不得包含时间序列的起点和终点
def slice_time_series(time_series, slice_index):
    slice_index = np.reshape(slice_index, (-1, 1)).squeeze()

    lis_start, lis_end = [], []
    start, end = 0, slice_index[0]
    lis_start.append(time_series[start:end])

    for i in range(1, len(slice_index)):
        start = end
        end = slice_index[i]
        if i % 2 == 0:
            lis_start.append(time_series[start:end])
        else:
            lis_end.append(time_series[start:end])
    lis_start.append(time_series[end:])
    return lis_start, lis_end


def concatenate_slices_with_false_labels(old_slice, new_slice):
    length = len(new_slice)
    lis = []
    # old
    lis_label = np.zeros(len(old_slice[0]))
    lis.extend(old_slice[0])
    for i in range(length):
        # new
        lis.extend(new_slice[i])
        temp_label = np.ones(len(new_slice[i]))
        lis_label = np.concatenate((lis_label, temp_label))
        # old
        lis.extend(old_slice[i + 1])
        temp_label = np.zeros(len(old_slice[i + 1]))
        lis_label = np.concatenate((lis_label, temp_label))

    return np.asarray(lis), lis_label


def alter_ts(train_data, lis, need_peak=False, need_valley=False, tiny_fluc_uniform=False, need_uniform=False,
             need_noise=False, transpose=False, need_transverse_stretching=False, transverse_stretching_ratio=0.5,
             need_vertical_stretching=False, vertical_stretching_ratio=0.5, uniform_height=0, valley_depth=0,
             need_change_height=False, height=0.0,
             peak_height=0.0):
    lis_start, lis_end = slice_time_series(train_data, lis)
    lis_temp = []
    for i in range(len(lis_end)):
        if need_transverse_stretching:
            temp = transverse_stretching(lis_end[i], transverse_stretching_ratio)
        if need_vertical_stretching:
            temp = vertical_stretching(lis_end[i], vertical_stretching_ratio)
        if tiny_fluc_uniform:
            temp = tiny_fluctuation_replacement(lis_end[i])
        if need_uniform:
            temp = uniform_replacement(lis_end[i], uniform_height)
        if need_noise:
            temp = add_tiny_fluctuation(lis_end[i])
        if need_valley:
            temp = add_valley(lis_end[i], valley_depth)
        if need_peak:
            temp = add_peak(lis_end[i], peak_height)
        if need_change_height:
            temp = change_height(lis_end[i], height)
        lis_temp.append(temp)
    new_ts, false_label = concatenate_slices_with_false_labels(lis_start, lis_temp)
    return new_ts, false_label


def complex_noise_in_SMD(y, lis1, lis2, lis3, lis4):
    length, features = y.shape
    half = int(length / 2)
    lis = []
    for index in range(features):
        temp = y[:, index]
        if index in lis1:
            noise = np.random.uniform(low=0.1, high=0.1 + 0.5, size=1)
            temp = temp + noise
            temp = np.where(temp < 1, temp, 1)
        elif index in lis2:
            indices = np.random.choice(range(0, length), size=int(length / 10))
            noise = np.random.uniform(low=0.2, high=0.2 + 0.5, size=len(indices))
            for j in range(len(indices)):
                temp[indices[j]] = temp[indices[j]] + noise[j]
            temp = np.where(temp < 1, temp, 1)
        elif index in lis3:
            indices = np.random.choice(range(0, length), size=int(length / 10))
            noise = np.random.uniform(low=0.2, high=0.2 + 0.5, size=len(indices))
            for j in range(len(indices)):
                temp[indices[j]] = temp[indices[j]] - noise[j]
            temp = np.where(temp > 0, temp, 0)
        elif index in lis4:
            if np.random.choice([True, False]):
                temp = temp + np.random.uniform(low=0.3, high=1, size=1)
                temp = np.where(temp < 1, temp, 1)
        lis.append(temp)
    lis = np.asarray(lis)
    lis = lis.reshape(length, features)
    return lis


def deform_SMD(machine_number):
    train_data = np.load(f"processed dataset\\SMD\\machine-{machine_number}_train.pkl", allow_pickle=True)
    test_label = np.load(f"processed dataset\\SMD\\machine-{machine_number}_test_label.pkl", allow_pickle=True)
    ratio = np.round(uni.count_anomaly_percentage_in_test(test_label), 2)
    ano_length, lis1, lis2, lis3, lis4 = example_SMD_deform_config(machine_number)
    lis = generate_slices(train_data, ratio, ano_length)
    lis_start, lis_end = slice_time_series(train_data, lis)
    lis_temp = []
    for i in range(len(lis_end)):
        temp = complex_noise_in_SMD(lis_end[i], lis1, lis2, lis3, lis4)
        lis_temp.append(temp)
    new_ts, false_label = concatenate_slices_with_false_labels(lis_start, lis_temp)
    return new_ts, false_label


def example_SMD_deform_config(number):
    if number == "1-1":
        return 500, [0, 1, 2, 3, 9, 11, 18, 19, 20, 21, 27, 30, 31, 34, 35], [8, 12, 13, 14], [], [6, 23, 25, 26, 28, 29, 32, 33]
    elif number == "3-1":
        return 500, [], [0, 1, 2, 3, 18, 19, 20, 21, 22, 27, 30], [23, 25], [26, 32, 33, 34, 35]
    elif number == "3-9":
        return 100, [], [0, 1, 2, 3], [23, 25], [9, 18, 19, 20, 22, 26, 27, 30, 31, 32, 33, 34, 35]
    elif number == "3-11:":
        return 50, [], [0, 1, 2, 3, 9, 10, 29, 31], [5, 6], []
    elif number == "1-6":
        return 500, [], [0, 1, 2, 3, 15, 23, 24], [4], []
    elif number == "2-6":
        return 500, [], [], [], []
    return 500, [], [], [], []


def complex_noise_in_PSM(y):
    length, features = y.shape
    lis1 = [19, 21, 22, 23, 24]
    lis2 = [4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19]
    lis = []
    for index in range(features):
        temp = y[:, index]
        if index in lis1:
            indices = np.random.choice(range(0, length), size=int(length / 10))
            noise = np.random.uniform(low=0.2, high=0.2 + 0.5, size=len(indices))
            for j in range(len(indices)):
                temp[indices[j]] = temp[indices[j]] + noise[j]
            temp = np.where(temp < 1, temp, 1)
        elif index in lis2:
            indices = np.random.choice(range(0, length), size=int(length / 10))
            noise = np.random.uniform(low=0.2, high=0.2 + 0.5, size=len(indices))
            for j in range(len(indices)):
                temp[indices[j]] = temp[indices[j]] - noise[j]
            temp = np.where(temp > 0, temp, 0)
        lis.append(temp)
    lis = np.asarray(lis)
    lis = lis.reshape(length, features)
    return lis


def deform_PSM():
    train_data = np.load("processed dataset\\PSM\\train.npy", allow_pickle=True)
    ratio = 0.10
    lis = generate_slices(train_data, ratio, 500)
    lis_start, lis_end = slice_time_series(train_data, lis)
    lis_temp = []
    for i in range(len(lis_end)):
        temp = complex_noise_in_PSM(lis_end[i])
        lis_temp.append(temp)
    new_ts, false_label = concatenate_slices_with_false_labels(lis_start, lis_temp)
    return new_ts, false_label


def complex_noise_in_SWat(y):
    length, features = y.shape
    lis1 = [4, 7, 8, 26]
    lis2 = [1, 6, 27, 28, 30, 33, 38, 39, 40, 41, 42, 44, 45, 46]
    lis = []
    for index in range(features):
        temp = y[:, index]
        if index in lis1:
            indices = np.random.choice(range(0, length), size=int(length / 10))
            noise = np.random.uniform(low=0.2, high=0.2 + 0.5, size=len(indices))
            for j in range(len(indices)):
                temp[indices[j]] = temp[indices[j]] + noise[j]
            temp = np.where(temp < 1, temp, 1)
        elif index in lis2:
            indices = np.random.choice(range(0, length), size=int(length / 10))
            noise = np.random.uniform(low=0.2, high=0.2 + 0.5, size=len(indices))
            for j in range(len(indices)):
                temp[indices[j]] = temp[indices[j]] - noise[j]
            temp = np.where(temp > 0, temp, 0)
        lis.append(temp)
    lis = np.asarray(lis)
    lis = lis.reshape(length, features)
    return lis


def deform_SWat():
    train_data = np.load("processed dataset\\SWaT\\train.npy", allow_pickle=True)
    ratio = 0.10
    downsample = True
    if downsample:
        lis = generate_slices(train_data, ratio, 100)
    else:
        lis = generate_slices(train_data, ratio, 1000)
    lis_start, lis_end = slice_time_series(train_data, lis)
    lis_temp = []
    for i in range(len(lis_end)):
        temp = complex_noise_in_SWat(lis_end[i])
        lis_temp.append(temp)
    new_ts, false_label = concatenate_slices_with_false_labels(lis_start, lis_temp)
    return new_ts, false_label


def deform_SMAP_MSL(dataset, number):
    train_data = uni.load_data(dataset, number, "train")
    train_data0 = train_data[:, 0]
    if dataset == "SMAP":
        if number == "A-1":
            lis = generate_slices(train_data0, 0.05, 50)
            new_ts0, false_label = alter_ts(train_data0, lis, need_valley=True, valley_depth=-1)
        elif number == "A-2":
            lis = generate_slices(train_data0, 0.05, 50, border=500)
            new_ts0, false_label = alter_ts(train_data0, lis, need_uniform=True)
        elif number == "B-1":
            lis = generate_slices(train_data0, 0.05, 50, border=500)
            new_ts0, false_label = alter_ts(train_data0, lis, need_uniform=True, uniform_height=1)
        elif number == "D-1":
            lis = generate_slices(train_data0, 0.05, 50, border=500)
            new_ts0, false_label = alter_ts(train_data0, lis, need_uniform=True, uniform_height=1)
        elif number == "E-1":
            lis = generate_slices(train_data0, 0.05, 50, border=500)
            new_ts0, false_label = alter_ts(train_data0, lis, need_uniform=True, uniform_height=0.5)
        elif number == "F-1":
            lis = generate_slices(train_data0, 0.05, 10, border=500)
            new_ts0, false_label = alter_ts(train_data0, lis, need_valley=True, valley_depth=1)
        elif number == "G-1":
            lis = generate_slices(train_data0, 0.05, 50, border=500)
            new_ts0, false_label = alter_ts(train_data0, lis, need_uniform=True)
        elif number == "P-1":
            lis = generate_slices(train_data0, 0.05, 50, border=500)
            new_ts0, false_label = alter_ts(train_data0, lis, need_uniform=True)
        elif number == "R-1":
            lis = generate_slices(train_data0, 0.05, 50, border=500)
            new_ts0, false_label = alter_ts(train_data0, lis, need_valley=True, valley_depth=-1)
        elif number == "S-1":
            lis = generate_slices(train_data0, 0.05, 100, border=500)
            new_ts0, false_label = alter_ts(train_data0, lis, need_uniform=True, uniform_height=-1)
    elif dataset == "MSL":
        if number == "C-1":
            lis = generate_slices(train_data0, 0.05, 20, border=500)
            new_ts0, false_label = alter_ts(train_data0, lis, need_peak=True, peak_height=0.7)
        elif number == "D-14":
            lis = generate_slices(train_data0, 0.05, 20, border=500)
            new_ts0, false_label = alter_ts(train_data0, lis, need_peak=True, peak_height=0.7)
        elif number == "F-4":
            lis = generate_slices(train_data0, 0.05, 20, border=100)
            new_ts0, false_label = alter_ts(train_data0, lis, need_uniform=True)
        elif number == "M-1":
            lis = generate_slices(train_data0, 0.05, 20, border=100)
            new_ts0, false_label = alter_ts(train_data0, lis, need_transverse_stretching=True,
                                            transverse_stretching_ratio=0.5)
        elif number == "P-10":
            lis = generate_slices(train_data0, 0.05, 20, border=100)
            new_ts0, false_label = alter_ts(train_data0, lis, need_valley=True, valley_depth=1)
        elif number == "S-2":
            lis = generate_slices(train_data0, 0.05, 5, border=100)
            new_ts0, false_label = alter_ts(train_data0, lis, need_peak=True, peak_height=0.7)
        elif number == "T-5":
            lis = generate_slices(train_data0, 0.05, 5, border=100)
            new_ts0, false_label = alter_ts(train_data0, lis, need_peak=True, peak_height=0.7)

    train_data[:, 0] = new_ts0
    new_ts = train_data
    return new_ts, false_label

def deform(dataset, number):
    if dataset == "SMD":
        return deform_SMD(number)
    elif dataset == "PSM":
        return deform_PSM()
    elif dataset == "SMAP" or dataset == "MSL":
        return deform_SMAP_MSL(dataset, number)
    elif dataset == "SWat":
        return deform_SWat()



