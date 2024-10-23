import os
import pandas as pd
import numpy as np
from pickle import dump

datasets = ['SMD', 'SWat', 'SMAP', 'MSL', 'PSM']
datasets = ['SWat']

# preprocess functions are mainly from tranAD
def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    """
    读取txt文本数据，转为numpy数组，然后将numpy数组保存在output_folder/dataset_category.pkl中
    """
    #   从文件中读取数据转为numpy数组  dataset_folder/category/filename
    #   ServerMachineDataset/test_label/filename
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    #   将Numpy数组保存为pickle文件
    with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)

def load_and_save2(category, filename, dataset, dataset_folder, shape, output_folder):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(':')[0], line.split(':')[1].split(',')
        start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
        temp[start-1:end-1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"{dataset}_{category}.npy"), temp[:, 0])


# 标准化方法取自tranAD
def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return (a / 2 + 0.5)


def normalize2(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a


def normalize3(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


def preprocess_and_save(dataset):
    script_dir = "dataset"
    processed_dataset = "processed dataset"
    output_folder = os.path.join(processed_dataset, dataset)
    os.makedirs(output_folder, exist_ok=True)
    if dataset == 'SMD':
        dataset_folder = os.path.join(script_dir, dataset)
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                # 将txt文件转为pkl文件
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder, output_folder)
                load_and_save('test', filename, filename.strip('.txt'), dataset_folder, output_folder)
                load_and_save('test_label', filename, filename.strip('.txt'), dataset_folder, output_folder)

    elif dataset == 'PSM':
        dataset_folder = os.path.join(script_dir, "PSM")

        train_data = pd.read_csv(f"{dataset_folder}\\train.csv").values[:, 1:]
        train_data = np.nan_to_num(train_data)
        test_data = pd.read_csv(f"{dataset_folder}\\test.csv").values[:, 1:]
        test_data = np.nan_to_num(test_data)
        labels = pd.read_csv(f"{dataset_folder}\\test_label.csv").values[:, 1:]
        # 标准化
        train, min_a, max_a = normalize3(train_data)
        test_data, _, _ = normalize3(test_data, min_a, max_a)
        np.save(f"{output_folder}\\train.npy", train_data)
        np.save(f"{output_folder}\\test.npy", test_data)
        np.save(f"{output_folder}\\labels.npy", labels)
    elif dataset == "SWat":
        dataset_folder = os.path.join(script_dir, "SWat")
        # follow the preprocessing in GDN down sample by 10
        GDN_preprocess = True
        dtype_dict = {column: str for column in range(1, 54)}
        train_data = pd.read_csv(f"{dataset_folder}\\SWat_Dataset_Normal_v1.csv", dtype=dtype_dict)
        del train_data["Timestamp"]
        del train_data["Normal/Attack"]
        test_data = pd.read_csv(f"{dataset_folder}\\SWat_Dataset_Attack_v0.csv", delimiter=';', dtype=dtype_dict)
        del test_data["Timestamp"]
        # labels
        labels = pd.Series([float(label != 'Normal') for label in test_data["Normal/Attack"].values])
        if GDN_preprocess:
            test_data["Normal/Attack"] = labels
            # train
            for i in list(train_data):
                train_data[i] = train_data[i].apply(lambda x: str(x).replace(",", "."))
            train_data = train_data.astype(float)
            train_data = train_data.iloc[:, 1:].rolling(10).mean()
            train_data = train_data[(train_data.index+1) % 10 == 0]
            # test
            for i in list(test_data):
                test_data[i] = test_data[i].apply(lambda x: str(x).replace(",", "."))
            test_data = test_data.astype(float)
            test_data = test_data.iloc[:, 1:].rolling(10).mean()
            test_data = test_data[(test_data.index + 1) % 10 == 0]

            f = lambda s: 1 if s > 0.5 else 0
            swat_labels = test_data['Normal/Attack'].apply(f)
            del test_data['Normal/Attack']

            train_data.columns = [None] * len(train_data.columns)
            test_data.columns = [None] * len(test_data.columns)
            train, min_a, max_a = normalize3(train_data)
            test, _, _ = normalize3(test_data, min_a, max_a)

            np.save(f"{output_folder}\\train.npy", np.asarray(train))
            np.save(f"{output_folder}\\test.npy", np.asarray(test))
            np.save(f"{output_folder}\\labels.npy", np.asarray(swat_labels))
        else:
            # labels
            del test_data["Normal/Attack"]
            np.save(f"{output_folder}\\labels.npy", np.asarray(labels))
            # train
            for i in list(train_data):
                train_data[i] = train_data[i].apply(lambda x: str(x).replace(",", "."))
            train_data.columns = [None] * len(train_data.columns)
            train_data = train_data.astype(float)
            train, min_a, max_a = normalize3(train_data)
            np.save(f"{output_folder}\\train.npy", np.asarray(train))

            # test
            for i in list(test_data):
                test_data[i] = test_data[i].apply(lambda x: str(x).replace(",", "."))
            test_data.columns = [None] * len(test_data.columns)
            test_data = test_data.astype(float)
            test, _, _ = normalize3(test_data, min_a, max_a)
            np.save(f"{output_folder}\\test.npy", np.asarray(test))

    elif dataset in ['SMAP', 'MSL']:
        file = os.path.join(script_dir, 'SMAPMSL', 'labeled_anomalies.csv')
        values = pd.read_csv(file)
        values = values[values['spacecraft'] == dataset]
        filenames = values['chan_id'].values.tolist()
        dataset_folder2 = os.path.join(script_dir, 'SMAPMSL')
        for fn in filenames:
            train = np.load(f'{dataset_folder2}\\train\\{fn}.npy')
            test = np.load(f'{dataset_folder2}\\test\\{fn}.npy')
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test, min_a, max_a)
            np.save(f'{output_folder}\\{fn}_train.npy', train)
            np.save(f'{output_folder}\\{fn}_test.npy', test)
            labels = np.zeros(test.shape[0])
            indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
            indices = indices.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i + 1]] = 1
            np.save(f'{output_folder}\\{fn}_labels.npy', labels)


if __name__ == "__main__":
    for item in datasets:
        print(f"Preprocessing dataset {item}")
        preprocess_and_save(item)
