##############
## SEED数据集提取每个通道5个频段的DE特征，
## 并将62个通道转化为8*9*4的三维输入，其中8*9表示62个通道转化后的二维平面，5表示5种频段
##############

import os
import math
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import loadmat


def decompose(file, name):
    # trial*channel*sample
    data = loadmat(file)
    frequency = 200

    decomposed_de = np.empty([0, 62, 5])
    label = np.array([])
    all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    for trial in range(15):
        # tmp_idx = trial + 3
        # tmp_name = list(data.keys())[tmp_idx]
        tmp_trial_signal = data[name + '_eeg' + str(trial + 1)]
        num_sample = int(len(tmp_trial_signal[0]) / 100)
        # print('{}-{}'.format(trial + 1, num_sample))

        temp_de = np.empty([0, num_sample])
        label = np.append(label, [all_label[trial]] * num_sample)

        for channel in range(62):
            trial_signal = tmp_trial_signal[channel]
            # 因为SEED数据没有基线信号部分

            delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 51, frequency, order=3)

            DE_delta = np.zeros(shape=[0], dtype=float)
            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)

            for index in range(num_sample):
                DE_delta = np.append(DE_delta, compute_DE(delta[index * 100:(index + 1) * 100]))
                DE_theta = np.append(DE_theta, compute_DE(theta[index * 100:(index + 1) * 100]))
                DE_alpha = np.append(DE_alpha, compute_DE(alpha[index * 100:(index + 1) * 100]))
                DE_beta = np.append(DE_beta, compute_DE(beta[index * 100:(index + 1) * 100]))
                DE_gamma = np.append(DE_gamma, compute_DE(gamma[index * 100:(index + 1) * 100]))
            temp_de = np.vstack([temp_de, DE_delta])
            temp_de = np.vstack([temp_de, DE_theta])
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])

        temp_trial_de = temp_de.reshape(-1, 5, num_sample)
        temp_trial_de = temp_trial_de.transpose([2, 0, 1])
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])

    print("trial_DE shape:", decomposed_de.shape)
    return decomposed_de, label


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def get_ns(t):
    if t == 1:
        ns = 6788
    elif t == 2:
        ns = 3394  # 10182
    elif t == 3:
        ns = 2257  # 6771
    elif t == 4:
        ns = 1692  # 5076
    elif t == 5:
        ns = 1354  # 4062
    elif t == 6:
        ns = 1126  # 3378
    elif t == 7:
        ns = 962  # 2886
    elif t == 8:
        ns = 842  # 2526
    elif t == 9:
        ns = 746  # 2238
    elif t == 10:
        ns = 675  # 2025
    elif t == 11:
        ns = 611  # 1833
    elif t == 12:
        ns = 559  # 1677
    return ns


def process_matlab_data(data_path, output_path, people_name, short_name):
    X = np.empty([0, 62, 5])
    y = np.empty([0, 1])

    for i in range(len(people_name)):
        file_name = data_path + people_name[i]
        print('processing %s: %s' % (people_name[i], file_name))
        decomposed_de, label = decompose(file_name, short_name[i])
        X = np.vstack([X, decomposed_de])
        y = np.append(y, label)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.save('%s/X_1D.npy' % output_path, X)
    np.save('%s/y.npy' % output_path, y)


def create_model_data(data_path, num_people, img_rows, img_cols, num_chan, t):
    X = np.load('%s/X_1D.npy' % data_path)
    y = np.load('%s/y.npy' % data_path)

    # 生成8*9的矩阵形式
    X89 = np.zeros((len(y), 8, 9, 5))
    X89[:, 0, 2, :] = X[:, 3, :]
    X89[:, 0, 3:6, :] = X[:, 0:3, :]
    X89[:, 0, 6, :] = X[:, 4, :]
    for i in range(5):
        X89[:, i + 1, :, :] = X[:, 5 + i * 9:5 + (i + 1) * 9, :]
    X89[:, 6, 1:8, :] = X[:, 50:57, :]
    X89[:, 7, 2:7, :] = X[:, 57:62, :]
    np.save('%s/X_89.npy' % data_path, X89)

    falx = np.load('%s/X_89.npy' % data_path)
    falx = falx.reshape((num_people, 6788, img_rows, img_cols, 5))
    ns = get_ns(t)

    new_x = np.zeros((num_people, ns, t, img_rows, img_cols, 5))
    new_y = np.array([])

    for nb in range(num_people):
        z = 0
        i = 0
        while i + t <= 470:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, 1)
            i = i + t
            z = z + 1
        i = 470
        while i + t <= 936:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, 0)
            i = i + t
            z = z + 1
        i = 936
        while i + t <= 1348:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, -1)
            i = i + t
            z = z + 1
        i = 1348
        while i + t <= 1824:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, -1)
            i = i + t
            z = z + 1
        i = 1824
        while i + t <= 2194:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, 0)
            i = i + t
            z = z + 1
        i = 2194
        while i + t <= 2584:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, 1)
            i = i + t
            z = z + 1
        i = 2584
        while i + t <= 3058:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, -1)
            i = i + t
            z = z + 1
        i = 3058
        while i + t <= 3490:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, 0)
            i = i + t
            z = z + 1
        i = 3490
        while i + t <= 4020:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, 1)
            i = i + t
            z = z + 1
        i = 4020
        while i + t <= 4494:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, 1)
            i = i + t
            z = z + 1
        i = 4494
        while i + t <= 4964:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, 0)
            i = i + t
            z = z + 1
        i = 4964
        while i + t <= 5430:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, -1)
            i = i + t
            z = z + 1
        i = 5430
        while i + t <= 5900:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, 0)
            i = i + t
            z = z + 1
        i = 5900
        while i + t <= 6376:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, 1)
            i = i + t
            z = z + 1
        i = 6376
        while i + t <= 6788:
            new_x[nb, z] = falx[nb, i:i + t]
            new_y = np.append(new_y, -1)
            i = i + t
            z = z + 1
        # print('{}-{}'.format(nb, z))

    np.save('%s/X_89_t%s.npy' % (data_path, t), new_x)
    np.save('%s/y_89_t%s.npy' % (data_path, t), new_y)


# 究极整合版
if __name__ == '__main__':
    data_path = './data/eeg/'  # 输入目录
    output_path = './data/model/'  # 输出目录
    img_rows, img_cols, num_chan = 8, 9, 5
    t = 1

    # people_name = ['1_20131027', '1_20131030', '1_20131107',
    #                 '6_20130712', '6_20131016', '6_20131113',
    #                 '7_20131027', '7_20131030', '7_20131106',
    #                 '15_20130709', '15_20131016', '15_20131105',
    #                 '12_20131127', '12_20131201', '12_20131207',
    #                 '10_20131130', '10_20131204', '10_20131211',
    #                 '2_20140404', '2_20140413', '2_20140419',
    #                 '5_20140411', '5_20140418', '5_20140506',
    #                 '8_20140511', '8_20140514', '8_20140521',
    #                 '13_20140527', '13_20140603', '13_20140610',
    #                 '3_20140603', '3_20140611', '3_20140629',
    #                 '14_20140601', '14_20140615', '14_20140627',
    #                 '11_20140618', '11_20140625', '11_20140630',
    #                 '9_20140620', '9_20140627', '9_20140704',
    #                 '4_20140621', '4_20140702', '4_20140705']
    # short_name = ['djc', 'djc', 'djc', 'mhw', 'mhw', 'mhw', 'phl', 'phl', 'phl',
    #                'zjy', 'zjy', 'zjy', 'wyw', 'wyw', 'wyw', 'ww', 'ww', 'ww',
    #                'jl', 'jl', 'jl', 'ly', 'ly', 'ly', 'sxy', 'sxy', 'sxy',
    #                'xyl', 'xyl', 'xyl', 'jj', 'jj', 'jj', 'ys', 'ys', 'ys',
    #                'wsf', 'wsf', 'wsf', 'wk', 'wk', 'wk', 'lqj', 'lqj', 'lqj']

    people_name = ['1_20131027', '1_20131030', '1_20131107']
    short_name = ['djc', 'djc', 'djc']

    # 转换matlab数据
    process_matlab_data(data_path, output_path, people_name, short_name)

    # 生成模型数据
    create_model_data(output_path, len(people_name), img_rows, img_cols, num_chan, t)
    print('complete')
