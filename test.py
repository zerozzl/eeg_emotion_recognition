import os
import time
import random
import codecs
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import keras
from keras import backend as K
from keras.layers import Input
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from create_data import get_ns


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_confusion_matrix(output_path, y_true, y_pred, labels=["0", "1", "-1"]):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = cm_norm * 100
    cm_norm = np.around(cm_norm, decimals=2)  # 保留两位小数

    plt.matshow(cm_norm, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    # plt.colorbar()

    for i in range(len(cm_norm)):
        for j in range(len(cm_norm)):
            plt.annotate('%s' % cm_norm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    font_label = FontProperties(fname='./data/SimHei.ttf', size=13)
    plt.ylabel('实际类别', fontproperties=font_label)
    plt.xlabel('预测类别', fontproperties=font_label)

    font_ticks = FontProperties(fname='./data/SimHei.ttf', size=11)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, fontproperties=font_ticks)
    plt.yticks(xlocations, labels, fontproperties=font_ticks)

    # plt.show()
    plt.savefig('%s/confusion_matrix.png' % output_path)
    plt.close('all')
    print('=================================================================')
    print('=================================================================')
    print('=================================================================')
    print('=================================================================')


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    data_path = './data/model/'
    output_path = './runtime/'
    num_people = 1
    num_classes = 3
    batch_size = 128
    kfold_num = 5
    img_rows, img_cols, num_chan = 8, 9, 4
    model_type = 'cnn'  # cnn, densenet, resnet
    feat_time = 1
    se_head = False
    se_tail = False
    seed = 1
    labels_name = ['中性', '积极', '消极']  # 0, 1, -1

    set_seed(seed)

    runtime_path = '%s/t%s/%s' % (output_path, feat_time, model_type)
    if se_head:
        runtime_path += '_se_head'
    if se_tail:
        runtime_path += '_se_tail'

    falx = np.load('%s/X_89_t%s.npy' % (data_path, feat_time))
    y = np.load('%s/y_89_t%s.npy' % (data_path, feat_time))

    one_y_1 = np.array([y[:get_ns(feat_time)]] * 3).reshape((-1,))
    one_y_1 = to_categorical(one_y_1, num_classes)

    acc_list = []
    std_list = []
    all_acc = []
    label_gold_list = []
    label_pred_list = []
    with codecs.open('%s/test.log' % runtime_path, 'w', 'utf-8') as fout:
        for nb in range(num_people):
            fout.write('==================== %s ====================\n' % nb)

            K.clear_session()
            start = time.time()
            one_falx_1 = falx[nb * 3:nb * 3 + 3]
            one_falx_1 = one_falx_1.reshape((-1, feat_time, img_rows, img_cols, 5))
            one_y = one_y_1
            one_falx = one_falx_1[:, :, :, :, 1:5]

            kfold = StratifiedKFold(n_splits=kfold_num, shuffle=True, random_state=seed)
            cvscores = []

            lowest_metric = 100
            label_gold = []
            label_pred = []
            for fi, (train, test) in enumerate(kfold.split(one_falx, one_y.argmax(1))):
                fold_path = '%s/%s/f%s' % (runtime_path, nb, fi)
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)

                img_size = (img_rows, img_cols, num_chan)
                model = keras.models.load_model('%s/%s/best.h5' % (runtime_path, nb))

                # evaluate the model
                x_test = one_falx[test]
                y_test = one_y[test]
                scores = model.evaluate(
                    [x_test[:, nf] for nf in range(feat_time)],
                    y_test, verbose=1
                )

                accuracy = (scores[1] * 100)
                print("%.2f%%" % accuracy)
                fout.write('fold acc： %s\n' % accuracy)
                all_acc.append(accuracy)

                # 画图
                y_pred = model([x_test[:, nf] for nf in range(feat_time)])
                y_pred = K.eval(tf.argmax(y_pred, axis=1, output_type=tf.int32)).tolist()
                y_true = np.argmax(y_test, axis=1).tolist()
                plot_confusion_matrix(fold_path, y_true, y_pred, labels=labels_name)

                if accuracy <= lowest_metric:
                    lowest_metric = accuracy
                    label_gold = y_true
                    label_pred = y_pred

            label_gold_list.extend(label_gold)
            label_pred_list.extend(label_pred)

            # print("all acc: {}".format(all_acc))
            print('mean acc: %s, std: %s' % (np.mean(all_acc), np.std(all_acc)))
            fout.write('mean acc: %s, std: %s\n' % (np.mean(all_acc), np.std(all_acc)))
            acc_list.append(np.mean(all_acc))
            std_list.append(np.std(all_acc))
            print("进度： {}".format(nb))
            all_acc = []
            end = time.time()
            print("%.2f" % (end - start))  # run time

        print('all acc: %s' % acc_list)
        print('all std: %s' % std_list)
        print('mean acc: %s, std: %s' % (np.mean(acc_list), np.std(std_list)))

        fout.write('==================== ALL ====================\n')
        fout.write('all acc: %s\n' % acc_list)
        fout.write('all std: %s\n' % std_list)
        fout.write('mean acc: %s, std: %s\n' % (np.mean(acc_list), np.std(std_list)))

        plot_confusion_matrix(runtime_path, label_gold_list, label_pred_list, labels=labels_name)
