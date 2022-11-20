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


def get_folds_data(num_people, feat_time, kfold_num, img_rows, img_cols):
    folds_data = {}
    folds_data_person = {}
    for nb in range(num_people):
        one_falx_1 = falx[nb * 3:nb * 3 + 3]
        one_falx_1 = one_falx_1.reshape((-1, feat_time, img_rows, img_cols, 5))

        # ###============= random select ============####
        # permutation = np.random.permutation(one_y_1.shape[0])
        # one_falx_2 = one_falx_1[permutation, :]
        # one_falx = one_falx_2[0:3400]
        # one_y_2 = one_y_1[permutation, :]
        # one_y = one_y_2[0:3400]
        # ###============= random select ============####

        one_y = one_y_1
        one_falx = one_falx_1[:, :, :, :, 1:5]

        # print(one_y.shape)
        # print(one_falx.shape)
        # x_train, x_test, y_train, y_test = train_test_split(one_falx, one_y, test_size=0.25)
        kfold = StratifiedKFold(n_splits=kfold_num, shuffle=True, random_state=seed)
        for fi, (train, test) in enumerate(kfold.split(one_falx, one_y.argmax(1))):
            # x_train = one_falx[train]
            # y_train = one_y[train]
            x_test = one_falx[test]
            y_test = one_y[test]

            if fi in folds_data:
                # folds_data[fi]['x_train'] = np.concatenate([folds_data[fi]['x_train'], x_train], axis=0)
                # folds_data[fi]['y_train'] = np.concatenate([folds_data[fi]['y_train'], y_train], axis=0)
                folds_data[fi]['x_test'] = np.concatenate([folds_data[fi]['x_test'], x_test], axis=0)
                folds_data[fi]['y_test'] = np.concatenate([folds_data[fi]['y_test'], y_test], axis=0)
            else:
                folds_data[fi] = {}
                # folds_data[fi]['x_train'] = x_train
                # folds_data[fi]['y_train'] = y_train
                folds_data[fi]['x_test'] = x_test
                folds_data[fi]['y_test'] = y_test

            # folds_data_person[fi][nb]['x_train'] = x_train
            # folds_data_person[fi][nb]['y_train'] = y_train
            folds_data_person[fi] = {nb: {}}
            folds_data_person[fi][nb]['x_test'] = x_test
            folds_data_person[fi][nb]['y_test'] = y_test

    return folds_data, folds_data_person


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
    output_path = './runtime_all/'
    num_people = 1
    num_classes = 3
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

    img_size = (img_rows, img_cols, num_chan)
    falx = np.load('%s/X_89_t%s.npy' % (data_path, feat_time))
    y = np.load('%s/y_89_t%s.npy' % (data_path, feat_time))

    one_y_1 = np.array([y[:get_ns(feat_time)]] * 3).reshape((-1,))
    one_y_1 = to_categorical(one_y_1, num_classes)

    people_acc_list = {}
    label_gold = []
    label_pred = []
    folds_data, folds_data_person = get_folds_data(num_people, feat_time, kfold_num, img_rows, img_cols)
    with codecs.open('%s/test.log' % runtime_path, 'w', 'utf-8') as fout:
        for fi in folds_data:
            start = time.time()
            K.clear_session()

            x_test = folds_data[fi]['x_test']
            y_test = folds_data[fi]['y_test']

            fold_path = '%s/f%s' % (runtime_path, fi)

            input_list = [Input(shape=img_size) for _ in range(feat_time)]
            if os.path.exists('%s/best.h5' % fold_path):
                model = keras.models.load_model('%s/best.h5' % fold_path)
            else:
                model = keras.models.load_model('%s/last.h5' % fold_path)

            # evaluate the model
            scores = model.evaluate(
                [x_test[:, nf] for nf in range(feat_time)],
                y_test, verbose=1
            )

            accuracy = (scores[1] * 100)
            print("%.2f%%" % accuracy)
            fout.write('fold acc： %s\n' % accuracy)

            # 画图
            y_pred = model([x_test[:, nf] for nf in range(feat_time)])
            y_pred = K.eval(tf.argmax(y_pred, axis=1, output_type=tf.int32)).tolist()
            y_true = np.argmax(y_test, axis=1).tolist()
            plot_confusion_matrix(fold_path, y_true, y_pred, labels=labels_name)
            label_gold.extend(y_true)
            label_pred.extend(y_pred)

            people_acc = {}
            for nb in folds_data_person[fi]:
                person_x = folds_data_person[fi][nb]['x_test']
                person_y = folds_data_person[fi][nb]['y_test']
                person_y_pred = model([person_x[:, nf] for nf in range(feat_time)])
                person_y_pred = K.eval(tf.argmax(person_y_pred, axis=1, output_type=tf.int32)).tolist()
                person_y_true = np.argmax(person_y, axis=1).tolist()
                person_acc = np.mean(np.equal(np.array(person_y_pred), np.array(person_y_true)))
                people_acc[nb] = person_acc

                if nb in people_acc_list:
                    people_acc_list[nb].append(person_acc)
                else:
                    people_acc_list[nb] = [person_acc]

            with codecs.open('%s/people_accuracy.log' % fold_path, 'w', 'utf-8') as fout_per:
                for nb in people_acc:
                    fout_per.write('%s: %s\n' % (nb, people_acc[nb]))

            # print("all acc: {}".format(all_acc))
            # print('mean acc: %s, std: %s' % (np.mean(all_acc), np.std(all_acc)))
            # fout.write('mean acc: %s, std: %s\n' % (np.mean(all_acc), np.std(all_acc)))
            # acc_list.append(np.mean(all_acc))
            # std_list.append(np.std(all_acc))
            # print("进度： {}".format(nb))
            # all_acc = []
            end = time.time()
            print("%.2f" % (end - start))  # run time

        plot_confusion_matrix(runtime_path, label_gold, label_pred, labels=labels_name)
        with codecs.open('%s/people_accuracy.log' % runtime_path, 'w', 'utf-8') as fout_peo:
            for nb in people_acc_list:
                fout_peo.write('%s: %s\n' % (nb, people_acc_list[nb]))
