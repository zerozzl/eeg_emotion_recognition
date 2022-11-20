import os
import time
import copy
import random
import codecs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import keras
from keras import backend as K
from keras.layers import Input
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from sklearn.model_selection import StratifiedKFold

from create_data import get_ns
from model import build_model


class RunHistory(Callback):

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_losses = {'batch': [], 'epoch': []}
        self.val_accuracy = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_losses['batch'].append(logs.get('val_loss'))
        self.val_accuracy['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_losses['epoch'].append(logs.get('val_loss'))
        self.val_accuracy['epoch'].append(logs.get('val_accuracy'))

    def save_history(self, output_path):
        with codecs.open('%s/train_loss.log' % output_path, 'w', 'utf-8') as fout:
            for loss in self.losses['epoch']:
                fout.write('%s\n' % loss)
        with codecs.open('%s/train_accuracy.log' % output_path, 'w', 'utf-8') as fout:
            for acc in self.accuracy['epoch']:
                fout.write('%s\n' % acc)
        self.plot('%s/train_loss.png' % output_path, self.losses['epoch'], 'epoch', 'loss', color='red')
        self.plot('%s/train_accuracy.png' % output_path, self.accuracy['epoch'], 'epoch', 'accuracy', color='blue')
        self.plot_twinx('%s/train_loss_accuracy.png' % output_path,
                        [self.losses['epoch'], self.accuracy['epoch']], 'epoch', ['loss', 'accuracy'])

        with codecs.open('%s/val_loss.log' % output_path, 'w', 'utf-8') as fout:
            for loss in self.val_losses['epoch']:
                if loss is not None:
                    fout.write('%s\n' % loss)
        with codecs.open('%s/val_accuracy.log' % output_path, 'w', 'utf-8') as fout:
            for acc in self.val_accuracy['epoch']:
                if acc is not None:
                    fout.write('%s\n' % acc)
        self.plot('%s/val_loss.png' % output_path, self.val_losses['epoch'], 'epoch', 'loss', color='red')
        self.plot('%s/val_accuracy.png' % output_path, self.val_accuracy['epoch'], 'epoch', 'accuracy', color='blue')
        self.plot_twinx('%s/val_loss_accuracy.png' % output_path,
                        [self.val_losses['epoch'], self.val_accuracy['epoch']], 'epoch', ['loss', 'accuracy'])

    def plot(self, plot_file, data, xlabel, ylabel, color='blue'):
        iters = range(len(data))
        plt.figure()
        plt.plot(iters, data, label=ylabel, color=color)
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(plot_file)
        plt.close('all')

    def plot_twinx(self, plot_file, data, xlabel, ylabel):
        iters = range(len(data[0]))
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(iters, data[0], color='red')
        ax2.plot(iters, data[1], color='blue')

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel[0])
        ax2.set_ylabel(ylabel[1])

        ax1.legend([ylabel[0]], bbox_to_anchor=(1.0, 0.3))
        ax2.legend([ylabel[1]], bbox_to_anchor=(1.0, 0.2))

        plt.grid(True)
        plt.savefig(plot_file)
        plt.close('all')


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot(plot_file, data, xlabel, ylabel, color='blue'):
    iters = range(len(data))
    plt.figure()
    plt.plot(iters, data, label=ylabel, color=color)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(plot_file)
    plt.close('all')


def plot_folds(plot_file, data, xlabel, ylabel,
               colors=['#223cce', '#340768', '#014e40', '#d22325', '#8e1e02']):
    iters = range(len(data[list(data.keys())[0]]))
    plt.figure()

    for idx, (fi, value) in enumerate(data.items()):
        plt.plot(iters, value, label='%s' % fi, color=colors[idx])

    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(0.9, 0.4))
    plt.savefig(plot_file)
    plt.close('all')


def plot_history(runtime_path):
    train_accuracy_list = {}
    val_accuracy_list = {}
    train_loss_list = {}
    val_loss_list = {}

    train_accuracy_mean = []
    val_accuracy_mean = []
    train_loss_mean = []
    val_loss_mean = []

    folds = []
    sub_fols = os.listdir(runtime_path)
    for sub_fol in sub_fols:
        if os.path.isdir('%s/%s' % (runtime_path, sub_fol)):
            folds.append(sub_fol)

    for fi in folds:
        ft_acc = []
        fv_acc = []
        ft_loss = []
        fv_loss = []

        with codecs.open('%s/%s/train_accuracy.log' % (runtime_path, fi), 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                ft_acc.append(float(line))
        train_accuracy_list[fi] = ft_acc
        if len(train_accuracy_mean) == 0:
            train_accuracy_mean = copy.deepcopy(ft_acc)
        else:
            for idx in range(len(ft_acc)):
                train_accuracy_mean[idx] = train_accuracy_mean[idx] + ft_acc[idx]

        with codecs.open('%s/%s/val_accuracy.log' % (runtime_path, fi), 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                fv_acc.append(float(line))
        val_accuracy_list[fi] = fv_acc
        if len(val_accuracy_mean) == 0:
            val_accuracy_mean = copy.deepcopy(fv_acc)
        else:
            for idx in range(len(fv_acc)):
                val_accuracy_mean[idx] = val_accuracy_mean[idx] + fv_acc[idx]

        with codecs.open('%s/%s/train_loss.log' % (runtime_path, fi), 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                ft_loss.append(float(line))
        train_loss_list[fi] = ft_loss
        if len(train_loss_mean) == 0:
            train_loss_mean = copy.deepcopy(ft_loss)
        else:
            for idx in range(len(ft_loss)):
                train_loss_mean[idx] = train_loss_mean[idx] + ft_loss[idx]

        with codecs.open('%s/%s/val_loss.log' % (runtime_path, fi), 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                fv_loss.append(float(line))
        val_loss_list[fi] = fv_loss
        if len(val_loss_mean) == 0:
            val_loss_mean = copy.deepcopy(fv_loss)
        else:
            for idx in range(len(fv_loss)):
                val_loss_mean[idx] = val_loss_mean[idx] + fv_loss[idx]

    train_accuracy_mean = [val / len(train_accuracy_list) for val in train_accuracy_mean]
    val_accuracy_mean = [val / len(val_accuracy_list) for val in val_accuracy_mean]
    train_loss_mean = [val / len(train_loss_list) for val in train_loss_mean]
    val_loss_mean = [val / len(val_loss_list) for val in val_loss_mean]

    plot_folds('%s/train_accuracy.png' % runtime_path, train_accuracy_list, 'epoch', 'accuracy')
    plot_folds('%s/val_accuracy.png' % runtime_path, val_accuracy_list, 'epoch', 'accuracy')
    plot_folds('%s/train_loss.png' % runtime_path, train_loss_list, 'epoch', 'loss')
    plot_folds('%s/val_loss.png' % runtime_path, val_loss_list, 'epoch', 'loss')
    plot('%s/train_accuracy_mean.png' % runtime_path, train_accuracy_mean, 'epoch', 'accuracy', color='blue')
    plot('%s/val_accuracy_mean.png' % runtime_path, val_accuracy_mean, 'epoch', 'accuracy', color='blue')
    plot('%s/train_loss_mean.png' % runtime_path, train_loss_mean, 'epoch', 'loss', color='red')
    plot('%s/val_loss_mean.png' % runtime_path, val_loss_mean, 'epoch', 'loss', color='red')

    return train_accuracy_mean, val_accuracy_mean, train_loss_mean, val_loss_mean


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    data_path = './data/model/'
    output_path = './runtime/'
    num_people = 1
    num_classes = 3
    epoch_size = 3
    batch_size = 128
    valid_split = 0
    kfold_num = 5
    img_rows, img_cols, num_chan = 8, 9, 4
    model_type = 'cnn'  # cnn, densenet, resnet
    feat_time = 1
    se_head = False
    se_tail = False
    seed = 1

    set_seed(seed)

    runtime_path = '%s/t%s/%s' % (output_path, feat_time, model_type)
    if se_head:
        runtime_path += '_se_head'
    if se_tail:
        runtime_path += '_se_tail'
    if not os.path.exists(runtime_path):
        os.makedirs(runtime_path)

    falx = np.load('%s/X_89_t%s.npy' % (data_path, feat_time))
    y = np.load('%s/y_89_t%s.npy' % (data_path, feat_time))

    one_y_1 = np.array([y[:get_ns(feat_time)]] * 3).reshape((-1,))
    one_y_1 = to_categorical(one_y_1, num_classes)

    acc_list = []
    std_list = []
    all_acc = []
    train_accuracy_mean_list = []
    val_accuracy_mean_list = []
    train_loss_mean_list = []
    val_loss_mean_list = []
    with codecs.open('%s/train.log' % runtime_path, 'w', 'utf-8') as fout:
        for nb in range(num_people):
            fout.write('==================== %s ====================\n' % nb)
            model_save_path = '%s/%s/' % (runtime_path, nb)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            K.clear_session()
            start = time.time()
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
            cvscores = []

            best_metric = 0
            for fi, (train, test) in enumerate(kfold.split(one_falx, one_y.argmax(1))):
                fold_path = '%s/f%s' % (model_save_path, fi)
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)
                tensorboard = TensorBoard(log_dir=fold_path, histogram_freq=1, write_grads=True, write_graph=True,
                                          write_images=True, embeddings_freq=0, embeddings_layer_names=None)

                img_size = (img_rows, img_cols, num_chan)
                input_list = [Input(shape=img_size) for _ in range(feat_time)]
                model = build_model(feat_time, img_size, input_list, backbone_type=model_type,
                                    se_head=se_head, se_tail=se_tail)
                # model.summary()

                # Compile model
                model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.adam_v2.Adam(),
                              metrics=['accuracy'])
                # Fit the model
                x_train = one_falx[train]
                y_train = one_y[train]
                train_history = RunHistory()
                checkpoint = ModelCheckpoint(filepath='%s/best.h5' % fold_path, monitor='val_accuracy', mode='max',
                                             save_best_only=True, verbose=1)

                model.fit(
                    [x_train[:, nf] for nf in range(feat_time)],
                    y_train, epochs=epoch_size, batch_size=batch_size, validation_split=valid_split,
                    callbacks=[checkpoint, train_history, tensorboard], verbose=1
                )
                model.save('%s/last.h5' % fold_path)

                # load best model
                if os.path.exists('%s/best.h5' % fold_path):
                    model = keras.models.load_model('%s/best.h5' % fold_path)

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
                train_history.save_history(fold_path)

                # save model
                if accuracy > best_metric:
                    best_metric = accuracy
                    model.save('%s/best.h5' % model_save_path)

            train_accuracy_mean, val_accuracy_mean, train_loss_mean, val_loss_mean = plot_history(model_save_path)
            train_accuracy_mean_list.append(train_accuracy_mean)
            val_accuracy_mean_list.append(val_accuracy_mean)
            train_loss_mean_list.append(train_loss_mean)
            val_loss_mean_list.append(val_loss_mean)

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

        train_accuracy_mean_sum = [0] * len(train_accuracy_mean_list[0])
        val_accuracy_mean_sum = [0] * len(val_accuracy_mean_list[0])
        train_loss_mean_sum = [0] * len(train_loss_mean_list[0])
        val_loss_mean_sum = [0] * len(val_loss_mean_list[0])

        for train_accuracy_mean in train_accuracy_mean_list:
            for idx in range(len(train_accuracy_mean)):
                train_accuracy_mean_sum[idx] += train_accuracy_mean[idx]
        for val_accuracy_mean in val_accuracy_mean_list:
            for idx in range(len(val_accuracy_mean)):
                val_accuracy_mean_sum[idx] += val_accuracy_mean[idx]
        for train_loss_mean in train_loss_mean_list:
            for idx in range(len(train_loss_mean)):
                train_loss_mean_sum[idx] += train_loss_mean[idx]
        for val_loss_mean in val_loss_mean_list:
            for idx in range(len(val_loss_mean)):
                val_loss_mean_sum[idx] += val_loss_mean[idx]

        plot('%s/train_accuracy_mean.png' % runtime_path, train_accuracy_mean, 'epoch', 'accuracy', color='blue')
        plot('%s/val_accuracy_mean.png' % runtime_path, val_accuracy_mean, 'epoch', 'accuracy', color='blue')
        plot('%s/train_loss_mean.png' % runtime_path, train_loss_mean, 'epoch', 'loss', color='red')
        plot('%s/val_loss_mean.png' % runtime_path, val_loss_mean, 'epoch', 'loss', color='red')
