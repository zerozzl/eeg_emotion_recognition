import os
import time
import random
import codecs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import keras
from keras import backend as K
from keras.layers import Input
from keras.callbacks import Callback, TensorBoard
from sklearn.model_selection import StratifiedKFold

from model import build_model


class TrainHistory(Callback):

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))

    def save_history(self, output_path):
        with codecs.open('%s/loss.log' % output_path, 'w', 'utf-8') as fout:
            for loss in self.losses['epoch']:
                fout.write('%s\n' % loss)
        with codecs.open('%s/accuracy.log' % output_path, 'w', 'utf-8') as fout:
            for acc in self.accuracy['epoch']:
                fout.write('%s\n' % acc)

        self.plot('%s/loss.png' % output_path, self.losses['epoch'], 'epoch', 'loss')
        self.plot('%s/accuracy.png' % output_path, self.accuracy['epoch'], 'epoch', 'accuracy')

    def plot(self, plot_file, data, xlabel, ylabel):
        iters = range(len(data))
        plt.figure()
        plt.plot(iters, data, label=ylabel)
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(plot_file)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    data_path = './data/model/'
    output_path = './runtime/'
    num_classes = 3
    epoch_size = 100
    batch_size = 128
    img_rows, img_cols, num_chan = 8, 9, 4
    model_type = 'cnn'  # cnn, densenet, resnet
    features = [0, 1, 2, 3, 4, 5]
    se_head = False
    se_tail = False
    seed = 1

    set_seed(seed)

    runtime_path = '%s/%s' % (output_path, model_type)
    if se_head:
        runtime_path += '_se_head'
    if se_tail:
        runtime_path += '_se_tail'
    if not os.path.exists(runtime_path):
        os.makedirs(runtime_path)

    falx = np.load('%s/X_89_t6.npy' % data_path)
    y = np.load('%s/y_89_t6.npy' % data_path)

    one_y_1 = np.array([y[:1126]] * 3).reshape((-1,))
    one_y_1 = to_categorical(one_y_1, num_classes)

    acc_list = []
    std_list = []
    all_acc = []
    with codecs.open('%s/train.log' % runtime_path, 'w', 'utf-8') as fout:
        # for nb in range(15):
        for nb in range(1):
            fout.write('==================== %s ====================\n' % nb)
            model_save_path = '%s/%s/' % (runtime_path, nb)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            K.clear_session()
            start = time.time()
            one_falx_1 = falx[nb * 3:nb * 3 + 3]
            one_falx_1 = one_falx_1.reshape((-1, 6, img_rows, img_cols, 5))

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
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            cvscores = []

            best_metric = 0
            for fi, (train, test) in enumerate(kfold.split(one_falx, one_y.argmax(1))):
                fold_path = '%s/f%s' % (model_save_path, fi)
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)
                tensorboard = TensorBoard(log_dir=fold_path, histogram_freq=1, write_grads=True, write_graph=True,
                                          write_images=True, embeddings_freq=0, embeddings_layer_names=None)

                img_size = (img_rows, img_cols, num_chan)

                input_list = [Input(shape=img_size) for _ in range(len(features))]
                model = build_model(img_size, input_list, backbone_type=model_type,
                                    se_head=se_head, se_tail=se_tail)
                # model.summary()

                # Compile model
                model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.adam_v2.Adam(),
                              metrics=['accuracy'])
                # Fit the model
                x_train = one_falx[train]
                y_train = one_y[train]
                history = TrainHistory()

                model.fit(
                    [x_train[:, nf] for nf in features],
                    y_train, epochs=epoch_size, batch_size=batch_size,
                    callbacks=[history, tensorboard], verbose=1
                )

                # evaluate the model
                x_test = one_falx[test]
                y_test = one_y[test]
                scores = model.evaluate(
                    [x_test[:, nf] for nf in features],
                    y_test, verbose=1
                )

                accuracy = (scores[1] * 100)
                print("%.2f%%" % accuracy)
                fout.write('fold acc： %s\n' % accuracy)
                all_acc.append(accuracy)
                history.save_history(fold_path)

                # save model
                model.save('%s/last.h5' % model_save_path)
                if accuracy > best_metric:
                    best_metric = accuracy
                    model.save('%s/best.h5' % model_save_path)

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
