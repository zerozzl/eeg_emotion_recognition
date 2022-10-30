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
    model_type = 'densenet'  # cnn, densenet, resnet
    features = [0, 1, 2, 3, 4, 5]
    se_head = True
    se_tail = True
    seed = 1

    set_seed(seed)

    runtime_path = '%s/%s' % (output_path, model_type)
    if se_head:
        runtime_path += '_se_head'
    if se_tail:
        runtime_path += '_se_tail'

    falx = np.load('%s/X_89_t6.npy' % data_path)
    y = np.load('%s/y_89_t6.npy' % data_path)

    one_y_1 = np.array([y[:1126]] * 3).reshape((-1,))
    one_y_1 = to_categorical(one_y_1, num_classes)

    acc_list = []
    std_list = []
    all_acc = []
    with codecs.open('%s/test.log' % runtime_path, 'w', 'utf-8') as fout:
        # for nb in range(15):
        for nb in range(1):
            fout.write('==================== %s ====================\n' % nb)

            K.clear_session()
            start = time.time()
            one_falx_1 = falx[nb * 3:nb * 3 + 3]
            one_falx_1 = one_falx_1.reshape((-1, 6, img_rows, img_cols, 5))
            one_y = one_y_1
            one_falx = one_falx_1[:, :, :, :, 1:5]

            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            cvscores = []

            best_metric = 0
            for train, test in kfold.split(one_falx, one_y.argmax(1)):
                img_size = (img_rows, img_cols, num_chan)

                input_list = [Input(shape=img_size) for _ in range(len(features))]
                model = keras.models.load_model('%s/%s/best.h5' % (runtime_path, nb))

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
