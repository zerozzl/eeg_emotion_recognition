import tensorflow as tf
from keras import models, layers

from models.DenseNet import dense_block, transition_block


def se_block(x, reduce=16, name='se'):
    num_feature = x.get_shape().as_list()[3]
    sequeeze = layers.GlobalAveragePooling2D(name=name + '_sequeeze')(x)

    excitation = layers.Dense(num_feature // reduce, name=name + '_SE_fc1')(sequeeze)
    excitation = layers.Activation('relu', name=name + '_SE_relu')(excitation)
    excitation = layers.Dense(num_feature, name=name + '_SE_fc2')(excitation)
    excitation = layers.Activation('sigmoid', name=name + '_sigmoid')(excitation)
    excitation = layers.Reshape((1, 1, num_feature), name=name + '_reshape')(excitation)
    scale = tf.multiply(x, excitation)

    return scale


def res_block(x, layer_size, kernel_size, name, dropout_rate=None):
    x1 = layers.Conv2D(layer_size, kernel_size=kernel_size, padding='same', use_bias=True, name=name + '_2_conv')(x)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    if dropout_rate:
        x1 = layers.Dropout(dropout_rate, name=name + '_1_dropout')(x1)

    x = layers.Concatenate(name=name + '_concat')([x, x1])
    return x


def backbone_cnn(input_dim):
    seq = models.Sequential()
    seq.add(layers.Conv2D(64, 5, activation='relu', padding='same', name='conv1', input_shape=input_dim))
    seq.add(layers.Conv2D(128, 4, activation='relu', padding='same', name='conv2'))
    seq.add(layers.Conv2D(256, 4, activation='relu', padding='same', name='conv3'))
    seq.add(layers.Conv2D(64, 1, activation='relu', padding='same', name='conv4'))
    seq.add(layers.MaxPooling2D(2, 2, name='pool1'))
    seq.add(layers.Flatten(name='fla1'))
    seq.add(layers.Dense(512, activation='relu', name='dense1'))
    seq.add(layers.Reshape((1, 512), name='reshape'))
    return seq


def backbone_densenet(input_shape, blocks=[1, 1, 1, 1], growth_rate=32, reduction=0.5,
                      dropout_rate=None, se_head=False, se_tail=False):
    inputs = layers.Input(shape=input_shape)
    if se_head:
        x = se_block(inputs, reduce=2, name='se_head')
    else:
        x = inputs

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.Conv2D(64, 3, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)

    # x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    # x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], growth_rate=growth_rate, name='conv2', dropout_rate=dropout_rate)
    x = transition_block(x, reduction=reduction, name='pool2')
    x = dense_block(x, blocks[1], growth_rate=growth_rate, name='conv3', dropout_rate=dropout_rate)
    x = transition_block(x, reduction=reduction, name='pool3')
    x = dense_block(x, blocks[2], growth_rate=growth_rate, name='conv4', dropout_rate=dropout_rate)
    # x = transition_block(x, reduction=reduction, name='pool4')
    # x = dense_block(x, blocks[3], growth_rate=growth_rate, name='conv5', dropout_rate=dropout_rate)

    x = layers.BatchNormalization(epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if se_tail:
        x = se_block(x, name='se_tail')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Reshape((1, -1), name='reshape')(x)

    model = models.Model(inputs, x)
    return model


def backbone_resnet(input_shape, dropout_rate=None, se_head=False, se_tail=False):
    inputs = layers.Input(shape=input_shape)
    if se_head:
        x = se_block(inputs, reduce=2, name='se_head')
    else:
        x = inputs

    x = res_block(x, layer_size=16, kernel_size=5, name='conv1', dropout_rate=dropout_rate)
    x = res_block(x, layer_size=32, kernel_size=5, name='conv2', dropout_rate=dropout_rate)
    x = res_block(x, layer_size=256, kernel_size=3, name='conv3', dropout_rate=dropout_rate)
    x = layers.Conv2D(64, 1, activation='relu', padding='same', name='conv4')(x)

    if se_tail:
        x = se_block(x, reduce=16, name='se_tail')

    x = layers.Flatten(name='fla1')(x)
    x = layers.Dense(512, activation='relu', name='dense1')(x)
    x = layers.Reshape((1, 512), name='reshape')(x)

    model = models.Model(inputs, x)
    return model


def build_model(feat_time, input_dim, input_list, backbone_type='cnn', se_head=False, se_tail=False):
    if backbone_type == 'cnn':
        backbone = backbone_cnn(input_dim)
    elif backbone_type == 'densenet':
        backbone = backbone_densenet(input_dim, se_head=se_head, se_tail=se_tail)
    elif backbone_type == 'resnet':
        backbone = backbone_resnet(input_dim, se_head=se_head, se_tail=se_tail)

    if feat_time > 1:
        out_all = layers.Concatenate(axis=1)(
            [backbone(inp) for inp in input_list]
        )
        lstm_layer = layers.LSTM(128, name='lstm')(out_all)
        out_layer = layers.Dense(3, activation='softmax', name='out')(lstm_layer)
    else:
        out = [backbone(inp) for inp in input_list][0][:, 0, :]
        out_layer = layers.Dense(3, activation='softmax', name='out')(out)

    model = models.Model(input_list, out_layer)
    return model
