import copy
import tensorflow as tf
import tensorflow.keras.backend as K


def dense_conv_block(x, stage, branch, filters, dropout_rate=None, **opts):
    inter_feat_maps = filters * 2
    blockname = 'dense_block_{}_{}'.format(stage, branch)
    x = tf.keras.layers.BatchNormalization(name=blockname + '/bn1')(x)
    x = tf.keras.layers.ReLU(name=blockname + '/relu1')(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_rate, name=blockname + '/dp1')(x) if dropout_rate else x
    x = conv_layer(x, num_filters=inter_feat_maps, kernel_size=1, padding='valid', act_type=None,
                   scope=blockname + '/squeeze1x1', use_bias=False, **opts)
    x = tf.keras.layers.BatchNormalization(name=blockname + '/bn2')(x)
    x = tf.keras.layers.ReLU(name=blockname + '/relu2')(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_rate, name=blockname + '/dp2')(x) if dropout_rate else x
    x = conv_layer(x, num_filters=inter_feat_maps, kernel_size=3, padding='same', act_type=None,
                   scope=blockname + '/expand3x3', use_bias=False, **opts)
    return x


def dense_block(x, stage, num_layers, filters, growth_rate, dropout_rate=None, growth_filters=True, **opts):
    list_features = [x]
    for i in range(num_layers):
        branch = i + 1
        x = dense_conv_block(x, stage, branch, growth_rate, dropout_rate, **opts)
        x = squeeze_excitation(x, scope='squeeze_excit_{}_{}'.format(stage, branch))
        list_features.append(x)
        x = tf.keras.layers.Concatenate(name='concat_{}_branch_{}'.format(stage, branch),
                                        axis=-1)(copy.copy(list_features))
        if growth_filters:
            filters += growth_rate
    return x, filters


def dense_transition_block(x, stage, filters, compression=1.0, dropout_rate=None, **opts):
    blockname = 'trans_block_{}'.format(stage)
    x = tf.keras.layers.BatchNormalization(name=blockname + '/bn')(x)
    x = tf.keras.layers.ReLU(name=blockname + '/relu')(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_rate, name=blockname + '/dp1')(x) if dropout_rate else x
    x = conv_layer(x, num_filters=int(filters * compression), kernel_size=1, padding='valid', act_type=None,
                   scope=blockname + '/squeeze1x1', use_bias=False, **opts)
    x = tf.keras.layers.MaxPooling2D((2, 2), name=blockname + '/pool')(x)
    return x


def conv_layer(x, num_filters, kernel_size, padding='same', strides=1, act_type='relu',
               batch_norm=False, scope='conv', **opts):
    x = tf.keras.layers.Conv2D(num_filters,
                               kernel_size=kernel_size,
                               strides=strides, name=scope, padding=padding, **opts)(x)
    x = tf.keras.layers.BatchNormalization(name=scope + '/bn')(x) if batch_norm else x
    if act_type == 'relu':
        x = tf.keras.layers.ReLU(name=scope + '/relu')(x)
    return x


def squeeze_excitation(x, ratio=16, scope='se', **opts):
    """
    squeeze-excitation
    https://arxiv.org/abs/1709.01507
    """
    opts = dict(opts, kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    filters = x.get_shape()[-1]  # num channels (B, H, W, C)
    blockname = scope + '/squeeze_excitation'
    se = tf.keras.layers.GlobalAveragePooling2D(name=blockname + '/gap')(x)
    se = tf.keras.layers.Reshape((1, 1, filters), name=blockname + '/reshape1')(se)
    se = tf.keras.layers.Dense(filters // ratio, use_bias=False, name=blockname + '/fc1', **opts)(se)
    se = tf.keras.layers.ReLU(name=blockname + '/relu1')(se)
    se = tf.keras.layers.Dense(filters, use_bias=False, name=blockname + '/fc2', **opts)(se)
    se = tf.keras.layers.Activation('sigmoid', name=blockname + '/sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, filters), name=blockname + '/reshape2')(se)
    x = tf.keras.layers.Multiply(name=blockname + '/mult')([x, se])
    return x
