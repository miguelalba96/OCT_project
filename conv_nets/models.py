import os
import tempfile
import tensorflow as tf
import conv_nets.layers as layers


def sequential_model_1(crop_size, **kwargs):
    inputs = tf.keras.Input(shape=tuple(crop_size) + (3,), name='input')  # (128, 136, 136, 3)
    opts = {
        'kernel_initializer': tf.keras.initializers.VarianceScaling(mode='fan_in',
                                                                    distribution='truncated_normal'),
        'bias_initializer': tf.keras.initializers.Constant(0.1)
    }
    opts.update(**kwargs)
    x = inputs
    x = layers.conv_layer(x, 8, 7, batch_norm=True, scope='conv1', **opts)
    x = layers.conv_layer(x, 8, 3, batch_norm=True, scope='conv2', **opts)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(x)

    x = layers.conv_layer(x, 16, 3, batch_norm=True, scope='conv3', **opts)
    x = layers.conv_layer(x, 16, 3, batch_norm=True, scope='conv4', **opts)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)

    x = layers.conv_layer(x, 32, 3, batch_norm=True, scope='conv5', **opts)
    x = layers.conv_layer(x, 32, 3, batch_norm=True, scope='conv6', **opts)
    x = layers.conv_layer(x, 32, 3, batch_norm=True, scope='conv7', **opts)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool3')(x)

    x = layers.conv_layer(x, 64, 3, batch_norm=True, scope='conv8', **opts)
    x = layers.conv_layer(x, 64, 3, batch_norm=True, scope='conv9', **opts)
    x = layers.conv_layer(x, 96, 3, batch_norm=True, scope='conv10', **opts)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool4')(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = tf.keras.layers.Dense(64, name='fc1', activation='relu', **opts)(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)
    x = tf.keras.layers.Dense(128, name='fc2', activation='relu', **opts)(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_2')(x)
    prob = tf.keras.layers.Dense(4, activation='softmax', name='prob', **opts)(x)
    model = tf.keras.Model(inputs=inputs, outputs=prob, name='model1')
    return model


def dense_net_red(crop_size, **kwargs):
    inputs = tf.keras.Input(shape=tuple(crop_size) + (3,))
    opts = {
        'kernel_initializer': tf.keras.initializers.VarianceScaling(mode='fan_in',
                                                                    distribution='truncated_normal'),
        'bias_initializer': tf.keras.initializers.Constant(0.1)
    }
    opts.update(**kwargs)
    x = inputs
    x = layers.conv_layer(x, 8, 7, batch_norm=True, scope='conv1', **opts)
    x = tf.keras.layers.MaxPooling2D((3, 2), name='pool1')(x)

    growth_rate = 8
    num_filters = 16
    num_layers = [4, 6, 6, 8, 8]
    dropout_rate = 0.1
    compress = 0.5
    
    stage = 0
    for i in range(len(num_layers) - 1):
        stage = i + 2
        x, filters = layers.dense_block(x, stage, num_layers[i], num_filters,
                                        growth_rate, dropout_rate=dropout_rate, **opts)
        x = layers.dense_transition_block(x, stage, filters, compression=compress, dropout_rate=dropout_rate, **opts)
        num_filters = int(filters * compress)

    final_stage = stage + 1
    x, filters = layers.dense_block(x, final_stage, num_layers[-1], num_filters, growth_rate,
                                    dropout_rate=dropout_rate, **opts)
    x = tf.keras.layers.BatchNormalization(name='bn_final')(x)
    x = tf.keras.layers.ReLU(name='relu_final')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    prob = tf.keras.layers.Dense(4, activation='softmax', name='prob', **opts)(x)
    model = tf.keras.Model(inputs=inputs, outputs=prob, name='reduced_dense')
    return model


def get_base(weights='imagenet', input_shape=(68, 100, 3), kernel_reg=1e-4):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights=weights)
    regularizer = tf.keras.regularizers.l2(kernel_reg)
    for layer in base_model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    model_json = base_model.to_json()

    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    base_model.save_weights(tmp_weights_path)

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(tmp_weights_path, by_name=True)
    return model
