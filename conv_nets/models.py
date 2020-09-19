import os
import tempfile
import tensorflow as tf
import conv_nets.layers as layers


class LAEyeDeepNET(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super(LAEyeDeepNET, self).__init__(name=name)
        self.conv1 = layers.Conv2D(8, kernel_size=3, name='conv1', batch_norm=True, **kwargs)
        self.conv2 = layers.Conv2D(8, kernel_size=3, name='conv2', batch_norm=True, **kwargs)
        self.maxpool1 = layers.Pooling(type_pool='max', name='pool1')
        self.conv3 = layers.Conv2D(16, kernel_size=3, name='conv3', batch_norm=True, **kwargs)
        self.conv4 = layers.Conv2D(16, kernel_size=3, name='conv4', batch_norm=True, **kwargs)
        self.maxpool2 = layers.Pooling(type_pool='max', name='pool2')
        self.conv5 = layers.Conv2D(32, kernel_size=3, name='conv5', batch_norm=True, **kwargs)
        self.conv6 = layers.Conv2D(32, kernel_size=3, name='conv6', batch_norm=True, **kwargs)
        self.conv65 = layers.Conv2D(32, kernel_size=3, name='conv65', batch_norm=True, **kwargs)
        self.maxpool3 = layers.Pooling(type_pool='max', name='poo3')
        self.conv7 = layers.Conv2D(48, kernel_size=3, name='conv7', batch_norm=True, **kwargs)
        self.conv8 = layers.Conv2D(64, kernel_size=3, name='conv8', batch_norm=True, **kwargs)
        self.conv85 = layers.Conv2D(96, kernel_size=3, name='conv8', batch_norm=True, **kwargs)
        self.maxpool4 = layers.Pooling(type_pool='max', name='poo4')
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.fc1 = tf.keras.layers.Dense(64, name='fc1', activation='relu')
        self.dp1 = tf.keras.layers.Dropout(0.5, name='dropout_1')
        self.fc2 = tf.keras.layers.Dense(128, name='fc2', activation='relu')
        self.dp2 = tf.keras.layers.Dropout(0.5, name='dropout_2')
        self.prob = tf.keras.layers.Dense(4, activation='softmax', name='prob', **kwargs)

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv65(x)
        x = self.maxpool3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv85(x)
        x = self.maxpool4(x)
        x = self.gap(x)
        x = self.fc1(x)
        x = self.dp1(x, training=training)
        x = self.fc2(x)
        x = self.dp2(x, training=training)
        x = self.prob(x)
        return x


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



class EyeDeepNET(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super(EyeDeepNET, self).__init__(name=name)
        self.conv1 = layers.Conv2D(8, kernel_size=3, name='conv1', batch_norm=True, **kwargs)
        self.conv2 = layers.Conv2D(8, kernel_size=3, name='conv2', batch_norm=True, **kwargs)
        self.maxpool1 = layers.Pooling(type_pool='max', name='pool1')
        self.conv3 = layers.Conv2D(16, kernel_size=3, name='conv3', batch_norm=True, **kwargs)
        self.conv4 = layers.Conv2D(16, kernel_size=3, name='conv4', batch_norm=True, **kwargs)
        self.maxpool2 = layers.Pooling(type_pool='max', name='pool2')
        self.conv5 = layers.Conv2D(32, kernel_size=3, name='conv5', batch_norm=True, **kwargs)
        self.conv6 = layers.Conv2D(32, kernel_size=3, name='conv6', batch_norm=True, **kwargs)
        self.maxpool3 = layers.Pooling(type_pool='max', name='poo3')
        self.conv7 = layers.Conv2D(48, kernel_size=3, name='conv7', batch_norm=True, **kwargs)
        self.conv8 = layers.Conv2D(64, kernel_size=3, name='conv8', batch_norm=True, **kwargs)
        self.maxpool4 = layers.Pooling(type_pool='max', name='poo4')
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.fc1 = tf.keras.layers.Dense(64, name='fc1', activation='relu')
        self.dp1 = tf.keras.layers.Dropout(0.5, name='dropout_1')
        self.fc2 = tf.keras.layers.Dense(128, name='fc2', activation='relu')
        self.dp2 = tf.keras.layers.Dropout(0.5, name='dropout_2')
        self.prob = tf.keras.layers.Dense(4, activation='softmax', name='prob', **kwargs)

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool4(x)
        x = self.gap(x)
        x = self.fc1(x)
        x = self.dp1(x, training=training)
        x = self.fc2(x)
        x = self.dp2(x, training=training)
        x = self.prob(x)
        return x


class AttentionNetV2(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super(AttentionNetV2, self).__init__(name=name)
        self.conv = layers.Conv2D(8, kernel_size=3, name='init_conv', batch_norm=True, **kwargs)
        # self.pool1 = layers.Pooling(pool_size=2, name='pool1')
        self.block1 = layers.IdentityBlock([8, 8, 8], kernel_size=3, name='block1', **kwargs)
        self.block2 = layers.build_ResNeXt_block(filters=16, strides=2, groups=4, repeat_num=2,
                                                 name='block2', attention=True, **kwargs)
        self.block3 = layers.build_ResNeXt_block(filters=32, strides=2, groups=4, repeat_num=3,
                                                 name='block3', attention=True, **kwargs)
        self.block4 = layers.build_ResNeXt_block(filters=32, strides=2, groups=4, repeat_num=4,
                                                 name='block4', attention=True, **kwargs)
        self.block5 = layers.build_ResNeXt_block(filters=48, strides=2, groups=4, repeat_num=5,
                                                 name='block5', attention=True, **kwargs)
        self.block6 = layers.build_ResNeXt_block(filters=64, strides=2, groups=4, repeat_num=6,
                                                 name='block6', attention=True, **kwargs)
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        # self.fc1 = tf.keras.layers.Dense(64, name='fc1', activation='relu')
        self.dp = tf.keras.layers.Dropout(0.5, name='dropout_1')
        self.prob = tf.keras.layers.Dense(4, activation='softmax', name='prob', **kwargs)

    def call(self, inputs, training=False):
        # x = self.pool1(self.conv(inputs))
        x = self.conv(inputs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.gap(x)
        # x = self.fc1(x)
        x = self.dp(x, training=training)
        x = self.prob(x)
        return x


class AttentionNetV3(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super(AttentionNetV3, self).__init__(name=name)
        self.conv1 = layers.Conv2D(4, kernel_size=5, name='init_conv', batch_norm=True, **kwargs)
        self.conv2 = layers.Conv2D(8, kernel_size=3, strides=2, padding='valid', name='conv2', batch_norm=True, **kwargs)
        self.pool1 = layers.Pooling(pool_size=2, name='pool1')
        self.block1 = layers.IdentityBlock([8, 8, 8], kernel_size=3, name='block1', **kwargs)
        self.block2 = layers.build_ResNeXt_block(filters=16, strides=1, groups=3, repeat_num=1,
                                                 name='block2', attention=True, **kwargs)
        self.pool2 = layers.Pooling(pool_size=2, name='pool2')
        self.block3 = layers.build_ResNeXt_block(filters=32, strides=1, groups=2, repeat_num=2,
                                                 name='block3', attention=True, **kwargs)
        self.pool3 = layers.Pooling(pool_size=2, name='pool3')
        self.block4 = layers.build_ResNeXt_block(filters=48, strides=1, groups=1, repeat_num=3,
                                                 name='block4', attention=True, **kwargs)
        self.pool4 = layers.Pooling(pool_size=2, name='pool4')
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.dp = tf.keras.layers.Dropout(0.5, name='dropout_1')
        self.prob = tf.keras.layers.Dense(4, activation='softmax', name='prob', **kwargs)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(self.conv2(x))
        x = self.block1(x)
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.gap(x)
        x = self.dp(x, training=training)
        x = self.prob(x)
        return x


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


class Wrapper(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super(Wrapper, self).__init__(name=name)
        self.conv1 = layers.Conv2D(3, kernel_size=3, name='init_conv', batch_norm=True, **kwargs)
        self.base_model = get_base(**kwargs)
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.dp = tf.keras.layers.Dropout(0.5, name='dropout_1')
        self.prob = tf.keras.layers.Dense(4, activation='softmax', name='prob', **kwargs)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.base_model(x)
        x = self.gap(x)
        x = self.dp(x, training=training)
        x = self.prob(x)
        return x


class SmallResNet(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super(SmallResNet, self).__init__(name=name)
        self.conv1 = layers.Conv2D(8, kernel_size=3, name='conv1', batch_norm=True, **kwargs)
        self.conv2 = layers.Conv2D(8, kernel_size=3, name='conv2', batch_norm=True, **kwargs)
        self.maxpool1 = layers.Pooling(type_pool='max', name='pool1')
        self.block1 = layers.IdentityBlock([8, 8, 8], kernel_size=3, name='block1', **kwargs)
        self.block2 = layers.ConvBlock([16, 16, 16], kernel_size=3, name='block2', **kwargs)
        self.block3 = layers.ConvBlock([32, 32, 32], kernel_size=3, name='block3', **kwargs)
        self.block4 = layers.ConvBlock([48, 48, 48], kernel_size=3, name='block4', **kwargs)
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.fc1 = tf.keras.layers.Dense(64, name='fc1', activation='relu')
        self.dp1 = tf.keras.layers.Dropout(0.5, name='dropout_1')
        self.fc2 = tf.keras.layers.Dense(128, name='fc2', activation='relu')
        self.dp2 = tf.keras.layers.Dropout(0.5, name='dropout_2')
        self.prob = tf.keras.layers.Dense(4, activation='softmax', name='prob', **kwargs)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.fc1(x)
        x = self.dp1(x, training=training)
        x = self.fc2(x)
        x = self.dp2(x, training=training)
        x = self.prob(x)
        return x

