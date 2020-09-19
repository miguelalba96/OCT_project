import tensorflow as tf
import tensorflow.keras.backend as K


# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # self.layers = []
#         for i in range(20):
#             setattr(self, "layer%i" % i, MyLayer())
#
#     def call(self, inputs):
#         x = inputs[0]
#         y = inputs[1]
#         for i in range(20):
#             x = getattr(self, "layer%i" % i)(x, y)
#         return x


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size=3, name='conv', strides=1, padding='same', depth_wise=False,
                 activ='relu', batch_norm=False, **kwargs):
        super(Conv2D, self).__init__(name=name)
        self.depth_wise = depth_wise
        self.num_filters = num_filters
        self.activation = activ
        self.batch_norm = batch_norm
        if self.depth_wise:
            self.conv = tf.keras.layers.DepthwiseConv2D(kernel_size, strides, padding,
                                                        depth_multiplier=num_filters,
                                                        name='depth_wise_conv', **kwargs)
        else:
            self.conv = tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                                               name='conv1', padding=padding, **kwargs)

        if batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(name='bn')
        if activ:
            self.activ = tf.keras.layers.Activation(activ, name=activ)

    def call(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.activation:
            x = self.activ(x)
        return x


class Pooling(tf.keras.layers.Layer):
    def __init__(self, type_pool='max', pool_size=(2, 2), name='pool', strides=None, **kwargs):
        super(Pooling, self).__init__(name=name)
        self.type_pool = type_pool
        if type_pool == 'avg':
            self.pool = tf.keras.layers.AveragePooling2D(pool_size, strides=strides, name='avg_pool', **kwargs)
        else:
            self.pool = tf.keras.layers.MaxPooling2D(pool_size, strides=strides, name='max_pool', **kwargs)

    def call(self, inputs):
        return self.pool(inputs)


class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, name='identity', **kwargs):
        super(IdentityBlock, self).__init__(name=name)
        self.filters = filters
        self.conv1 = tf.keras.layers.Conv2D(filters[0], kernel_size=1, strides=1,
                                            name='conv1', padding='valid', **kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.relu1 = tf.keras.layers.ReLU(name='relu1')
        self.conv2 = tf.keras.layers.Conv2D(filters[1], kernel_size=kernel_size, strides=1,
                                            name='conv2', padding='same', **kwargs)
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.relu2 = tf.keras.layers.ReLU(name='relu2')

        self.conv3 = tf.keras.layers.Conv2D(filters[2], kernel_size=1, strides=1,
                                            name='conv3', padding='valid', **kwargs)
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.relu3 = tf.keras.layers.ReLU(name='relu3')
        self.relu = tf.keras.layers.ReLU(name='relu')

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x += inputs
        return self.relu(x)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, name='block', strides=2, **kwargs):
        super(ConvBlock, self).__init__(name=name)
        self.filters = filters
        self.conv1 = tf.keras.layers.Conv2D(filters[0], kernel_size=1, strides=strides,
                                            name='conv1', padding='valid', **kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.relu1 = tf.keras.layers.ReLU(name='relu1')
        self.conv2 = tf.keras.layers.Conv2D(filters[1], kernel_size=kernel_size, strides=1,
                                            name='conv2', padding='same', **kwargs)
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.relu2 = tf.keras.layers.ReLU(name='relu2')

        self.conv3 = tf.keras.layers.Conv2D(filters[2], kernel_size=1, strides=1,
                                            name='conv3', padding='valid', **kwargs)
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.relu3 = tf.keras.layers.ReLU(name='relu3')

        self.conv4 = tf.keras.layers.Conv2D(filters[2], kernel_size=1, strides=strides,
                                            name='conv4', padding='valid', **kwargs)
        self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')
        self.relu = tf.keras.layers.ReLU(name='relu')

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        inputs = self.conv4(inputs)
        inputs = self.bn4(inputs)
        x += inputs
        return self.relu(x)


class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, output_dim, ratio=4, name='sq_ex', **kwargs):
        super(SqueezeExcitation, self).__init__(name=name)
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.fc1 = tf.keras.layers.Dense(output_dim // ratio, name='fc1', **kwargs)
        self.relu1 = tf.keras.layers.ReLU(name='relu1')
        self.fc2 = tf.keras.layers.Dense(output_dim, name='fc2', **kwargs)
        self.reshape = tf.keras.layers.Reshape((1, 1, output_dim), name='reshape')
        self.mult = tf.keras.layers.Multiply(name='mult')

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.gap(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = tf.nn.sigmoid(x)
        x = self.reshape(x)
        x = self.mult([inputs, x])
        return x


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, ratio=8, name='channel_att', **kwargs):
        super(ChannelAttention, self).__init__(name=name)
        self.shared1 = tf.keras.layers.Dense(channels // ratio, activation='relu', use_bias=True, **kwargs)
        self.shared2 = tf.keras.layers.Dense(channels, use_bias=True, **kwargs)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.reshape1 = tf.keras.layers.Reshape((1, 1, channels), name='reshape1')
        self.maxpool = tf.keras.layers.GlobalMaxPooling2D(name='maxp')
        self.reshape2 = tf.keras.layers.Reshape((1, 1, channels), name='reshape2')
        self.add = tf.keras.layers.Add(name='add')
        self.mult = tf.keras.layers.Multiply(name='mult')

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_outputs
        return tf.TensorShape(shape)

    def call(self, inputs, **kwargs):
        x1 = inputs
        x2 = inputs

        x1 = self.avgpool(x1)
        x1 = self.reshape1(x1)
        x1 = self.shared1(x1)
        x1 = self.shared2(x1)

        x2 = self.maxpool(x2)
        x2 = self.reshape2(x2)
        x2 = self.shared1(x2)
        x2 = self.shared2(x2)

        x = self.add([x1, x2])
        x = tf.nn.sigmoid(x)
        x = self.mult([inputs, x])
        return x


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, name='spatial_att', **kwargs):
        super(SpatialAttention, self).__init__(name=name)
        self.avgpool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))
        self.maxpool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))
        self.concat = tf.keras.layers.Concatenate(name='concat', axis=3)
        self.conv = tf.keras.layers.Conv2D(1, kernel_size=kernel_size, strides=1, activation='sigmoid',
                                           name='attention', padding='same', use_bias=False, **kwargs)
        self.mult = tf.keras.layers.Multiply(name='mult')

    def call(self, inputs, **kwargs):
        x = inputs
        x1 = self.avgpool(x)
        x2 = self.maxpool(x)
        x = self.concat([x1, x2])
        x = self.conv(x)
        x = self.mult([inputs, x])
        return x


class CBAM(tf.keras.layers.Layer):
    def __init__(self, channels, name='cbam_block', **kwargs):
        super(CBAM, self).__init__(name=name)
        self.channel_attention = ChannelAttention(channels=channels, **kwargs)
        self.spatial_attention = SpatialAttention(**kwargs)

    def call(self, inputs, **kwargs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x


class GroupConvBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, out_channels, channel_groups=1, name='group_conv', strides=1, **kwargs):
        super(GroupConvBlock, self).__init__(name=name)
        self.channel_groups = channel_groups
        self.out_channels = out_channels // channel_groups
        self.input_channels = input_channels // channel_groups
        for i in range(channel_groups):
            setattr(self, "conv%i" % i,
                    tf.keras.layers.Conv2D(channel_groups, kernel_size=3, padding='same', use_bias=False,
                                           strides=strides, **kwargs))

    def call(self, inputs, **kwargs):
        feats = []
        for i in range(self.channel_groups):
            channels = inputs[:, :, :, i * self.input_channels: (i + 1) * self.input_channels]
            xi = getattr(self, "conv%i" % i)(channels)
            feats.append(xi)
        x = tf.concat(feats, axis=-1)
        return x


class DepthWiseDense(tf.keras.layers.Layer):
    def __init__(self, nodes_in, nodes_out=2, name='depth_wise', **kwargs):
        super(DepthWiseDense, self).__init__(name=name)
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.split = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=nodes_in, axis=1))
        for i in range(nodes_in):
            setattr(self, "dense%i" % i, tf.keras.layers.Dense(nodes_out, **kwargs))

    def call(self, inputs, **kwargs):
        outputs = []
        inputs = self.split(inputs)
        for i in range(self.nodes_in):
            split_i = inputs[i]
            xi = getattr(self, "dense%i" % i)(split_i)
            outputs.append(xi)
        return tf.concat(outputs, axis=-1)


class ResNextBottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups, name='resnext_bn', attention=False, **kwargs):
        super(ResNextBottleNeck, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1,
                                            name='conv1', padding="same", **kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.group_conv = GroupConvBlock(input_channels=filters, out_channels=filters, channel_groups=groups,
                                         name='group_conv', strides=strides, **kwargs)
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.conv2 = tf.keras.layers.Conv2D(filters=2 * filters, kernel_size=(1, 1), strides=1,
                                            name='conv2', padding="same", **kwargs)
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.shortcut_conv = tf.keras.layers.Conv2D(filters=2 * filters, kernel_size=(1, 1), strides=strides,
                                                    name='conv3', padding="same", **kwargs)
        self.shortcut_bn = tf.keras.layers.BatchNormalization(name='bn4')
        self.attention = attention
        if attention:
            self.attention = CBAM(2 * filters, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.group_conv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        if self.attention:
            x = self.attention(x)

        output = tf.nn.relu(tf.keras.layers.add([x, shortcut]))
        return output


def build_ResNeXt_block(filters, strides, groups, repeat_num, name='rnext_block', **kwargs):
    block = tf.keras.Sequential(name=name)
    block.add(ResNextBottleNeck(filters=filters, strides=strides, groups=groups, name='rnext_block_0', **kwargs))
    for i in range(1, repeat_num):
        block.add(ResNextBottleNeck(filters=filters, strides=1, groups=groups, name='rnext_block_{}'.format(i)))
    return block


def conv_layer(x, num_filters, kernel_size, padding='same', strides=1, act_type='relu',
               batch_norm=False, scope='conv', **opts):
    x = tf.keras.layers.Conv2D(num_filters,
                               kernel_size=kernel_size,
                               strides=strides, name=scope, padding=padding, **opts)(x)
    x = tf.keras.layers.BatchNormalization(name=scope + '/bn')(x) if batch_norm else x
    if act_type == 'relu':
        x = tf.keras.layers.ReLU(name=scope + '/relu')(x)
    return x
