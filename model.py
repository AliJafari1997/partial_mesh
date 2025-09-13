import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class channel_attention(tf.keras.layers.Layer):
    """ 
    channel attention module 
    """
    def __init__(self, ratio=8, **kwargs):
        self.ratio = ratio
        super(channel_attention, self).__init__(**kwargs)

    def get_config(self):
        config = super(channel_attention, self).get_config().copy()
        config.update({
            'ratio': self.ratio
        })
        return config

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = tf.keras.layers.Dense(channel // self.ratio,
                                                 activation='relu',
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        super(channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        channel = inputs.get_shape().as_list()[-1]

        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)    
        avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
        max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        feature = tf.keras.layers.Add()([avg_pool, max_pool])
        feature = tf.keras.layers.Activation('sigmoid')(feature)

        return tf.keras.layers.multiply([inputs, feature])

class spatial_attention(tf.keras.layers.Layer):
    """ spatial attention module 
    """
    def __init__(self, kernel_size=7, **kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention, self).__init__(**kwargs)


    def get_config(self):
        config = super(spatial_attention, self).get_config().copy()
        config.update({
            'kernel_size': self.kernel_size
        })
        return config

    def build(self, input_shape):
        self.conv2d = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size,
                                             strides=1, padding='same', activation='sigmoid',
                                             kernel_initializer='he_normal', use_bias=False)
        super(spatial_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        feature = self.conv2d(concat)	
            
        return tf.keras.layers.multiply([inputs, feature])


def ch_attention(x, ratio = 8):
    x = channel_attention(ratio = ratio)(x)
    return x


def s_attention(x, kernel_size=7):
    x = spatial_attention(kernel_size=kernel_size)(x)
    return x


def CBAM(x, ratio=8, kernel_size=7):
    ch_attention = channel_attention(ratio = ratio)
    s_attention = spatial_attention(kernel_size=kernel_size)
    x = ch_attention(x)
    x = s_attention(x)
    return x

def partial_mesh(d1, d2, d3, d4, idx, num_filters, strides = 1, ratio=8, kernel_size=7):

    if idx == 4:
        d1, d2, d3, d4 = spatial_attention(kernel_size=kernel_size)(d1), channel_attention(ratio = ratio)(d2), channel_attention(ratio = ratio)(d3), channel_attention(ratio = ratio)(d4)
        
        d4 = Conv2D(num_filters, 3, strides=strides, padding='same')(d4)

        d3 = UpSampling2D((2, 2))(d3)
        d3 = Conv2D(num_filters, 3, strides=strides, padding='same')(d3)


        d2 = UpSampling2D((4, 4))(d2)
        d2 = Conv2D(num_filters, 3, strides=strides, padding='same')(d2)


        d1 = UpSampling2D((8, 8))(d1)
        d1 = Conv2D(num_filters, 3, strides=strides, padding='same')(d1)


    elif idx == 3:

        d4 = spatial_attention(kernel_size=kernel_size)(d4)
        d4 = AveragePooling2D((2, 2))(d4)
        d4 = Conv2D(num_filters * 2, 3, strides=strides, padding='same')(d4)

        d3 = spatial_attention(kernel_size=kernel_size)(d3)
        d3 = Conv2D(num_filters * 2, 3, strides=strides, padding='same')(d3)

        d2 = channel_attention(ratio = ratio)(d2)
        d2 = UpSampling2D((2, 2))(d2)
        d2 = Conv2D(num_filters * 2, 3, strides=strides, padding='same')(d2)

        d1 = channel_attention(ratio = ratio)(d1)
        d1 = UpSampling2D((4, 4))(d1)
        d1 = Conv2D(num_filters * 2, 3, strides=strides, padding='same')(d1)

    elif idx == 2:
        d4 = spatial_attention(kernel_size=kernel_size)(d4)
        d4 = AveragePooling2D((4, 4))(d4)
        d4 = Conv2D(num_filters * 4, 3, strides=strides, padding='same')(d4)

        d3 = spatial_attention(kernel_size=kernel_size)(d3)
        d3 = AveragePooling2D((2, 2))(d3)
        d3 = Conv2D(num_filters * 4, 3, strides=strides, padding='same')(d3)  

        d2 = spatial_attention(kernel_size=kernel_size)(d2)
        d2 = Conv2D(num_filters * 4, 3, strides=strides, padding='same')(d2)  

        d1 = channel_attention(ratio = ratio)(d1)
        d1 = UpSampling2D((2, 2))(d1)
        d1 = Conv2D(num_filters * 4, 3, strides=strides, padding='same')(d1)

    elif idx == 1:
        d4 = spatial_attention(kernel_size=kernel_size)(d4)
        d4 = AveragePooling2D((8, 8))(d4)
        d4 = Conv2D(num_filters * 8, 3, strides=strides, padding='same')(d4)

        d3 = spatial_attention(kernel_size=kernel_size)(d3)
        d3 = AveragePooling2D((4, 4))(d3)
        d3 = Conv2D(num_filters * 8, 3, strides=strides, padding='same')(d3)

        d2 = spatial_attention(kernel_size=kernel_size)(d2)
        d2 = AveragePooling2D((2, 2))(d2)
        d2 = Conv2D(num_filters * 8, 3, strides=strides, padding='same')(d2)

        d1 = spatial_attention(kernel_size=kernel_size)(d1)
        d1 = Conv2D(num_filters * 8, 3, strides=strides, padding='same')(d1)

    return d1 * d2 * d3 * d4

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_model(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_model(input_shape)
    model.summary()
