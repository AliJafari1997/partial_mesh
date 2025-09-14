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
def partial_mesh(e1, e2, e3, idx, num_filters, strides = 1, ratio=8, kernel_size=7):

    if idx == 1:
        e1, e2, e3 = spatial_attention(kernel_size=kernel_size)(e1), CBAM(e2), CBAM(e3)
        
        e1 = Conv2D(num_filters, 3, strides=strides, padding='same')(e1)

        e2 = UpSampling2D((2, 2))(e2)
        e2 = Conv2D(num_filters, 3, strides=strides, padding='same')(e2)

        e3 = UpSampling2D((4, 4))(e3)
        e3 = Conv2D(num_filters, 3, strides=strides, padding='same')(e3)

    elif idx == 2:

        e1 = spatial_attention(kernel_size=kernel_size)(e1)
        e1 = AveragePooling2D((2, 2))(e1)
        e1 = Conv2D(num_filters * 2, 3, strides=strides, padding='same')(e1)

        e2 = spatial_attention(kernel_size=kernel_size)(e2)
        e2 = Conv2D(num_filters * 2, 3, strides=strides, padding='same')(e2)

        e3 = CBAM(e3)
        e3 = UpSampling2D((2, 2))(e3)
        e3 = Conv2D(num_filters * 2, 3, strides=strides, padding='same')(e3)

    elif idx == 3:
        e1 = spatial_attention(kernel_size=kernel_size)(e1)
        e1 = AveragePooling2D((4, 4))(e1)
        e1 = Conv2D(num_filters * 4, 3, strides=strides, padding='same')(e1)

        e2 = spatial_attention(kernel_size=kernel_size)(e2)
        e2 = AveragePooling2D((2, 2))(e2)
        e2 = Conv2D(num_filters * 4, 3, strides=strides, padding='same')(e2)  

        e3 = spatial_attention(kernel_size=kernel_size)(e3)
        e3 = Conv2D(num_filters * 4, 3, strides=strides, padding='same')(e3)  

    return e1 * e2 * e3


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



def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x

def residual_block(inputs, num_filters, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)

    """ Shortcut Connection (Identity Mapping) """
    s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)

    """ Addition """
    x = x + s
    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_resunet(input_shape):
    """ RESUNET Architecture """

    inputs = Input(input_shape)

    """ Endoder 1 """
    x = Conv2D(64, 3, padding="same", strides=1)(inputs)
    x = batchnorm_relu(x)
    x = Conv2D(64, 3, padding="same", strides=1)(x)
    s = Conv2D(64, 1, padding="same")(inputs)
    s1 = x + s

    """ Encoder 2, 3 """
    s2 = residual_block(s1, 128, strides=2)
    s3 = residual_block(s2, 256, strides=2)

    """ Bridge """
    b = residual_block(s3, 512, strides=2)

    par1 = partial_mesh(s1, s2, s3, 1, 64, strides = 1, ratio=8, kernel_size=7)
    par2 = partial_mesh(s1, s2, s3, 2, 64, strides = 1, ratio=8, kernel_size=7)
    par3 = partial_mesh(s1, s2, s3, 3, 64, strides = 1, ratio=8, kernel_size=7)
    
    """ Decoder 1, 2, 3 """

    x = decoder_block(b, par3, 256)
    x = decoder_block(x, par2, 128)
    x = decoder_block(x, par1, 64)

    """ Classifier """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    """ Model """
    model = Model(inputs, outputs, name="RESUNET")

    return model

if __name__ == "__main__":
    shape = (224, 224, 3)
    model = build_resunet(shape)
    model.summary()
