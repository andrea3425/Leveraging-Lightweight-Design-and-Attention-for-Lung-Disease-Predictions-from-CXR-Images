import tensorflow as tf
from .mobilenet_v1 import conv_block

def bottleneck(input, expansion, stride, alpha, filters, block_id):
    """Bottleneck residual block."""

    # Number of input channels
    k = input.shape[-1]

    # Expansion
    X = tf.keras.layers.Conv2D(filters=(expansion*k), kernel_size=(1,1), strides=(1,1), padding='same', name=f"expansion_{block_id}")(input)
    X = tf.keras.layers.BatchNormalization(name=f"expansion_{block_id}_bn")(X)
    X = tf.keras.layers.Activation('relu', name=f"expansion_{block_id}_relu")(X)
    
    # Depthwise convolution
    X = tf.keras.layers.DepthwiseConv2D((3,3), strides=stride, padding='same', name=f"conv_dw_{block_id}")(X)
    X = tf.keras.layers.BatchNormalization(name=f"conv_dw_{block_id}_bn")(X)
    X = tf.keras.layers.Activation('relu', name=f"conv_dw_{block_id}_relu")(X)

    # Projection
    X = tf.keras.layers.Conv2D(filters=int(filters * alpha), kernel_size=(1,1), strides=(1,1), padding='same', name=f"projection_{block_id}")(X)
    X = tf.keras.layers.BatchNormalization(name=f"projection_{block_id}_bn")(X)

    if k == X.shape[-1] and stride == (1,1):
        X = tf.keras.layers.Add(name=f"shortcut_{block_id}")([X, input])

    return X


def mobilenet_v2(input_shape, include_top=True, alpha=1.0, classes=3):
    '''Implementation of MobileNetV2 architecture'''

    X_input = tf.keras.layers.Input(input_shape, name='input_layer')

    X = conv_block(X_input, filters=32, kernel_size=(3,3), strides=(2,2), alpha=alpha, id="1")

    X = bottleneck(input=X, expansion=1, stride=1, alpha=alpha, filters=16, block_id="1")

    X = bottleneck(input=X, expansion=6, stride=2, alpha=alpha, filters=24, block_id="2")
    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=24, block_id="3")

    X = bottleneck(input=X, expansion=6, stride=2, alpha=alpha, filters=32, block_id="4")
    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=32, block_id="5")
    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=32, block_id="6")

    X = bottleneck(input=X, expansion=6, stride=2, alpha=alpha, filters=64, block_id="7")
    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=64, block_id="8")
    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=64, block_id="9")
    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=64, block_id="10")

    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=96, block_id="11")
    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=96, block_id="12")
    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=96, block_id="13")

    X = bottleneck(input=X, expansion=6, stride=2, alpha=alpha, filters=160, block_id="14")
    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=160, block_id="15")
    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=160, block_id="16")

    X = bottleneck(input=X, expansion=6, stride=1, alpha=alpha, filters=320, block_id="17")

    X = conv_block(X, filters=1280, kernel_size=(1,1), strides=(1,1), alpha=alpha, id="2")
    X = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(X)

    if include_top == False:
        return tf.keras.models.Model(X_input, X, name='mobilenet_v2')

    X = tf.keras.layers.Dense(units=classes, activation=None, name='dense')(X)
    return tf.keras.models.Model(X_input, X, name='mobilenet_v2')