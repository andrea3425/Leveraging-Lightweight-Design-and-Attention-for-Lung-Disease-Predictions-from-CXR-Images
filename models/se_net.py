import tensorflow as tf
from .mobilenet_v1 import conv_block, depthwise_conv_block
from .mobilenet_v2 import bottleneck

def se_block(input, reduction_ratio, id):
    '''Squeeze-and-Exitation Block.'''

    c = input.shape[-1]

    # Squeeze
    X = tf.keras.layers.GlobalAveragePooling2D(name=f"se_{id}_avg_pool")(input)
    X = tf.keras.layers.Reshape((1, 1, c))(X)

    # Excitation
    X = tf.keras.layers.Dense(units=(c // reduction_ratio), activation="relu", use_bias=False, name=f"fc1_{id}")(X)
    X = tf.keras.layers.Dense(units=c, activation="sigmoid", use_bias=False, name=f"fc2_{id}")(X)

    return tf.keras.layers.Multiply(name=f"gate_{id}")([input, X])

### SE-MobileNetV1

def se_depthwise_conv_block(input, filters, strides=(1,1), alpha=1.0, reduction_ratio=16, se=False, id=None):
    '''Depthwise Separable Convolution with Squeeze-and-Exitation'''

    k = input.shape[-1]

    # Depth-Wise Convolution
    X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=strides, padding='same', name=f'conv_dw_{id}')(input)
    X = tf.keras.layers.BatchNormalization(name=f'conv_dw_{id}_bn')(X)
    X = tf.keras.layers.Activation('relu', name=f'conv_dw_{id}_relu')(X)

    # Point-Wise Convolution
    X = tf.keras.layers.Conv2D(filters=int(filters*alpha), kernel_size=(1,1), name=f'conv_pw_{id}')(X)
    X = tf.keras.layers.BatchNormalization(name=f'conv_pw_{id}_bn')(X)
    X = tf.keras.layers.Activation('relu', name=f'conv_pw_{id}_relu')(X)
    
    # SE Block
    if se:
        X = se_block(X, reduction_ratio, id=id)

    return X

def se_mobilenet_v1(input_shape, reduction_ratio=16, include_top=True, alpha=1.0, classes=3):
    '''MobileNetV1 enhanced with SE blocks'''

    X_input = tf.keras.layers.Input(input_shape, name='input_layer')

    X = conv_block(X_input, 32, (3,3), (2,2), alpha, id=1)
    X = se_depthwise_conv_block(X, 64, (1,1), alpha, reduction_ratio, se=False, id=1)

    X = se_depthwise_conv_block(X, 128, (2,2), alpha, reduction_ratio, se=False, id=2)
    X = se_depthwise_conv_block(X, 128, (1,1), alpha, reduction_ratio, se=False, id=3)

    X = se_depthwise_conv_block(X, 256, (2,2), alpha, reduction_ratio, se=False, id=4)
    X = se_depthwise_conv_block(X, 256, (1,1), alpha, reduction_ratio, se=False, id=5)

    X = se_depthwise_conv_block(X, 512, (2,2), alpha, reduction_ratio, se=True, id=6)
    X = se_depthwise_conv_block(X, 512, (1,1), alpha, reduction_ratio, se=True, id=7)
    X = se_depthwise_conv_block(X, 512, (1,1), alpha, reduction_ratio, se=True, id=8)
    X = se_depthwise_conv_block(X, 512, (1,1), alpha, reduction_ratio, se=True, id=9)
    X = se_depthwise_conv_block(X, 512, (1,1), alpha, reduction_ratio, se=True, id=10)
    X = se_depthwise_conv_block(X, 512, (1,1), alpha, reduction_ratio, se=True, id=11)

    X = se_depthwise_conv_block(X, 1024, (2,2), alpha, reduction_ratio, se=True, id=12)
    X = se_depthwise_conv_block(X, 1024, (1,1), alpha, reduction_ratio, se=True, id=13)
    X = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(X)

    if include_top == False:
        return tf.keras.models.Model(X_input, X, name='se_mobilenet_v1')

    #X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(units=classes, activation=None, name='dense')(X)

    return tf.keras.models.Model(X_input, X, name='se_mobilenet_v1')

### SE-MobileNetV2

def se_bottleneck(input, expansion, stride, alpha, filters, reduction_ratio, se=False, block_id=None):
    """Bottleneck residual block with Squeeze-and-Exitation."""

    # Number of input channels
    k = input.shape[-1]

    # Expansion
    X = tf.keras.layers.Conv2D(filters=(expansion * k), kernel_size=(1,1), strides=(1,1), name=f"expansion_{block_id}")(input)
    X = tf.keras.layers.BatchNormalization(name=f"expansion_{block_id}_bn")(X)
    X = tf.keras.layers.Activation('relu', name=f"expansion_{block_id}_relu")(X)

    # Depthwise convolution
    X = tf.keras.layers.DepthwiseConv2D((3,3), strides=stride, padding='same', name=f"conv_dw_{block_id}")(X)
    X = tf.keras.layers.BatchNormalization(name=f"conv_dw_{block_id}_bn")(X)
    X = tf.keras.layers.Activation('relu', name=f"conv_dw_{block_id}_relu")(X)

    # Squeeze-and-Excitation
    if se:
        X = se_block(X, reduction_ratio, id=block_id)

    # Projection
    X = tf.keras.layers.Conv2D(filters=int(filters * alpha), kernel_size=(1,1), strides=(1,1), name=f"projection_{block_id}")(X)
    X = tf.keras.layers.BatchNormalization(name=f"projection_{block_id}_bn")(X)

    # Shortcut connection
    if k == X.shape[-1] and stride == (1,1):
        X = tf.keras.layers.Add(name=f"shortcut_{block_id}")([X, input])

    return X

def se_mobilenet_v2(input_shape, reduction_ratio=16, include_top=True, alpha=1.0, classes=3):
    '''MobileNetV2 with SE Blocks.'''

    X_input = tf.keras.layers.Input(input_shape, name='input_layer')

    X = conv_block(X_input, filters=32, kernel_size=(3,3), strides=(2,2), alpha=alpha, id="1")

    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=16, reduction_ratio=reduction_ratio, se=False, block_id="1")

    X = se_bottleneck(X, expansion=6, stride=(2,2), alpha=alpha, filters=24, reduction_ratio=reduction_ratio, se=False, block_id="2")
    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=24, reduction_ratio=reduction_ratio, se=False, block_id="3")

    X = se_bottleneck(X, expansion=6, stride=(2,2), alpha=alpha, filters=32, reduction_ratio=reduction_ratio, se=False, block_id="4")
    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=32, reduction_ratio=reduction_ratio, se=False, block_id="5")
    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=32, reduction_ratio=reduction_ratio, se=False, block_id="6")

    X = se_bottleneck(X, expansion=6, stride=(2,2), alpha=alpha, filters=64, reduction_ratio=reduction_ratio, se=False, block_id="7")
    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=64, reduction_ratio=reduction_ratio, se=False, block_id="8")
    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=64, reduction_ratio=reduction_ratio, se=False, block_id="9")
    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=64, reduction_ratio=reduction_ratio, se=False, block_id="10")

    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=96, reduction_ratio=reduction_ratio, se=True, block_id="11")
    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=96, reduction_ratio=reduction_ratio, se=True, block_id="12")
    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=96, reduction_ratio=reduction_ratio, se=True, block_id="13")

    X = se_bottleneck(X, expansion=6, stride=(2,2), alpha=alpha, filters=160, reduction_ratio=reduction_ratio, se=True, block_id="14")
    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=160, reduction_ratio=reduction_ratio, se=True, block_id="15")
    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=160, reduction_ratio=reduction_ratio, se=True, block_id="16")

    X = se_bottleneck(X, expansion=6, stride=(1,1), alpha=alpha, filters=320, reduction_ratio=reduction_ratio, se=True, block_id="17")

    X = conv_block(X, filters=1280, kernel_size=(1,1), strides=(1,1), alpha=alpha, id="2")
    X = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(X)

    if include_top == False:
        return tf.keras.models.Model(X_input, X, name='se_mobilenet_v2')

    X = tf.keras.layers.Dense(units=classes, activation=None, name='dense')(X)
    return tf.keras.models.Model(X_input, X, name='se_mobilenet_v2')
