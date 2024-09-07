import tensorflow as tf

def conv_block(X, filters=32, kernel_size=(3,3), strides=(2,2), alpha=1.0, id="1"):
    '''Standard Convolution'''

    # First Conv
    X = tf.keras.layers.Conv2D(filters=int(filters*alpha), kernel_size=kernel_size, strides=strides, padding='same', name=f'conv_{id}')(X)
    X = tf.keras.layers.BatchNormalization(name=f'conv_{id}_bn')(X)
    X = tf.keras.layers.Activation('relu', name=f'conv_{id}_relu')(X)

    return X

def depthwise_conv_block(X, filters, strides=(1,1), alpha=1.0, id=None):
    '''Depthwise Separable Convolution'''

    # Depth-Wise Convolution
    X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=strides, padding='same', name=f'conv_dw_{id}')(X)
    X = tf.keras.layers.BatchNormalization(name=f'conv_dw_{id}_bn')(X)
    X = tf.keras.layers.Activation('relu', name=f'conv_dw_{id}_relu')(X)

    # Point-Wise Convolution
    X = tf.keras.layers.Conv2D(filters=int(filters*alpha), kernel_size=(1,1), name=f'conv_pw_{id}')(X)
    X = tf.keras.layers.BatchNormalization(name=f'conv_pw_{id}_bn')(X)
    X = tf.keras.layers.Activation('relu', name=f'conv_pw_{id}_relu')(X)

    return X


def mobilenet_v1(input_shape, include_top=True, alpha=1.0, classes=3):
    '''Implementation of MobileNetV1 architecture'''

    X_input = tf.keras.layers.Input(input_shape, name='input_layer')

    X = conv_block(X_input, 32, (3,3), (2,2), alpha=alpha, id="1")
    X = depthwise_conv_block(X, 64, (1,1), alpha, id=1)

    X = depthwise_conv_block(X, 128, (2,2), alpha, id=2)
    X = depthwise_conv_block(X, 128, (1,1), alpha, id=3)

    X = depthwise_conv_block(X, 256, (2,2), alpha, id=4)
    X = depthwise_conv_block(X, 256, (1,1), alpha, id=5)

    X = depthwise_conv_block(X, 512, (2,2), alpha, id=6)
    for i in range(5):
        X = depthwise_conv_block(X, 512, (1,1), alpha, id=7+i)

    X = depthwise_conv_block(X, 1024, (2,2), alpha, id=12)
    X = depthwise_conv_block(X, 1024, (1,1), alpha, id=13)
    X = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(X)

    if include_top == False:
        return tf.keras.models.Model(X_input, X, name='mobilenet_v1')

    X = tf.keras.layers.Dense(units=classes, activation=None, name='dense')(X)

    return tf.keras.models.Model(X_input, X, name='mobilenet_v1')