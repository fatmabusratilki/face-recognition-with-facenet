import tensorflow as tf
from tensorflow.keras import layers, models

def FaceNet(embedding_size=128):
    inputs = tf.keras.Input(shape=(160, 160, 3))

    x = layers.Conv2D(64, (3, 3), padding='valid', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(embedding_size, activation=None)(x)

    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)
    return models.Model(inputs, outputs)
