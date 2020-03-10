
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU

# IMGWIDTH = 256


class Meso1():
    """
    Feature extraction + Classification
    """

    def __init__(self,image_size=256, learning_rate=0.001, dl_rate=1):
        self.image_size = image_size
        self.model = self.init_model(dl_rate)
    #         self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    def init_model(self, dl_rate):
        x = Input(shape=(self.image_size, self.image_size, 3))

        x1 = Conv2D(16, (3, 3), dilation_rate=dl_rate, strides=1, padding='same', activation='relu')(x)
        x1 = Conv2D(4, (1, 1), padding='same', activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(8, 8), padding='same')(x1)

        y = Flatten()(x1)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)
        return KerasModel(inputs=x, outputs=y)


class Meso4():
    def __init__(self,image_size=256, learning_rate=0.001):
        self.image_size = image_size
        self.model = self.init_model()

    def init_model(self):
        x = Input(shape=(self.image_size, self.image_size, 3))

        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(rate =0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return KerasModel(inputs=x, outputs=y)


class Meso42():
    def __init__(self,image_size=256, learning_rate=0.001):
        self.image_size = image_size
        self.model = self.init_model()
    #         self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    def init_model(self):
        x = Input(shape=(self.image_size, self.image_size, 3))

        x1 = Conv2D(4, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)

        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(16, (3, 3), padding='same', activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x2 = Conv2D(32, (3, 3), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        #         x2 = x2 +x1
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(x3)
        x3 = BatchNormalization()(x3)
        #         x3 = x3+x2
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(64, (3, 3), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = Conv2D(128, (3, 3), padding='same', activation='relu')(x4)
        x4 = BatchNormalization()(x4)
        #         x4 = x4+x3
        x4 = MaxPooling2D(pool_size=(2, 2), padding='same')(x4)

        x5 = Conv2D(128, (3, 3), padding='same', activation='relu')(x4)
        x5 = BatchNormalization()(x5)
        x5 = Conv2D(256, (3, 3), padding='same', activation='relu')(x5)
        x5 = BatchNormalization()(x5)
        #         x5 = x5+x4
        x5 = MaxPooling2D(pool_size=(2, 2), padding='same')(x5)

        x6 = Conv2D(256, (3, 3), padding='same', activation='relu')(x5)
        x6 = BatchNormalization()(x6)
        x6 = Conv2D(512, (3, 3), padding='same', activation='relu')(x6)
        x6 = BatchNormalization()(x6)
        #         x6=x6+x5
        x6 = MaxPooling2D(pool_size=(2, 2), padding='same')(x6)

        x7 = Conv2D(1024, (3, 3), padding='same', activation='relu')(x6)
        x7 = BatchNormalization()(x7)

        y = Flatten()(x7)
        y = Dropout(rate =0.2)(y)
        y = Dense(1024)(y)
        y = LeakyReLU(alpha=0.1)(y)

        y = Dropout(rate = 0.2)(y)
        y = Dense(256)(y)
        y = LeakyReLU(alpha=0.1)(y)

        y = Dropout(0.2)(y)
        y = Dense(64)(y)
        y = LeakyReLU(alpha=0.1)(y)

        y = Dropout(0.2)(y)
        y = Dense(1, activation='sigmoid')(y)

        return KerasModel(inputs=x, outputs=y)


class MesoInception4():
    def __init__(self,image_size=256, learning_rate=0.001):
        self.image_size = image_size
        self.model = self.init_model()
    #         self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)

            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)

            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate=2, strides=1, padding='same', activation='relu')(x3)

            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate=3, strides=1, padding='same', activation='relu')(x4)

            y = Concatenate(axis=-1)([x1, x2, x3, x4])

            return y

        return func

    def init_model(self):
        x = Input(shape=(self.image_size, self.image_size, 3))

        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(rate  = 0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(rate  = 0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return KerasModel(inputs=x, outputs=y)
