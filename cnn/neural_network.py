from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense, Dropout
# model = Sequential()

# model.add((Conv2D(60,(5,5),input_shape=(32, 32, 1) ,padding = 'same' ,activation='relu')))
# model.add((Conv2D(60, (5,5),padding="same",activation='relu')))
# model.add(MaxPooling2D(pool_size=(2,2)))
# #model.add(Dropout(0.25))

# model.add((Conv2D(30, (3,3),padding="same", activation='relu')))
# model.add((Conv2D(30, (3,3), padding="same", activation='relu')))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.5))

# model.add(Flatten())
# model.add(Dense(500,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))


class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, Saved_Weights_Path=None):
        # Initialize the Model
        # model = Sequential()

        # # First CONV => RELU => POOL Layer
        # model.add(Conv2D(20, 5, 5, padding="same",
        #           input_shape=(depth, height, width)))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2),
        #           strides=(2, 2), data_format="channels_first"))

        # # Second CONV => RELU => POOL Layer
        # model.add(Conv2D(50, 5, 5, padding="same"))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2),
        #           strides=(2, 2), data_format="channels_first", padding="same"))

        # # Third CONV => RELU => POOL Layer
        # # Convolution -> ReLU Activation Function -> Pooling Layer
        # model.add(Conv2D(100, 5, 5, padding="same"))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2),
        #           strides=(2, 2), data_format="channels_first", padding="same"))

        # # FC => RELU layers
        # #  Fully Connected Layer -> ReLU Activation Function
        # model.add(Flatten())
        # model.add(Dense(500))
        # model.add(Activation("relu"))

        # # Using Softmax Classifier for Linear Classification
        # model.add(Dense(total_classes))
        # model.add(Activation("softmax"))

        model = Sequential()

        model.add((Conv2D(60, (5, 5), input_shape=(28, 28, 1),
                  padding='same', activation='relu')))
        model.add((Conv2D(60, (5, 5), padding="same", activation='relu')))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add((Conv2D(30, (3, 3), padding="same", activation='relu')))
        model.add((Conv2D(30, (3, 3), padding="same", activation='relu')))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        # If the saved_weights file is already present i.e model is pre-trained, load that weights
        if Saved_Weights_Path is not None:
            model.load_weights(Saved_Weights_Path)
        return model


# --------------------------------- EOC ------------------------------------
CNN.build(width=28, height=28, depth=1,
          total_classes=10, Saved_Weights_Path=None)
