from keras.datasets import cifar100
from keras import layers
from keras import Sequential
from keras.utils import to_categorical
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
NUM_CLASSES = 100

class NN():
    def __init__(self) -> None:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()
        # self.x_train, self.x_test, self.y_train, self.y_test = (self.x_train / 255) - 0.5, (self.x_test / 255) - 0.5, to_categorical(self.y_train, NUM_CLASSES), to_categorical(self.y_test, NUM_CLASSES)
        self.batch_size, self.epochs = 1024, 15
        
    def vgg_16(self) -> Sequential():
        model = Sequential()
        model.add(layers.Conv2D(input_shape=(32,32,3), filters=32, kernel_size= (3,3), activation="relu", padding="same"))
        model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=NUM_CLASSES, activation="softmax"))
        model.summary()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=['accuracy'])
        return model
    
    def vgg_scaled_down(self) -> Sequential():
        model=Sequential()
        model.add(layers.Conv2D(input_shape=(32,32,3), filters=32, kernel_size= (3,3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(layers.Conv2D(filters=64, kernel_size= (3,3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(layers.Conv2D(filters=128, kernel_size= (3,3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(layers.Conv2D(filters=256, kernel_size= (3,3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(layers.Conv2D(filters=512, kernel_size= (3,3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),padding='same'))

        model.add(layers.Flatten())
        model.add(layers.Dense(512,activation='relu'))
        model.add(layers.Dense(NUM_CLASSES ,activation='softmax'))
        model.summary()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=['accuracy'])
        return model
    
    def exc(self, model: Sequential()) -> None:
        model.fit(self.x_train, self.y_train, batch_size = self.batch_size, epochs = self.epochs)

test = NN()
test.exc(test.vgg_scaled_down())