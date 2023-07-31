from keras import datasets
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers.legacy import Adam
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class NN():
    def __init__(self, learning_rate = 1e-3, batch_size = 128, epochs = 10) -> None:
        [self.x_train, self.y_train], [self.x_test, self.y_test] = datasets.mnist.load_data()
        self.x_test, self.x_train = self.x_test / 255, self.x_train / 255
        self.input_shape, self.output_shape = [self.x_train.shape[1], self.x_train.shape[2]], 10
        self.learning_rate, self.batch_size, self.epochs = learning_rate, batch_size, epochs

    def model(self) -> Sequential:
        model = Sequential()
        model.add(Flatten(input_shape = self.input_shape))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.output_shape, activation='softmax'))
        model.optimizer = Adam(learning_rate = self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', metrics =['accuracy'])
        model.summary()
        return model

    def execute(self) -> None:
        model = self.model()
        model.fit(self.x_train, self.y_train, 
                  epochs = self.epochs, 
                  batch_size = self.batch_size, 
                  validation_data = (self.x_test, self.y_test))
        scores = model.evaluate(self.x_test, self.y_test)
        print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100), end = '')
        print("\t%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
                
        
neural_network = NN(learning_rate = 1e-3, batch_size = 128, epochs = 10)
neural_network.execute()
