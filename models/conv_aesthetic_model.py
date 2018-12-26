from base.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten

class ConvAestheticModel(BaseModel):
    def __init__(self, config):
        super(ConvAestheticModel, self).__init__(config)
        self._build_model()

    def _build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(256, 256, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(5, 5), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, kernel_size=(5, 5), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(7, 7), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(7, 7), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(9, 9), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))

        self.model.compile(
              loss='categorical_crossentropy',
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])