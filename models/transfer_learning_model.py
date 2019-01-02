from base.base_model import BaseModel
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from tensorflow.keras.regularizers import l2


class TransferLearningModel(BaseModel):
    def __init__(self, config):
        super(TransferLearningModel, self).__init__(config)
        self._build_transfer_learning_model()

    def _build_transfer_learning_model(self):
        input = Input(shape=(256, 256, 3))
        self.model = ResNet50(weights='imagenet', input_tensor=input, include_top=False, input_shape=(256, 256, 3))

        last_layer = self.model.output
        x= Flatten(name='flatten')(last_layer)
        out=Dense(
            2, 
            activation='linear', 
            kernel_regularizer=l2(0.01), 
            name='output_layer'
            )(x)

        self.model = Model(inputs=input,outputs=out)
        for layer in self.model.layers[:-1]:
            layer.trainable = False

        self.model.layers[-1].trainable
        self.model.compile(loss='categorical_hinge',optimizer='adadelta',metrics=['accuracy'])