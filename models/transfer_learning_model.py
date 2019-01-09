from base.base_model import BaseModel
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG16, MobileNet
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


class TransferLearningModel(BaseModel):
    def __init__(self, config):
        super(TransferLearningModel, self).__init__(config)
        self._build_transfer_learning_model()

    def _build_transfer_learning_model(self):
        input = Input(shape=(224, 224, 3))
        self.model = MobileNet(weights='imagenet', input_tensor=input, include_top=False, input_shape=(224, 224, 3))

        last_layer = self.model.output
        dropout = Dropout(0.5)(last_layer)
        x= Flatten(name='flatten')(dropout)
        # x = Dense(1000, activation='softmax')(x)
        out=Dense(
            2, 
            activation='linear', 
            kernel_regularizer=l2(0.01), 
            bias_regularizer=l2(0.01), 
            name='output_layer'
            )(x)

        self.model = Model(inputs=input,outputs=out)
        for layer in self.model.layers[:-1]:
            layer.trainable = False

        self.model.layers[-1].trainable
        # self.model.layers[-2].trainable
        self.model.compile(loss='categorical_hinge',optimizer=Adam(lr=0.0005, decay=0.003), metrics=['accuracy'])
