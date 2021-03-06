import comet_ml
from data_loader.tfrecord_data_loader import TfrecordDataLoader
from models.conv_aesthetic_model import ConvAestheticModel
from models.transfer_learning_model import TransferLearningModel
from trainers.conv_aesthetic_trainer import ConvAestheticModelTrainer
from models.resnet import ResnetBuilder
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from tensorflow.keras.models import Model

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = TfrecordDataLoader(config)

    print('Create the model.')
    #model = ConvAestheticModel(config)
    model = TransferLearningModel(config)
    #model = ResnetBuilder.build_resnet_18((256, 256, 3), 2)
    #model.compile(
    #    loss='hinge',
    #    optimizer=Adam(amsgrad=True, lr=0.0005),
    #    metrics=['accuracy']
    #)
    print('Create the trainer')
    
    trainer = ConvAestheticModelTrainer(model.model, data_loader.get_train_data(), data_loader.get_test_data(), config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
