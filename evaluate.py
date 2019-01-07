from data_loader.tfrecord_data_loader import TfrecordDataLoader
from models.transfer_learning_model import TransferLearningModel
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from tensorflow.train import latest_checkpoint

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    print('Create the data generator.')
    data_loader = TfrecordDataLoader(config)

    model = TransferLearningModel(config).model
    x, y = data_loader.get_train_data()
    loss, acc = model.evaluate(x=x, y=y, steps=340)
    print(loss)
    print(acc)

    latest = latest_checkpoint(config.checkpoint_dir)
    model.load_weights(latest)
    loss, acc = model.evaluate(x=x, y=y, steps=340)
    print(loss)
    print(acc)