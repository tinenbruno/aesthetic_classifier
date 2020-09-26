from data_loader.tfrecord_data_loader import TfrecordDataLoader
from models.transfer_learning_model import TransferLearningModel
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from tensorflow.train import latest_checkpoint
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

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
    latest = latest_checkpoint(config.checkpoint_dir)
    model.load_weights(latest)

    image_path = "image.png"

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    model.predict(x)
    print('Predicted:', decode_predictions(preds, top=1)[0])

if __name__ == '__main__':
    main()