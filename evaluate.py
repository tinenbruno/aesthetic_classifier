from data_loader.tfrecord_data_loader import TfrecordDataLoader
from models.transfer_learning_model import TransferLearningModel
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from tensorflow.train import latest_checkpoint
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K

def possible_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true, 0, 1)))


def true_positive(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    return K.sum(class_id_true * class_id_preds) 

def false_positive(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    return K.sum((1 - class_id_true) * class_id_preds) 

def false_negative(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    return K.sum(class_id_true * (1 - class_id_preds)) 

def true_negative(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    return K.sum((1 - class_id_true) * (1 - class_id_preds)) 

def categorical_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for
    multiclass classification problems.
    '''
    return K.mean(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)))


def specificity(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    specificity = true_negatives / (possible_negatives + K.epsilon())
    return specificity

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


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
    model.compile(loss='categorical_hinge', optimizer='adam', metrics=['accuracy', recall, precision, specificity, true_positive, false_positive, false_negative, true_negative, categorical_accuracy, possible_positives])
    x, y = data_loader.get_val_data()
    # loss, acc = model.evaluate(x=x, y=y, steps=100)
    # print(loss)
    # print(acc)

    latest = latest_checkpoint(config.checkpoint_dir)
    model.load_weights(latest)
    loss, *metrics = model.evaluate(x=x, y=y, steps=100)

    model.summary()
    print("accuracy: {}, recall: {}, precision: {}, specificity: {}, true_positive: {}, false_positive: {}, false_negative: {}, true_negative: {}, categorical_accuracy: {}, possible_positives: {}".format(*metrics))

if __name__ == '__main__':
    main()
