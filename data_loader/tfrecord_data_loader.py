from base.base_data_loader import BaseDataLoader
import tensorflow as tf

class TfrecordDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(TfrecordDataLoader, self).__init__(config)
        (self.image_train, self.label_train) = self._create_dataset(config.loader.training_records, config)
        (self.image_test, self.label_test) = self._create_dataset(config.loader.test_records, config)

    def get_train_data(self):
        return self.image_train, self.label_train
    
    def get_test_data(self):
        return self.image_test, self.label_test
        
    def _parse_function(self, proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {
            'train/image': tf.FixedLenFeature([], tf.string),
            "train/label": tf.FixedLenFeature([], tf.int64)
        }
        
        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)
        
        # Turn your saved image string into an array

        image = tf.image.decode_jpeg(parsed_features['train/image'], channels = 3)
        image = tf.image.resize_image_with_crop_or_pad(image, 200, 200)
        image  = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image.set_shape((200, 200, 3))

        return image, parsed_features["train/label"]

    def _create_dataset(self, filepath, config):
        
        # This works with arrays as well
        dataset = tf.data.TFRecordDataset(filepath)
        
        # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
        dataset = dataset.map(self._parse_function, num_parallel_calls=8)
        
        # This dataset will go on forever
        dataset = dataset.repeat()
        
        # Set the number of datapoints you want to load and shuffle 
        dataset = dataset.shuffle(config.loader.shuffle_buffer)
        
        # Set the batchsize
        dataset = dataset.batch(config.loader.batch_size)
        
        # Create an iterator
        iterator = dataset.make_one_shot_iterator()
        
        # Create your tf representation of the iterator
        image, label = iterator.get_next()

        # Bring your picture back in shape
        image = tf.reshape(image, config.loader.image_size)
        
        # Create a one hot array for your labels
        label = tf.one_hot(label, config.loader.num_classes)
        
        return image, label
