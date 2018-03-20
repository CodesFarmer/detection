import tensorflow as tf
import cv2

height = 128
width = 128
train_dataset = '/home/shenhui/program/python/detection/dataset/phonecall_train.tfrecord'
batch_size = 128
def parse_function_(serialized_example):
    feature = tf.parse_single_example(serialized_example,
                                      features={'image/encoded': tf.FixedLenFeature((), dtype=tf.string, default_value=''),
                                                'image/label': tf.FixedLenFeature((), dtype=tf.int64, default_value=0)})
    image = tf.decode_raw(feature['image/encoded'], out_type=tf.uint8)
    image = tf.reshape(image, [height, width, 3])
    image = tf.cast(image, dtype=tf.float32)/255.0 - 0.5
    label = tf.cast(feature['image/label'], dtype=tf.int32)
    return image, label

def train_input_fn():
    dataset = tf.contrib.data.TFRecordDataset(train_dataset)
    dataset = dataset.map(parse_function_, num_threads=1)
    dataset = dataset.shuffle(buffer_size=2048)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels