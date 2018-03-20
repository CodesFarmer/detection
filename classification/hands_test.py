import tensorflow as tf
import os
import numpy as np
import neuralnetwork.network as dnn


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# is_training = tf.placeholder(dtype=tf.bool, name='istraining')

model_dir = '/home/shenhui/program/python/detection/data/model/model_00040.pb'
data_task = 'train'
data_dir = '/home/shenhui/program/dataset/unity_hand'
train_file_name = '%s/hands_train.tfrecord' % data_dir
validation_file_name = '%s/hands_validation.tfrecord' % data_dir
height = 128
width = 128
batch_size = 128
num_epoch = 100
init_lr = 0.1

def _parse_function(serialized_example):
    feature2dict = {'image/encoded': tf.FixedLenFeature((), dtype=tf.string, default_value=''),
                    'image/label': tf.FixedLenFeature((), dtype=tf.int64, default_value=0)}
    features = tf.parse_single_example(serialized_example, features=feature2dict)
    #get the image
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [height, width, 3])
    #get the label
    label = tf.cast(features['image/label'], tf.int32)
    return image, label

def load_freeze_model(model_path):
    with tf.gfile.GFile(model_path, 'rb') as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())
        # sess.graph.as_default()
    with tf.Graph().as_default() as graph:
        # tf.import_graph_def(graph_def, input_map={'inputs:0': input_nn}, name='')
        tf.import_graph_def(graph_def, name='')
    return graph
graph = load_freeze_model(model_dir)
inputs = graph.get_tensor_by_name('Placeholder_1:0')
is_training = graph.get_tensor_by_name('istraining:0')
prob = graph.get_tensor_by_name('handnet/prob:0')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=graph) as sess:
    # load the model
    # def load_freeze_model(sess, model_path, input_nn):
    annotation = tf.placeholder(dtype=tf.int32, shape=[None, 1])
    annotation_ = tf.one_hot(annotation, 27)
    accuracy = tf.metrics.accuracy(
            tf.argmax(prob, axis=-1),
            tf.argmax(annotation_, axis=-1)
    )

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # set the training and test dataset
    validation_dataset = tf.contrib.data.TFRecordDataset([train_file_name])
    validation_dataset = validation_dataset.map(_parse_function)
    validation_dataset = validation_dataset.shuffle(buffer_size=10000)
    validation_dataset = validation_dataset.batch(batch_size)
    handle = tf.placeholder(dtype=tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(handle,
                                                           validation_dataset.output_types,
                                                           validation_dataset.output_shapes)
    next_batch = iterator.get_next()
    validation_iterator = validation_dataset.make_initializable_iterator()
    validation_handle = sess.run(validation_iterator.string_handle())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # for batch_index in range(100):
    num_eh = 0
    # while num_step < np.floor(52366/batch_size)*num_epoch:
    while num_eh < (num_epoch+1):
        sess.run(validation_iterator.initializer)
        for test_index in range(100):
            img, lbl = sess.run(next_batch, feed_dict={handle: validation_handle})
            lbl = np.expand_dims(lbl, axis=1)
            prediction = sess.run(accuracy, feed_dict={inputs: img, annotation:lbl, is_training: False})
            print(prediction[1])
        # print('%d / %d : %f' % (num_eh, num_epoch, acy[1]))
        num_eh += 1

    coord.request_stop()

    coord.join(threads)
    sess.close()