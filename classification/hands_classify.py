import tensorflow as tf
import numpy as np
import os
import cv2

import neuralnetwork.network as dnn


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

_WEIGHT_DECAY = 5e-4
is_training = tf.placeholder(dtype=tf.bool, name='istraining')
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
    # filename_queue = tf.train.string_input_producer([file_name], num_epochs=num_epoch)
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature2dict)
    #get the image
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [height, width, 3])
    #get the label
    label = tf.cast(features['image/label'], tf.int32)
    # images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size,
    #                                         capacity=batch_size*3, num_threads=1, min_after_dequeue=batch_size)
    # return images, labels
    return image, label

class hand_net(dnn.network):
    def setup(self): (
        self.feed('data')
        .conv(3, 3, 1, 1, 16, name='conv1', padding='VALID')
        .activate(name='relu1', activation='ReLU')
        .pool(3, 3, 2, 2, name='pool1', ptype_nn='MAX', padding='VALID')
        .mobile_unit(outchannels=32, name='mbn1', strides=[1, 2, 2, 1], padding='SAME', is_training=is_training)
        .mobile_unit(outchannels=32, name='mbn2', strides=[1, 1, 1, 1], padding='SAME', is_training=is_training)
        .mobile_unit(outchannels=64, name='mbn3', strides=[1, 2, 2, 1], padding='SAME', is_training=is_training)
        .mobile_unit(outchannels=64, name='mbn4', strides=[1, 1, 1, 1], padding='SAME', is_training=is_training)
        .mobile_unit(outchannels=256, name='mbn5', strides=[1, 2, 2, 1], padding='SAME', is_training=is_training)
        .mobile_unit(outchannels=256, name='mbn6', strides=[1, 1, 1, 1], padding='SAME', is_training=is_training)
        .mobile_unit(outchannels=512, name='mbn7', strides=[1, 2, 2, 1], padding='SAME', is_training=is_training)
        .mobile_unit(outchannels=512, name='mbn8', strides=[1, 1, 1, 1], padding='SAME', is_training=is_training)
        .fc(512, name='fc1')
        .activate(name='relu2', activation='ReLU')
        .fc(128, name='fc2')
        .activate(name='relu3', activation='ReLU')
        .fc(27, name='fc3')
        .softmax(1, name='prob')
    )


annotation = tf.placeholder(dtype=tf.int32, shape=[None, 1])
inputs = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
inputs_ = tf.transpose(inputs, [0, 3, 1, 2])
annotation_ = tf.one_hot(annotation, 27)
#there are two part of output: probability and regression
with tf.variable_scope('handnet'):
    neuralnetwork = hand_net({'data': inputs_}, data_format='NCHW')

logits = neuralnetwork.layers['prob']
loss_classify = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=annotation_, logits=logits))
loss_regular = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables('handnet')])
loss = loss_classify + loss_classify*_WEIGHT_DECAY
# loss = loss_classify

trainer = tf.train.GradientDescentOptimizer(learning_rate)
training = trainer.minimize(loss)

accuracy = tf.metrics.accuracy(
        tf.argmax(logits, axis=-1),
        tf.argmax(annotation_, axis=-1)
)

tf.identity(loss_classify, 'loss_classify')
loss_classify_sum = tf.summary.scalar('loss_classify', loss_classify)
tf.identity(loss_regular, 'loss_regular')
loss_regular_sum = tf.summary.scalar('loss_regular', loss_regular)
tf.identity(loss, 'total_loss')
total_loss_sum = tf.summary.scalar('total_loss', loss)
tf.identity(accuracy[1], 'accuracy')
accuracy_sum = tf.summary.scalar('accuracy', accuracy[1])
tf.identity(learning_rate, 'learning_rate')
tf.summary.scalar('learning_rate', learning_rate)
merged_train = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/home/shenhui/program/python/detection/data/train_dir/train')
merged_test = tf.summary.merge([loss_classify_sum, loss_regular_sum, total_loss_sum, accuracy_sum])
test_writer = tf.summary.FileWriter('/home/shenhui/program/python/detection/data/train_dir/test')

#set the training and test dataset
train_dataset = tf.contrib.data.TFRecordDataset([train_file_name])
train_dataset = train_dataset.map(_parse_function)
train_dataset = train_dataset.shuffle(buffer_size=50000)
train_dataset = train_dataset.batch(batch_size)
validation_dataset = tf.contrib.data.TFRecordDataset([validation_file_name])
validation_dataset = validation_dataset.map(_parse_function)
validation_dataset = validation_dataset.shuffle(buffer_size=50000)
validation_dataset = validation_dataset.batch(batch_size)

handle = tf.placeholder(dtype=tf.string, shape=[])
iterator = tf.contrib.data.Iterator.from_string_handle(handle,
                                                       train_dataset.output_types,
                                                       train_dataset.output_shapes)
next_batch = iterator.get_next()
train_iterator = train_dataset.make_initializable_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    train_handle = sess.run(train_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # for batch_index in range(100):
    num_eh = 0
    # while num_step < np.floor(52366/batch_size)*num_epoch:
    while num_eh < (num_epoch+1):
        sess.run(train_iterator.initializer)
        sess.run(validation_iterator.initializer)
        num_step = 1
        max_steps = np.floor(52366 / batch_size) + 1
        while num_step < max_steps:
            img, lbl = sess.run(next_batch, feed_dict={handle: train_handle})
            lbl = lbl - 1
            lbl = np.expand_dims(lbl, axis=1)
            # one_hot = sess.run(annotation_, feed_dict={annotation: lbl})
            # print(lbl[0])
            # print(one_hot[0])
            lr = init_lr*10/(10*np.ceil((num_eh+1)/20))
            summary, _ = sess.run([merged_train, training], feed_dict={inputs: img, annotation: lbl, is_training: True,
                                                                 learning_rate: lr})
            train_writer.add_summary(summary, num_step + num_eh*max_steps)
            num_step += 1
        for test_index in range(100):
            img, lbl = sess.run(next_batch, feed_dict={handle: validation_handle})
            lbl = np.expand_dims(lbl, axis=1)
            summary, acy = sess.run([merged_test, accuracy], feed_dict={inputs: img, annotation: lbl, is_training: False})
            test_writer.add_summary(summary, test_index*4 + num_eh*max_steps)
        neuralnetwork.freeze_model(sess, ['handnet/prob'],
                                   out_path='/home/shenhui/program/python/detection/data/model/model_%05d.pb' % num_eh)
        print('%d / %d : %f' % (num_eh, num_epoch, acy[1]))
        num_eh += 1

    coord.request_stop()

    coord.join(threads)
    sess.close()
