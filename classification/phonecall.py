import tensorflow as tf
import os
import cv2
import numpy as np
import neuralnetwork.network as dnn
import dataset.build_input_fun as input_fn


os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
_WEIGHT_DECAY = 4e-3
_MOMENTUM = 0.9

session_config = tf.ConfigProto()
session_config.gpu_options.per_process_gpu_memory_fraction = 0.5

class phonecall_net(dnn.network):
    def setup(self): (
        self.feed('data')
        .conv(3, 3, 1, 1, 16, name='conv1', padding='SAME')
        .activate(name='relu1', activation='ReLU')
        .pool(3, 3, 2, 2, name='pool1', ptype_nn='MAX', padding='SAME')
        .mobile_unit(32, name='mbn1_1', strides=[1, 2, 2, 1], padding='SAME', is_training=self.is_training)
        .mobile_unit(32, name='mbn1_2', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        .mobile_unit(32, name='mbn1_3', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        .pool(3, 3, 2, 2, name='pool2', ptype_nn='MAX', padding='SAME')
        .mobile_unit(64, name='mbn2_1', strides=[1, 2, 2, 1], padding='SAME', is_training=self.is_training)
        .mobile_unit(64, name='mbn2_2', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        .mobile_unit(64, name='mbn2_3', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        .mobile_unit(128, name='mbn3_1', strides=[1, 2, 2, 1], padding='SAME', is_training=self.is_training)
        .mobile_unit(128, name='mbn3_2', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        .mobile_unit(128, name='mbn3_3', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
        .fc(1024, name='fc1')
        .activate(name='relu2', activation='ReLU')
        .fc(256, name='fc2')
        .activate(name='relu3', activation='ReLU')
        .fc(32, name='fc3')
        .fc(2, name='fc4')
        # .softmax(axis=-1, name='prob')
    )


def model_fn(features, labels, mode):
    inputs = tf.reshape(features, [-1, 128, 128, 3])
    inputs = tf.cast(inputs, tf.float32)
    inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    logits = phonecall_net({'data': inputs}, is_training=is_training).layers['fc4']
    labels_ = tf.one_hot(labels, 2)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits))
    loss = cross_entropy + _WEIGHT_DECAY*tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    batch_size = 100
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_steps = tf.train.get_or_create_global_step()
        #set the learning rate
        initial_learning_rate = 0.0005
        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(500 * epoch) for epoch in [20, 30, 40]]
        values = [initial_learning_rate * decay for decay in [1, 1, 1, 1]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_steps, tf.int32), boundaries, values)
        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        #For comparision, we set the learning fixed
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM
        )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_steps)
        # train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    prediction = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels_, axis=1),
        tf.argmax(logits, axis=1)
    )
    metrics = {'accuracy': accuracy}
    #set the accuracy for plot figure
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss,
        predictions=prediction,
        eval_metric_ops=metrics
    )


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


run_config = tf.estimator.RunConfig(session_config=session_config).replace(save_checkpoints_secs=1e9)
cifar_classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='/home/shenhui/program/python/detection/model', config=run_config)
train_epochs = 20
# next_batch = input_fn.train_input_fn()
# sess = tf.Session()
# images = sess.run(next_batch)
# img = (images[0][0] + 0.5)*255
# img = img.astype(np.uint8)
# cv2.imshow('test', img)
# cv2.waitKey(0)
for _ in range(train_epochs // 1):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

    cifar_classifier.train(
        input_fn=train_input_fn)