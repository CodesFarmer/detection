import h5py as hdf5
import numpy as np
import tensorflow as tf
import os
import neuralnetwork.network as dnn

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
_WEIGHT_DECAY = 4e-3
_MOMENTUM = 0.9

session_config = tf.ConfigProto()
session_config.gpu_options.per_process_gpu_memory_fraction = 0.2
# is_training = tf.constant(True, dtype=tf.bool)
class cifar_nn(dnn.network):
    def setup(self):
        (
            self.feed('data')
            .conv(5, 5, 1, 1, 32, name='conv1', padding='SAME')
            .pool(3, 3, 2, 2, name='pool1', padding='VALID')
            .activate(name='relu1', activation='ReLU')
            .conv(5, 5, 1, 1, 32, name='conv2', padding='SAME')
            .activate(name='relu2', activation='ReLU')
            .pool(3, 3, 2, 2, name='pool2', ptype_nn='AVG', padding='VALID')
            .conv(5, 5, 1, 1, 64, name='conv4', padding='SAME')
            .activate(name='relu3', activation='ReLU')
            .pool(3, 3, 2, 2, name='pool3', ptype_nn='AVG', padding='VALID')
            .fc(64, name='fc1')
            .fc(10, name='fc2')
        )

class cifar_mn(dnn.network):
    def setup(self):
        (
            self.feed('data')
            .conv(5, 5, 1, 1, 32, name='conv1', padding='SAME')
            .activate(name='relu1', activation='ReLU')
            .pool(3, 3, 2, 2, name='pool1', padding='SAME')
            .mobile_unit(outchannels=32, name='mbn1', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
            .mobile_unit(outchannels=32, name='mbn2', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
            .mobile_unit(outchannels=32, name='mbn3', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
            .mobile_unit(outchannels=32, name='mbn4', strides=[1, 2, 2, 1], padding='SAME', is_training=self.is_training)
            .mobile_unit(outchannels=64, name='mbn5', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
            .mobile_unit(outchannels=64, name='mbn6', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
            .mobile_unit(outchannels=64, name='mbn7', strides=[1, 1, 1, 1], padding='SAME', is_training=self.is_training)
            .mobile_unit(outchannels=64, name='mbn8', strides=[1, 2, 2, 1], padding='SAME', is_training=self.is_training)
            .fc(64, name='fc1')
            .fc(10, name='fc2')
        )

def model_fn(features, labels, mode):
    inputs = tf.reshape(features['x'], [-1, 32, 32, 3])
    inputs = tf.cast(inputs, tf.float32)
    inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
    # logits = cifar_nn({'data' : inputs}).layers['fc2']
    # cifar_nn.training = True
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    logits = cifar_mn({'data': inputs}, is_training=is_training).layers['fc2']
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
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
        tf.argmax(labels, axis=1),
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

run_config = tf.estimator.RunConfig(session_config=session_config).replace(save_checkpoints_secs=1e9)
cifar_classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='/home/shenhui/program/python/detection/model', config=run_config)
train_epochs = 40
h5file = hdf5.File('/tmp/cifar10_data/CIFAR-10.hdf5')
train_data = h5file['/normalized_full/training/default']
train_data = train_data.value[0, :, :, :, :]
tr_label = h5file['/normalized_full/training/targets']
train_label = np.zeros(shape=[tr_label.shape[1], 10])
for tl in range(50000):
    train_label[tl, tr_label.value[0, tl, 0]] = 1
test_data = h5file['/normalized_full/test/default']
test_data = test_data.value[0, :, :, :, :]
te_label = h5file['/normalized_full/test/targets']
test_label = np.zeros(shape=[te_label.shape[1], 10])
for tl in range(10000):
    test_label[tl, te_label.value[0, tl, 0]] = 1

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': np.array(train_data)},
    y=np.array(train_label),
    batch_size=100,
    num_epochs=None,
    shuffle=True
)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': np.array(test_data)},
    y=np.array(test_label),
    batch_size=100,
    num_epochs=1,
    shuffle=True
)
for _ in range(train_epochs // 1):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

    cifar_classifier.train(
        input_fn=train_input_fn,
        steps=5000)

    # Evaluate the model and print results
    eval_results = cifar_classifier.evaluate(
        input_fn=test_input_fn)
    print(eval_results)