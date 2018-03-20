import tensorflow as tf
import neuralnetwork.network as dnn

class hand_pose_m8(dnn.network):
    def setup(self): (
        self.feed('data')
        .conv(5, 5, 1, 1, 16, name='conv1', padding='SAME')
        .activate(name='relu1', activation='ReLU')
        .pool(2, 2, 2, 2, name='pool1', ptype_nn='MAX', padding='VALID')
        .conv(5, 5, 1, 1, 32, name='conv2', padding='VALID')
        .activate(name='relu2', activation='ReLU')
        .pool(2, 2, 2, 2, name='pool2', ptype_nn='MAX', padding='VALID')
        .conv(3, 3, 1, 1, 64, name='conv3', padding='VALID')
        .activate(name='relu3', activation='ReLU')
        .pool(2, 2, 2, 2, name='pool3', ptype_nn='MAX', padding='VALID')
        .conv(3, 3, 1, 1, 128, name='conv4', padding='VALID')
        .activate(name='relu4', activation='ReLU')
        .fc(1024, name='fc1')
        .activate(name='relu5', activation='ReLU')
        .fc(1024, name='fc2')
        .activate(name='relu6', activation='ReLU')
        .fc(128, name='fc3')
        .fc(69, name='fc4')
    )

def model_fn(features, labels, model):
    inputs = tf.reshape(features['x'], [-1, 128, 128, 3])
    inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])

    with tf.variable_scope('hmodel'):
        network_hand_skeleton = hand_pose_m8({'data': inputs})
        pose_predicted = network_hand_skeleton.layers['fc4']
    loss_pose = tf.reduce_mean(tf.squared_difference(x=labels, y=pose_predicted))

