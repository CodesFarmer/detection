from six import string_types
import tensorflow as tf
import numpy as np

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def layer(op):
    def abstract_layer(self, *args, **kwargs):
        #get the name of current layer
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        #get the input of current layer
        if len(self.intermediate) == 0:
            raise ReferenceError('Empty input of layer %s', name)
        elif len(self.intermediate) == 1:
            nn_input = self.intermediate[0]
        else:
            nn_input = list(self.intermediate)
        #let the input pass the layer
        nn_output = op(self, nn_input, *args, **kwargs)
        print(name)
        print(nn_output.get_shape())
        #save the output into dict
        self.layers[name] = nn_output
        #set the intermediate as the output
        self.feed(nn_output)
        return self
    return abstract_layer

class network(object):
    def __init__(self, nn_input, trainable=True, data_format='NCHW', is_training=True):
        #set the input of the net
        self.inputs = nn_input
        #save the output of each layer
        self.layers = dict(nn_input)
        #Place the output of last layer
        self.intermediate = []
        #Set the network can be trained or not
        self.trainable = trainable
        #Data format of the network, NCHW or NHWC
        self.data_format = data_format
        #set the state of is_training for batch normalization
        self.is_training = is_training
        self.setup()
    def setup(self):
        raise NotImplementedError('The setup must be realized by the users')
    def get_unique_name(self, op_name):
        layer_id = len([prefix for prefix, _ in self.layers.items() if prefix.startswith(op_name)])
        return '%s%d' % (op_name, layer_id+1)
    def feed(self, *args):
        assert len(args) != 0
        self.intermediate = []
        for arg in args:
            if isinstance(arg, string_types):
                try:
                    arg = self.layers[arg]
                except:
                    raise KeyError('The network does not contain layer %s' % arg)
            self.intermediate.append(arg)
        return self
    #define the function for making variables
    def make_variables(self, name, shape, initializer='TRUNCATED'):
        if initializer.lower() == 'truncated':
            initialization = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
        elif initializer.lower() == 'zeros':
            initialization = tf.zeros(shape=shape)
        elif initializer.lower() == 'gamma':
            initialization = tf.random_gamma(shape=shape, alpha=1.5, beta=2.0)
        else:
            # raise RuntimeWarning('Initialization method %s does not support'%initializer)
            initialization = tf.random_normal(shape=shape, mean=0.0, stddev=0.1)
        return tf.get_variable(name=name, initializer=initialization, trainable=self.trainable)
    #convolutional layer
    @layer
    def conv(self, nn_input, k_h, k_w, s_h, s_w, out_channels, name, padding='VALID', initializer='GAUSSIAN'):
        #We get the depth of last feature map
        if self.data_format.lower() == 'nchw':
            in_channels = int(nn_input.get_shape()[1])
        else:
            in_channels = int(nn_input.get_shape()[-1])
        #We define the convolutional function
        convolue = lambda x, kernel: tf.nn.conv2d(x, kernel, [1, s_h, s_w, 1], padding=padding, data_format=self.data_format)
        with tf.variable_scope(name) as scope:
            #define the weights in convolutional layer
            weight = self.make_variables(name='weight', shape=[k_h, k_w, in_channels, out_channels], initializer=initializer)
            bias = self.make_variables(name='bias', shape=[1, out_channels, 1, 1])
            output = convolue(nn_input, weight)
            output = tf.add(output, bias)
            return output
    #batch normalization layer
    @layer
    def batch_norm(self, input_nn, name, is_training=True):
        with tf.variable_scope(name):
            output = tf.layers.batch_normalization(inputs=input_nn, axis=3,
                                                   momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                   center=True, scale=True, training=is_training, name=name, reuse=False, fused=True)
            return output
    #Activating function
    @layer
    def activate(self, input_nn, name, activation='ReLU'):
        with tf.variable_scope(name) as scope:
            if activation.lower() == 'relu':
                output = tf.nn.relu(input_nn, name=name)
                return output
            elif activation.lower() == 'sigmoid':
                output = tf.nn.sigmoid(input_nn, name=name)
                return output
            elif activation.lower() == 'prelu':
                i = int(input_nn.get_shape()[-1])
                alpha = self.make_variables('alpha', shape=(i,))
                output = tf.nn.relu(input_nn) + tf.multiply(alpha, tf.subtract(0.0, tf.nn.relu(tf.subtract(0.0, input_nn))))
                return output
            else:
                raise RuntimeError('Unknow activations: %s'%activation)
    #define the FC layer
    @layer
    def fc(self, input_nn, out_channels, name, initializer='GAUSSIAN'):
        #Get the input dimension of this layer
        in_shape = input_nn.get_shape().as_list()
        in_dimension = 1
        for num_dim in in_shape[1:]:
            in_dimension = in_dimension*int(num_dim)
        #add a fully connected layer
        with tf.variable_scope(name):
            weight = self.make_variables('weight', shape=[in_dimension, out_channels], initializer=initializer)
            bias = self.make_variables('bias', shape=[out_channels])
            #before multiple, we flat the matrix into vector
            featmap_flat = tf.reshape(input_nn, [-1, in_dimension])
            output = tf.add(tf.matmul(featmap_flat, weight), bias)
            return output
    #define the pooling layer
    @layer
    def pool(self, input_nn, k_h, k_w, s_h, s_w, name, ptype_nn='MAX', padding='SAME'):
        if self.data_format.lower() == 'nchw':
            kernel_size = [1, 1, k_h, k_w]
            kernel_stride = [1, 1, s_h, s_w]
        else:
            kernel_size = [1, k_h, k_w, 1]
            kernel_stride = [1, s_h, s_w, 1]
        with tf.variable_scope(name):
            if ptype_nn.lower() == 'max':
                output = tf.nn.max_pool(input_nn, ksize=kernel_size, strides=kernel_stride, padding=padding, data_format=self.data_format)
                return output
            elif ptype_nn.lower() == 'avg':
                output = tf.nn.avg_pool(input_nn, ksize=kernel_size, strides=kernel_stride, padding=padding, data_format=self.data_format)
                return output
            else:
                raise KeyError('Unknow pooling kernel %s'%ptype_nn)
    #define the dropout layer
    @layer
    def dropout(self, input_nn, keep_prob, name):
        with tf.variable_scope(name):
            output = tf.nn.dropout(input_nn, keep_prob=keep_prob)
            return output
    #Define a layer named reshape
    @layer
    def reshape(self, input_nn, shape, name):
        with tf.variable_scope(name):
            output = tf.reshape(input_nn, shape=shape, name=name)
            return output
    @layer
    def softmax(self, target, axis, name=None):
        softmax = tf.nn.softmax(target, axis=axis, name=name)
        return softmax
    def __batch_norm(self, input_nn, name, is_training=True):
        output = tf.layers.batch_normalization(inputs=input_nn, axis=3,
                                                momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                center=True, scale=True, training=is_training, fused=True)
        return output

    def __activate(self, input_nn, name, activation='PReLU'):
        with tf.variable_scope(name):
            if activation.lower() == 'relu':
                output = tf.nn.relu(input_nn, name=name)
                return output
            elif activation.lower() == 'sigmoid':
                output = tf.nn.sigmoid(input_nn, name=name)
                return output
            elif activation.lower() == 'prelu':
                i = int(input_nn.get_shape()[-1])
                alpha = self.make_variables('alpha', shape=(i,))
                output = tf.nn.relu(input_nn) + tf.multiply(alpha, tf.subtract(0.0, tf.nn.relu(tf.subtract(0.0, input_nn))))
                return output
            else:
                raise RuntimeError('Unknow activations: %s' % activation)
    @layer
    def mobile_unit(self, input_nn, outchannels, name, strides=[1, 1, 1, 1], padding='SAME', width_multiplier=1.0, min_depth=16, is_training=True):
        if self.data_format.lower() == 'nchw':
            in_dim_dwc = int(input_nn.get_shape()[1])
            strides = [strides[0], strides[3], strides[1], strides[2]]
        else:
            in_dim_dwc = int(input_nn.get_shape()[-1])
        ptw_filters = max(int(outchannels*width_multiplier), min_depth)
        conv1x1 = lambda x, w: tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format=self.data_format)
        with tf.variable_scope(name):
            kernel = self.make_variables(name='dw_kernel', shape=[3, 3, in_dim_dwc, 1])
            output = tf.nn.depthwise_conv2d(input=input_nn, filter=kernel, strides=strides, padding=padding, data_format=self.data_format)
            output = self.__batch_norm(output, name='bn1', is_training=is_training)
            output = self.__activate(output, name='relu1', activation='ReLU')
            in_dim_pwc = in_dim_dwc
            weight = self.make_variables(name='weight', shape=[1, 1, in_dim_pwc, ptw_filters])
            output = conv1x1(output, weight)
            output = self.__batch_norm(output, name='bn2', is_training=is_training)
            output = self.__activate(output, name='relu2', activation='ReLU')
            return output

    def load(self, sess, in_path):
        #load the model from frozen graph
        with tf.gfile.GFile(in_path, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        nodes = [node for node in graph_def.node]
        container = [ct for ct in nodes if ct.op == 'Const']
        for ct in container:
            with tf.variable_scope('', reuse=True):
                var = tf.get_variable(ct.name)
                sess.run(var.assign(ct.attr['value'].tensor))

    def save(self, sess, out_path='model/model_graph.pb'):
        #save the neural network
        vars = tf.all_variables()
        out_names = [var.name.split(':', 1)[0] for var in vars]
        in_graph_def = sess.graph.as_graph_def()
        # nodes = [node for node in in_graph_def.node]
        # for var in nodes:
        #     print(var.name)
        out_graph_def = tf.graph_util.convert_variables_to_constants(sess=sess,
                                                                     input_graph_def=in_graph_def,
                                                                     output_node_names=out_names)
        with tf.gfile.GFile(out_path, 'wb') as file:
            file.write(out_graph_def.SerializeToString())
    def freeze_model(self, sess, out_names, out_path='model/model_graph.pb'):
        #save the neural network
        # vars = tf.all_variables()
        # out_names = [var.name.split(':', 1)[0] for var in vars]
        in_graph_def = sess.graph.as_graph_def()
        out_graph_def = tf.graph_util.convert_variables_to_constants(sess=sess,
                                                                     input_graph_def=in_graph_def,
                                                                     output_node_names=out_names)
        with tf.gfile.GFile(out_path, 'wb') as file:
            file.write(out_graph_def.SerializeToString())