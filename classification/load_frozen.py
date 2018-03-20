import tensorflow as tf
import os


os.environ['CUDA_VISIBLE_DEVICES'] = ''

model_dir = '/home/shenhui/program/python/detection/data/model/model_00040.pb'
# load the model
# def load_freeze_model(sess, model_path, input_nn):
def load_freeze_model(sess, model_path):
    with tf.gfile.GFile(model_path, 'rb') as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())
        # sess.graph.as_default()
    with tf.Graph().as_default() as graph:
        # tf.import_graph_def(graph_def, input_map={'inputs:0': input_nn}, name='')
        tf.import_graph_def(graph_def, name='')
    return graph

sess = tf.Session()
graph = load_freeze_model(sess, model_dir)
# for op in graph.get_operations():
#     print(op.values())

inputs = graph.get_tensor_by_name('Placeholder_1:0')
# is_training = graph.get_tensor_by_name('handnet/istraining:0')