import tensorflow as tf
import glob
import numpy as np
import cv2
import sys


data_task = 'train'
file_name = 'hands_%s.tfrecord' % data_task
writer = tf.python_io.TFRecordWriter(file_name)
data_dir = '/home/public/nfs70/rnn_sh/hands_data/%s/*/*.png' % data_task
img_path = glob.glob(data_dir)
height = 128
width = 128

def get_middle_str(str, start_str, end_str):
    start = str.find(start_str)
    start = start + len(start_str)
    end = str.find(end_str)
    return str[start:end].strip()

def load_image(img_name):
    img = cv2.imread(img_name)
    img = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)
    img = img.astype(np.uint8)
    return img

def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#shuffle the images
np.random.shuffle(img_path)
for iter in range(0, len(img_path)):
    if iter % 1000 == 0:
        print('%d / %d image have been processed !' % (iter, len(img_path)))
        sys.stdout.flush()
    img = load_image(img_path[iter])
    label = int(get_middle_str(img_path[iter], '%s/' % data_task, '/img'))
    feature2dict = {'image/encoded': _byte_feature(img.tostring()),
                'image/label': _int64_feature(label)}
    example = tf.train.Example(features=tf.train.Features(feature=feature2dict))
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()