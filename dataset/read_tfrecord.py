import tensorflow as tf
import os
import cv2


os.environ['CUDA_VISIBLE_DEVICES'] = ''

data_task = 'train'
file_name = 'hands_%s.tfrecord' % data_task
height = 128
width = 128

with tf.Session() as sess:
    feature2dict = {'image/encoded': tf.FixedLenFeature((), dtype=tf.string, default_value=''),
                    'image/label': tf.FixedLenFeature((), dtype=tf.int64, default_value=0)}
    filename_queue = tf.train.string_input_producer([file_name], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature2dict)
    #get the image
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [height, width, 3])
    #get the label
    label = tf.cast(features['image/label'], tf.int32)
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10,
                                            capacity=30, num_threads=1, min_after_dequeue=10)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # for batch_index in range(100):
    num_all = 0
    while True:
        try:
            img, lbl = sess.run([images, labels])
        except:
            raise EOFError('The data is out of max')
        cv2.imshow('test', img[0])
        cv2.waitKey(1)
        num_all += 1
        print(num_all*10)

    coord.request_stop()

    coord.join(threads)
    sess.close()
