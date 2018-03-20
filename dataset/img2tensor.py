import tensorflow as tf
import os


def load_img(data_dir):
    img_classes = []
    img_paths = []
    img_labels = []
    for sub_dir in os.listdir(data_dir):
        cur_dir = os.path.join(data_dir, sub_dir)
        if os.path.isdir(cur_dir):
            img_classes.append(sub_dir)
            for img_dir in os.listdir(cur_dir):
                if img_dir.endswith('png') or img_dir.endswith('jpg'):
                    img_paths.append(os.path.join(cur_dir, img_dir))
                    img_labels.append(img_classes.index(sub_dir))
    return img_classes, img_paths, img_labels
