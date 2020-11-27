"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record

    for bdd:
    # create train tfrecord for seperate file
    # create test tfrecord for seperate file
    # create train tfrecord for all file
    # create test tfrecord for all file

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow.compat.v1 as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
import csv


def Detection2020_Train_CSV_To_TF():
    TFRecordOutputDir ='C:/Users/Administrator.WIN-2EPKD7D6018/PycharmProjects/json2csv/tfrecord/Detection2020/'
    csvInputPath='C:/Users/Administrator.WIN-2EPKD7D6018/PycharmProjects/json2csv/outputForDetection2020/Train_Detection2020.csv'
    currTFRecordOutputDir=TFRecordOutputDir+'Detection_2020_Train.tfrecord'
    writer = tf.python_io.TFRecordWriter(currTFRecordOutputDir)
    path = os.path.join('')
    currCsv = csvInputPath
    examples = pd.read_csv(currCsv)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords for: Train_Detection2020.csv')

def Detection2020_Test_CSV_To_TF():
    TFRecordOutputDir ='C:/Users/Administrator.WIN-2EPKD7D6018/PycharmProjects/json2csv/tfrecord/Detection2020/'
    csvInputPath='C:/Users/Administrator.WIN-2EPKD7D6018/PycharmProjects/json2csv/outputForDetection2020/Test_Detection2020.csv'
    currTFRecordOutputDir=TFRecordOutputDir+'Detection_2020_Test.tfrecord'
    writer = tf.python_io.TFRecordWriter(currTFRecordOutputDir)
    path = os.path.join('')
    currCsv = csvInputPath
    examples = pd.read_csv(currCsv)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords for: Test_Detection2020.csv')


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'bicycle':
        return 1
    elif row_label == 'bus':
        return 2
    elif row_label == 'car':
        return 3
    elif row_label == 'Go':
        return 4
    elif row_label == 'motorcycle':
        return 5
    elif row_label == 'pedestrian':
        return 6
    elif row_label == 'rider':
        return 7
    elif row_label == 'Slow Down':
        return 8
    elif row_label == 'Stop':
        return 9
    elif row_label == 'traffic light':
        return 10
    elif row_label == 'traffic sign':
        return 11
    elif row_label == 'truck':
        return 12
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    Detection2020_Train_CSV_To_TF()
    #Detection2020_Test_CSV_To_TF()


if __name__ == '__main__':
    tf.app.run()