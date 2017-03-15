
"""Routine for decoding the BRATS2014 binary file format."""

import os

from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

NUM_FILES_PER_ENTRY = 5
MRI_DIMS = 3
MHA_HEIGHT = 155
MHA_WIDTH = 240
MHA_DEPTH = 240

TRAIN_PATH = "/home/nhan/Desktop/x2goshared/BRATS2015/BRATS2015_Training/"
TEST_PATH = "/home/nhan/Desktop/x2goshared/BRATS2015/Testing/"

import skimage.io as sio

## Simple read mha to nparray, write nparray to file
## read array from file, reshape to get nparray

# func1
def read_mha(filename):
  #mha_data = io.imread(filename, plugin='simpleitk')
  m = sio.imread(filename[0], plugin='simpleitk')
  t = np.array([m])
  for i in xrange(1,5):
    p = sio.imread(filename[i], plugin='simpleitk')
    t = np.append(t, [p], axis=0)
  return t

# func2
def write_array(t, bin_path_name):
  f = open(bin_path_name, mode='wb')
  t.tofile(f) # order='C'
  f.close()
  print bin_path_name

# func3
def read_array_from_file(full_path_name):
  f = open(full_path_name, mode='rb')
  t = np.fromfile(f, dtype=np.int16)
  f.close()
  t = t.reshape((NUM_FILES_PER_ENTRY, \
                MHA_HEIGHT, \
                MHA_WIDTH, \
                MHA_DEPTH)) # order='C'
  return t

# func4
def view(v):
  plt.imshow(v[101,:,:], cmap='gray')
  plt.show()
  return

# func5
def generate_binary_input(path, train = True):
  if not train:
    print 'generate_binary_input() is not yet implemented for !train data'
    return

  bin_path = '' # path of binary input file
  bin_name = '' # name of binary input file
  path_file_list = 5*[None] # list of 5 mha files
  path_file_list_counter = 0  # counter tracking path_file_list
  for root, dirs, files in os.walk(path):
    
    if len(dirs)==5:
      pos = -1
      er = 0
      if 'LGG' in root:
        bin_name = 'L_'
        pos = root.rfind("LGG")
        er += 1
      if 'HGG' in root:
        bin_name = 'H_'
        pos = root.rfind("HGG")
        er += 1
      
      if pos==-1 or er != 1:
        print "error 1: " + root
        continue

      bin_name += (root[pos+4:] + '.in')
      bin_path = root[:pos]
      path_file_list = 5*[None]
      
    if not dirs:
      for f in files:
        if 'mha' in f:
          if 'T1c' in f:
            path_file_list[1] = root + '/' + f
          elif 'T1' in f:
            path_file_list[0] = root + '/' + f
          elif 'T2' in f:
            path_file_list[2] = root + '/' + f
          elif 'Flair' in f:
            path_file_list[3] = root + '/' + f
          elif 'OT' in f:
            path_file_list[4] = root + '/' + f
          else:
            print "error 2: " + root + '/' + f
          path_file_list_counter += 1

    if path_file_list_counter == 5:
      path_file_list_counter = a
      v = read_mha(path_file_list)
      write_array(v, bin_path + bin_name)
    
  return

def try_read():
  path = "/home/nhan/Desktop/x2goshared/BRATS2015/BRATS2015_Training/in"
  for root, dirs, files in os.walk(path):
    print str(len(files)) + " files"

  for i in range(10):
    t = read_array_from_file(root + "/" + files[i])
    print files[i] + ": ", t.shape

  return

def try_tfread():
  class MRIRecord(object):
    pass
  result = MRIRecord()

  path = "/home/nhan/Desktop/x2goshared/BRATS2015/BRATS2015_Training/trial_in/"
  
  filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(path + "*.in"))
  
  f_reader = tf.WholeFileReader()
  
  f_name, f_data = f_reader.read(filename_queue)

  f_decoded_data = tf.decode_raw(f_data, tf.int16)

  f_name_str = tf.string_split([f_name], '')

  f_name_uint8 = tf.decode_raw(f_name, tf.uint8)

  label_idx = len(path)

  f_label = f_name_uint8[label_idx]

  # if f_name[0] == 'L':
  #   result.label = 0
  # else:
  #   result.label = 1

  result = tf.reshape(f_decoded_data, [NUM_FILES_PER_ENTRY, MHA_HEIGHT, MHA_WIDTH, MHA_DEPTH])

  # Generate batch

  num_preprocess_threads = 1
  min_queue_examples = 256
  batch_size = 2
  results = tf.train.shuffle_batch(
    [result],
    batch_size = batch_size,
    num_threads = num_preprocess_threads,
    capacity = min_queue_examples + 3 * batch_size,
    min_after_dequeue = min_queue_examples)

  from tensorflow.python import debug as tf_debug

  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    record = result.eval()
    print record.shape
    print f_name
    print f_name_str.eval()
    print f_label.eval()


    print "Done"
    coord.request_stop()
    coord.join(threads)
    sess.close()

  return result

def inference(data):
  # Layers
  # h1 = tf.nn.conv3d(f_data1, f_labels)
  # h2 = tf.nn.conv3d(f_data2, f_labels)
  # h3 = tf.nn.conv3d(f_data3, f_labels)
  # h4 = tf.nn.conv3d(f_data4, f_labels)
  # w1 = tf.nn.pool3d(h1, f_labels)
  # w2 = tf.nn.pool3d(h2, f_labels)
  # w3 = tf.nn.pool3d(h3, f_labels)
  # w4 = tf.nn.pool3d(h4, f_labels)
  # hh1 = tf.nn.conv3d(w1, f_labels)
  # hh2 = tf.nn.conv3d(w2, f_labels)
  # hh3 = tf.nn.conv3d(w3, f_labels)
  # hh4 = tf.nn.conv3d(w4, f_labels)
  # ww1 = tf.nn.pool3d(hh1, f_labels)
  # ww2 = tf.nn.pool3d(hh2, f_labels)
  # ww3 = tf.nn.pool3d(hh3, f_labels)
  # ww4 = tf.nn.pool3d(hh4, f_labels)

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 1, 64], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv3d(f_data, kernel, [1, 1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
  
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 64, 64], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv3d(norm1, kernel, [1, 1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # pool2
    pool2 = tf.nn.max_pool3d(norm2, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer (WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights) + biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


  
if __name__ == "__main__":
  # f = ["VSD.Brain.XX.O.MR_T1.35536.mha", \
  #     "VSD.Brain.XX.O.MR_T1c.35535.mha", \
  #     "VSD.Brain.XX.O.MR_T2.35534.mha", \
  #     "VSD.Brain.XX.O.MR_Flair.35533.mha", \
  #     "VSD.Brain_3more.XX.O.OT.42283.mha"]
  # v = read_mha(f)
  # f2 = "data_105"
  # write_array(v, f2)
  # t = read_array_from_file(f2)

  # generate_binary_input(TRAIN_PATH)
  # generate_binary_input(TEST_PATH, False)
  print try_tfread()
