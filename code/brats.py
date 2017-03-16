import tensorflow as tf

import brats_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 20, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir',
                          '/home/nhan/Desktop/x2goshared/BRATS2015/BRATS2015_Training/trial_in/',
                          """Path to the BRATS *.in files.""")

def inputs():
  
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')

  mris, labels = brats_input.inputs(data_dir=FLAGS.data_dir,
                                             label_idx=len(FLAGS.data_dir),
                                             batch_size=FLAGS.batch_size)
  return mris, labels


def inference(mris):
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
 
  
