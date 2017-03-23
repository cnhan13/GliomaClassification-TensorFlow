"""Routine for decoding the BRATS2015 binary file format."""

import os

from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import skimage.io as sio


# Global constants describing the BRATS data set
NUM_FILES_PER_ENTRY = 5
MRI_DIMS = 3
MHA_HEIGHT = 155
MHA_WIDTH = 240
MHA_DEPTH = 240
MHA_CHANNEL = 1

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 20
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10 # DON'T KNOW YET

TRAIN_PATH = "/home/nhan/Desktop/x2goshared/BRATS2015/BRATS2015_Training/"
TEST_PATH = "/home/nhan/Desktop/x2goshared/BRATS2015/Testing/"

def _const1(): return tf.constant([1])
def _const4(): return tf.constant([4])

def read_brats(filename_queue, label_idx):
  class BRATSRecord(object):
    pass
  result = BRATSRecord()

  reader = tf.WholeFileReader()
  f_name_reader, f_raw_reader = reader.read(filename_queue)

  f_data = tf.decode_raw(f_raw_reader, tf.int16)

  f_name = tf.string_split([f_name_reader], '')
  f_name_uint8 = tf.decode_raw(f_name_reader, tf.uint8)

  _H_72 = tf.constant(72, dtype=tf.uint8)
  compare_label = tf.equal(f_name_uint8[label_idx], _H_72)

  result.label = tf.cond(compare_label, _const4, _const1)

  result.mri = tf.reshape(f_data, [NUM_FILES_PER_ENTRY,
                                   MHA_HEIGHT,
                                   MHA_WIDTH,
                                   MHA_DEPTH,
                                   MHA_CHANNEL])
  return result


def brats_reader(directory):

  filenames = ["/t1.mha", "/t1c.mha", "/t2.mha", "/flair.mha", "/ot.mha"]
  m = sio.imread(directory + filenames[0], plugin='simpleitk')
  t = np.array([m])
  for i in xrange(1,5):
    p = sio.imread(directory + filenames[i], plugin='simpleitk')
    t = np.append(t, [p], axis=0)
  
  return t 


def read_brats_dev(directories_queue, label_idx):
  class BRATSRecord(object):
    pass
  result = BRATSRecord()
  
  directory = directories_queue.dequeue()

  mris = tf.py_func(brats_reader, [directory], tf.int16)[0]
  print mris
  
  result.mris = tf.reshape(mris, [NUM_FILES_PER_ENTRY,
                                  MHA_HEIGHT,
                                  MHA_WIDTH,
                                  MHA_DEPTH,
                                  MHA_CHANNEL])

  directory_uint8 = tf.decode_raw(directory, tf.uint8)
  
  _H_72 = tf.constant(72, dtype=tf.uint8) # ascii of 'H'

  compare_op = tf.equal(directory_uint8[45], _H_72)
  
  result.label = tf.cond(compare_op, _const4, _const1)

  return result


def generate_record_and_label_batch(mris, label, min_queue_examples,
                                 batch_size, shuffle):
  # Generate batch
  num_preprocess_threads = 8

  if shuffle:
    records, label_batch = tf.train.shuffle_batch(
        [mris, label],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size,
        min_after_dequeue = min_queue_examples)
  else:
    records, label_batch = tf.train.batch(
        [mris, label],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size)
  
  # Display the training mris in the visualizer. HOW?
  # tf.summary.image('images', images)

  return records, tf.reshape(label_batch, [batch_size])


def inputs(data_dir, label_idx, batch_size):
  ## Create a queue of filenames to read
  #filenames = tf.train.match_filenames_once(data_dir + "*.in")
  #filename_queue = tf.train.string_input_producer(filenames)

  #read_input = read_brats(filename_queue, label_idx)

  # Create a queue of directories to read
  directories = tf.train.match_filenames_once(data_dir + "*brats*")
  directories_queue = tf.train.string_input_producer(directories)

  read_input = read_brats_dev(directories_queue, label_idx)

  casted_mris = tf.cast(read_input.mris, tf.float32)

  read_input.label.set_shape([1])

  # Ensure random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d BRATS records before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of 5-mri records and labels by building up a queue of records
  return generate_record_and_label_batch(casted_mris, read_input.label,
                                      min_queue_examples, batch_size,
                                      shuffle=False)


