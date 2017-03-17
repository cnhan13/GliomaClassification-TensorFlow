"""Routine for decoding the BRATS2015 binary file format."""

import os

from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf

# Global constants describing the BRATS data set
NUM_FILES_PER_ENTRY = 5
MRI_DIMS = 3
MHA_HEIGHT = 155
MHA_WIDTH = 240
MHA_DEPTH = 240
MHA_CHANNEL = 1

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5 # DON'T KNOW YET

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

def generate_mri_and_label_batch(mri, label, min_queue_examples,
                                 batch_size, shuffle):
  # Generate batch
  num_preprocess_threads = 8

  if shuffle:
    mris, label_batch = tf.train.shuffle_batch(
        [mri, label],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size,
        min_after_dequeue = min_queue_examples)
  else:
    mris, label_batch = tf.train.batch(
        [mri, label],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size)
  
  # Display the training mris in the visualizer. HOW?
  # tf.summary.image('images', images)

  return mris, tf.reshape(label_batch, [batch_size])


def inputs(data_dir, label_idx, batch_size):
  # Create a queue of filenames to read
  filenames = tf.train.match_filenames_once(data_dir + "*.in")
  filename_queue = tf.train.string_input_producer(filenames)

  read_input = read_brats(filename_queue, label_idx)

  casted_mri = tf.cast(read_input.mri, tf.float32)

  read_input.label.set_shape([1])

  # Ensure random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d BRATS records before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of 5-mri records and labels by building up a queue of records
  return generate_mri_and_label_batch(casted_mri, read_input.label,
                                      min_queue_examples, batch_size,
                                      shuffle=False)



