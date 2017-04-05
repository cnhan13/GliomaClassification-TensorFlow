from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import sys

import p5c1_brats_train as p5c1

FLAGS = tf.app.flags.FLAGS
### audi ###
tf.app.flags.DEFINE_string('eval_dir',
                           'eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 10,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 200,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
  
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(evaluate.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      
      saver.restore(sess, ckpt.model_checkpoint_path)

      # /my-path/train1/model.ckpt-0,
      # extract global_step fro mit.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / p5c1.FLAGS.batch_size))
      true_count = 0
      total_sample_count = num_iter * p5c1.FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        top_k_op = p5c1.deb(top_k_op, "top_k_op")
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.5f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(is_tumor_cropped=False):
  """Eval BRATS for a number of steps."""
  with tf.Graph().as_default() as g:
    # get records and labels for BRATS
    eval_data = FLAGS.eval_data == 'test'
    records, labels = p5c1.inputs(is_tumor_cropped,
                                  is_train_list=False,
                                  batch_size=p5c1.FLAGS.batch_size,
                                  set_number=evaluate.set_number)

    logits = p5c1.inference(records)
    labels = p5c1.deb(labels, "labels")
    logits = p5c1.deb(logits, "logits")

    # calculate predictions
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    
    # restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        p5c1.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # build the summary operation based on the TF collection of summaries
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(evaluate.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):
  
  evaluate.set_number = sys.argv[1]
  is_tumor_cropped = (sys.argv[2] == 0)
  evaluate.eval_dir = p5c1.FLAGS.common_dir
  evaluate.eval_dir += p5c1.FLAGS.tumor_dir if is_tumor_cropped else p5c1.FLAGS.brain_dir
  evaluate.checkpoint_dir = evaluate.eval_dir + p5c1.FLAGS.train_dir + evaluate.set_number
  evaluate.eval_dir += FLAGS.eval_dir + evaluate.set_number

  if tf.gfile.Exists(evaluate.eval_dir):
    tf.gfile.DeleteRecursively(evaluate.eval_dir)
  tf.gfile.MakeDirs(evaluate.eval_dir)
  evaluate(is_tumor_cropped=is_tumor_cropped)

if __name__ == '__main__':
  tf.app.run()
