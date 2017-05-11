from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import sys

import p5c1_train_deeper as p5c1

FLAGS = tf.app.flags.FLAGS
### audi ###
tf.app.flags.DEFINE_string('eval_dir',
                           'eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('num_examples', 50,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op,
              summary_op, keep_prob, labels):
  with tf.Session() as sess:
    class CntObj(object):
      pass
    cnt = CntObj

    # handle num_epochs in tf.string_input_producer
    sess.run(tf.local_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(evaluate.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      
      saver.restore(sess, ckpt.model_checkpoint_path)

      # /my-path/train1/model.ckpt-0,
      # extract global_step from it.
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
      cnt.true = 0
      cnt.total_sample = num_iter * p5c1.FLAGS.batch_size
      cnt.true_high = 0
      cnt.high = 0
      cnt.true_low = 0
      cnt.low = 0
      step = 0
      while step < num_iter and not coord.should_stop():
        #predictions = sess.run([top_k_op], feed_dict={keep_prob: 1.0})
        result = sess.run([top_k_op, labels], feed_dict={keep_prob: 1.0})

        cnt.true += np.sum(result[0]) # predictions
        high_count = np.sum(result[1])
        cnt.high += high_count
        cnt.low += p5c1.FLAGS.batch_size - high_count
        cnt.true_high += np.sum(result[0] * result[1])
        cnt.true_low += np.sum(result[0] * (1 - result[1]))
        step += 1

      # Compute @ 1
      accuracy = 1.0 * cnt.true / cnt.total_sample
      recall = 1.0 * cnt.true_high / cnt.high
      specificity = 1.0 * cnt.true_low / cnt.low
      precision = 1.0 * cnt.true_high / (cnt.true_high + cnt.low - cnt.true_low)
      f1_score = 2.0 * precision * recall / (precision + recall)
      print('%s' % (datetime.now()))
      print('\tAccuracy @ 1 = %.5f | True count = %d | Sample count = %d' %
          (accuracy, cnt.true, cnt.total_sample))
      print('\tRecall = %.5f | True high count = %d | High count = %d' %
          (recall, cnt.true_high, cnt.high))
      print('\tSpecificity = %.5f | True low count = %d | Low count = %d' %
          (specificity, cnt.true_low, cnt.low))
      print('\tPrecision = %.5f' % precision)
      print('\tF1 score = %.5f' % f1_score)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op, feed_dict={keep_prob: 1.0}))
      summary.value.add(tag='Accuracy', simple_value=accuracy)
      summary.value.add(tag='Recall', simple_value=recall)
      summary.value.add(tag='Specificity', simple_value=specificity)
      summary.value.add(tag='Precision', simple_value=precision)
      summary.value.add(tag='f1_score', simple_value=f1_score)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  return


def evaluate(is_tumor_cropped=False):
  """Eval BRATS for a number of steps."""
  with tf.Graph().as_default() as g:
    class SumObj(object):
      pass
    sm = SumObj()
    
    class AvgObj(object):
      pass
    avg = AvgObj()

    # get records and labels for BRATS
    records, labels = p5c1.inputs(is_tumor_cropped,
                                  is_train_list=False,
                                  batch_size=p5c1.FLAGS.batch_size,
                                  set_number=evaluate.set_number)
    keep_prob = tf.placeholder(tf.float32)
    logits = p5c1.inference(records, keep_prob)
    #logits = p5c1.deb(logits, "eval-logits")
    #labels = p5c1.deb(labels, "eval-labels")

    # calculate predictions
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    #top_k_op = p5c1.deb(top_k_op, "eval-top_k_op")
    
    # restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        p5c1.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore) 
    # build the summary operation based on the TF collection of summaries
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(evaluate.eval_dir, g)
    
    eval_once(saver, summary_writer, top_k_op, summary_op, keep_prob, labels)
    

def main(argv=None):
  
  evaluate.set_number = sys.argv[1]
  is_tumor_cropped = (sys.argv[2] == '1')
  model_id = sys.argv[3]

  evaluate.eval_dir = p5c1.FLAGS.common_dir
  evaluate.eval_dir += p5c1.FLAGS.tumor_dir if is_tumor_cropped else p5c1.FLAGS.brain_dir
  evaluate.checkpoint_dir = evaluate.eval_dir + p5c1.FLAGS.train_dir \
                          + evaluate.set_number + "_" + model_id
  evaluate.eval_dir += FLAGS.eval_dir + evaluate.set_number \
                      + "_" + model_id

  tf.gfile.MakeDirs(evaluate.eval_dir)

  evaluate(is_tumor_cropped=is_tumor_cropped)

if __name__ == '__main__':
  tf.app.run()
