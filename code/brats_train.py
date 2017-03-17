import tensorflow as tf

import brats

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './brats_train',
                           """Directory where to write event logs"""
                           """and checkpoints.""")
tf.app.flags.DEFINE_integer('max_steps', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")



def train():
  """Train BRATS for a number of steps"""
  # with tf.Graph().as_default():
  #   global_step = tf.contrib.framework.get_or_create_global_step()
  #   mris, labels = inputs()

  #DEBUG TOOLS
  #from tensorflow.python import debug as tf_debug
  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

  mris, labels = brats.inputs()
  
  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    record = mris[:, 1, :, :, :, :]
    print record
    a = record.eval()
    print a.shape
    print labels.eval()

    print "Done"
    coord.request_stop()
    coord.join(threads)
    sess.close()

def main(argv=None): # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
