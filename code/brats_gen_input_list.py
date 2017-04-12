import tensorflow as tf
import os.path
import random
import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('list_dir',
                           '/home/cnhan21/dl/BRATS2015/tumor_cropped/',
                           """Path to tumor cropped input lists""")
tf.app.flags.DEFINE_string('data_dir',
                           FLAGS.list_dir + 'BRATS2015_Training/',
                           """Path to the tumor cropped BRATS *.in files""")

def get_input_list(idx):
  train_list_name = FLAGS.list_dir + 'train_list' + str(idx)
  test_list_name = FLAGS.list_dir + 'test_list' + str(idx)

  if tf.gfile.Exists(train_list_name) and tf.gfile.Exists(test_list_name):
    print "Fetching existed list of files:"
    
    # open created list of 'train files'
    with open(train_list_name, 'rb') as f:
      train_list = pickle.load(f)
      print "File: {}".format(train_list_name)

    with open(test_list_name, 'rb') as f:
      test_list = pickle.load(f)
      print "File: {}".format(test_list_name)

  else:
    print "Creating new list of files to train:"

    # create new list of train files
    
    # Probability of taking a record of high grade
    # Train data info:        221 H, 54 L
    # Train quantity target:  70 H, 45 L
    # Train ratio target:     70/221 H, 45/54 H
    prob_take_high = 1./221
    prob_take_low = 1./54

    train_list = []
    test_list = []
    count_high_train = 0
    count_high_test = 0

    for (_, _, filenames) in os.walk(FLAGS.data_dir):
      for filename in filenames:
        prob_take = random.random()
        if (filename[0] == 'H' and prob_take < prob_take_high) or \
            (filename[0] == 'L' and prob_take < prob_take_low):
          
          train_list.append(filename)
          if filename[0] == 'H':
            count_high_train += 1

        else:
          test_list.append(filename)
          if filename[0] == 'H':
            count_high_test += 1

      break
    
    with open(train_list_name, 'wb') as f:
      pickle.dump(train_list, f)

    print "Total train files: {0} High-Prob: {1}-{2:.2f} Low-Prob: {3}-{4:.2f}"\
          .format(len(train_list),
                  count_high_train,
                  prob_take_high,
                  len(train_list) - count_high_train,
                  prob_take_low)
    print "File: {}".format(train_list_name)
    
    with open(test_list_name, 'wb') as f:
      pickle.dump(test_list, f)

    print "Total test files: {0} High: {1} Low: {2}"\
          .format(len(test_list),
                  count_high_test,
                  len(test_list) - count_high_test)
    print "File: {}".format(test_list_name)
    
  return train_list, test_list
  
