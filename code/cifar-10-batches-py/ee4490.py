import tensorflow as tf
import numpy as np

def loadAll():
	fs = []
	for i in range(1,6):
		file_name = "data_batch_" + str(i)
		fs.append(file_name)
	return fs

def batch2tf(batch):
	out = np.reshape(batch, (len(batch),-1,3), order='F')
	out = np.reshape(out, (len(out),32,32,3))
	return out

# file_name = "data_batch_" + str(1)
# print file_name
# data = unpickle(file_name)
# print data.keys()
# # data.keys is ['data', 'labels', 'batch_label', 'filenames']
# batch1 = batch2tf(data['data'])

# print batch1.shape

sess = tf.InteractiveSession()

# Load file into dictionary
def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	data_dict = cPickle.load(fo)
	fo.close()
	return data_dict

# Parse dictionary data to images and labels
def input(d, L):
	# Parse images
	print "Came here"
	x = np.reshape(d, (len(d),-1,3), order='F')
	x = np.reshape(x, (len(x),32,32,3))
	x = tf.constant(x, name="X")
	# Parse labels
	labels = []
	for i in range(10):
		labels.insert(i, tf.equal(L,[i]))
	labels_return = tf.transpose(tf.pack(labels))
	return tf.to_float(x), tf.to_float(labels_return)

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def maxpool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32, shape=None) # to fetch tensor of any shape
y_ = tf.placeholder(tf.float32, shape=[None,10])

W_conv1 = weight_variable([5,5,3,32])
b_conv1 = bias([32])

h_conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)
h_pool1 = maxpool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) +b_conv2)
h_pool2 = maxpool_2x2(h_conv2)

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:

	sess.run(tf.initialize_all_variables())
	print "Starting Sess"

	fs_list = loadAll()
	d = np.array([])
	L = []
	for i in range(len(fs_list)):
		datai = unpickle(fs_list[i]);

		if (i==0):
			d = datai['data']
			print d.shape
			L = datai['labels']
		else:
			d = np.vstack((d,datai['data']))
			L = L+datai['labels']

	data, labels = input(d,L)
	image_batch, label_batch = tf.train.shuffle_batch(
		[data, labels], batch_size=50, capacity=30000, min_after_dequeue=10000, num_threads=2, enqueue_many=True)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	print "Starting main loop"
	for i in xrange(200):
		imgs, lbls = sess.run([image_batch, label_batch])
		if i%10==0:
			train_accuracy = accuracy.eval(feed_dict={x:imgs, y_:lbls, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x:imgs, y_:lbls, keep_prob: 0.5})

	print "Done training"

	test_dict = unpickle("test_batch")
	print "here"
	test_data, test_labels = input(test_dict['data'], test_dict['labels'])
	test_imgs, test_lbls = sess.run([test_data, test_labels])
	print test_imgs, test_lbls
	print("test accuracy %g"%accuracy.eval(feed_dict={x:test_imgs, y_:test_lbls, keep_prob: 1.0}))

	writer = tf.train.SummaryWriter('./my_graph', sess.graph)
	writer.close()

	coord.request_stop()
	coord.join(threads)
	sess.close()
