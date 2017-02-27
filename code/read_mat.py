# import scipy.io as sio

# filename="test.mat"
# input = sio.loadmat(filename)['V']

# (240, 240, 155)
# input.shape

# See image:
# from matplotlib import pyplot as plt
# plt.imshow(input[:,:,75])
# plt.show()

import scipy.io as sio
import os

data = {};
# traverse the [...]/mat directory
for root, dirs, files in os.walk("/home/nhan/Documents/dl/ChiNhan_2016/mha/BRATS2014_training/mat"):
	print "root: " + root
	print "dirs: "
	print dirs
	if files:
		print "Parsing " + str(len(files)) + " files"
		for file in files:
			print "Processing " + file
			filename = root+'/'+file
			data[filename] = sio.loadmat(filename)['V']

print "Finished parsing ", len(data), " files"
print "Data files: ", data.keys().sort()