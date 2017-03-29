import numpy as np
import skimage.io as sio

import os

## Simple read mha to nparray, write nparray to file
## read array from file, reshape to get nparray

### audi
#TRAIN_PATH = "/home/cnhan21/Desktop/dl/BRATS2015/BRATS2015_Training/"
#TEST_PATH = "/home/cnhan21/Desktop/dl/BRATS2015/Testing/"

### farmer
TRAIN_PATH = "/home/ubuntu/dl/BRATS2015/BRATS2015_Training"

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
def view_slice(v):
  from matplotlib import pyplot as plt
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
      path_file_list_counter = 0
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

  generate_binary_input(TRAIN_PATH)
  # generate_binary_input(TEST_PATH, False)
