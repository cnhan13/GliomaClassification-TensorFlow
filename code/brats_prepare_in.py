import numpy as np
import skimage.io as sio
import SimpleITK as sitk

import os

## Simply read mha to nparray, write nparray to file
## read array from file, reshape to get nparray

### audi
TRAIN_PATH = "/home/cnhan21/dl/BRATS2015/BRATS2015_Training/"

NUM_FILES_PER_ENTRY = 5
# Top down look. Bottom: back. Top: front
MHA_DEPTH = 155 # old: height
MHA_HEIGHT= 240 # old: width
MHA_WIDTH = 240 # old: depth

#NEW_DEPTH = 148 - 0 + 1 # 0:149
#NEW_HEIGHT= 220 - 36 + 1 # 36:221
#NEW_WIDTH = 201 - 40 + 1 # 40:202

# func1
def read_mha(filename):
  #mha_data = io.imread(filename, plugin='simpleitk')
  #m = sio.imread(filename[0], plugin='simpleitk')
  m = sitk.ReadImage(filename[0])
  m = sitk.GetArrayFromImage(m)
  #m = m[0:149, 36:221, 40:202]
  t = np.array([m])
  for i in xrange(1,5):
    #p = sio.imread(filename[i], plugin='simpleitk')
    p = sitk.ReadImage(filename[i])
    p = sitk.GetArrayFromImage(p)
    #p = p[0:149, 36:221, 40:202]
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
                MHA_DEPTH, \
                MHA_HEIGHT, \
                MHA_WIDTH)) # order='C'
  return t

# func4
def view_slice(v):
  from matplotlib import pyplot as plt
  plt.imshow(v[101,:,:], cmap='gray')
  plt.show()
  return

# func5
lowD = MHA_DEPTH-1
lowH = MHA_HEIGHT-1
lowW = MHA_WIDTH-1
highD = 0
highH = 0
highW = 0
szD = 0
szH = 0
szW = 0
vmax = [0] * 5
vmin = [50000] * 5
def generate_binary_input(path, train = True):
  if not train:
    print 'generate_binary_input() is not yet implemented for not train data'
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
      find_min_max(v)
      #find_largest_tumor_size(v) #[115, 166, 129]
      #write_tumor(v, bin_path + bin_name)
      #write_tumor(v, bin_path + bin_name)
  return

def find_min_max(v):
  global vmax
  global vmin

  for i in xrange(5):
    amx = np.amax(v[i])
    amn = np.amin(v[i])
    if amx > vmax[i]:
      vmax[i] = amx
    if amn < vmin[i]:
      vmin[i] = amn
  print vmax
  print vmin
  # (min, max) = (-551, 7779)


def write_tumor(v, bb_path):
  TU_D = 115
  TU_H = 166
  TU_W = 129
  print bb_path
  print v.shape
  # Find tumor
  t = np.amax(np.amax(v[4], 1), 1)
  low = 0
  high = MHA_DEPTH
  for j in xrange(MHA_DEPTH):
    if t[j] != 0:
      low = j
      break
  for j in xrange(MHA_DEPTH-1, -1, -1):
    if t[j] != 0:
      high = j
      break
  margin = (TU_D + 1 - (high - low + 1)) / 2
  startD = min(max(low - margin, 0), MHA_DEPTH - TU_D)
  endD = startD + TU_D
  
  t = np.amax(np.amax(v[4], 0), 1)
  low = 0
  high = MHA_HEIGHT
  for j in xrange(MHA_HEIGHT):
    if t[j] != 0:
      low = j
      break
  for j in xrange(MHA_HEIGHT, -1, -1, -1):
    if t[j] != 0:
      high = j
      break
  margin = (TU_H + 1 - (high - low + 1)) / 2
  startH = min(max(low - margin, 0), MHA_HEIGHT - TU_H)
  endH = startH + TU_H
  
  t = np.amax(np.amax(v[4], 0), 0)
  low = 0
  high = MHA_WIDTH
  for j in xrange(MHA_WIDTH):
    if t[j] != 0:
      low = j
      break
  for j in xrange(MHA_WIDTH-1, -1, -1):
    if t[j] != 0:
      high = j
      break
  margin = (TU_W + 1 - (high - low + 1)) / 2
  startW = min(max(low - margin, 0), MHA_WIDTH - TU_W)
  endW = startW + TU_W
  
  tum = v[0, startD:endD, startH:endH, startW:endW]
  print tum.shape
  sio.imsave('test.mha', tum, plugin='simpleitk')
  u = np.array([v[0, startD:endD, startH:endH, startW:endW]])
  for i in xrange(1,5):
    m = v[i, startD:endD, startH:endH, startW:endW]
    u = np.append(u, [m], axis=0)
  print u.shape

  #write_array(u, bb_path)
  return

def write_brain(v, bb_path):
  global lowD
  global highD
  global lowH
  global highH
  global lowW
  global highW
  # Brain focused
  for i in xrange(4, 5):
    t = np.amax(np.amax(v[i], 1), 1)
    for j in xrange(lowD+1):
      if t[j] != 0:
        lowD = j
        break
    for j in xrange(MHA_DEPTH-1, highD-1, -1):
      if t[j] != 0:
        highD = j
        break

    t = np.amax(np.amax(v[i], 0), 1)
    for j in xrange(lowH+1):
      if t[j] != 0:
        lowH = j
        break
    for j in xrange(MHA_HEIGHT-1, highH-1, -1):
      if t[j] != 0:
        highH = j
        break

    t = np.amax(np.amax(v[i], 0), 0)
    for j in xrange(lowW+1):
      if t[j] != 0:
        lowW = j
        break
    for j in xrange(MHA_WIDTH-1, highW-1, -1):
      if t[j] != 0:
        highW = j
        break
  write_array(v, bin_path + bin_name)

  print [lowD, highD, lowH, highH, lowW, highW]
  # whole brain [0, 148, 36, 220, 40, 201]
  # tumor       [7, 145, 40, 213, 50, 193]
  return

def find_largest_tumor_size(v):
  global szD
  global szH
  global szW

  #Tumor focus
  for i in xrange(4, 5):
    t = np.amax(np.amax(v[i], 1), 1)
    low = 0
    high = MHA_DEPTH
    for j in xrange(MHA_DEPTH):
      if t[j] != 0:
        low = j
        break
    for j in xrange(MHA_DEPTH-1, -1, -1):
      if t[j] != 0:
        high = j
        break
    if szD < high - low + 1:
      szD = high - low + 1
    
    t = np.amax(np.amax(v[i], 0), 1)
    low = 0
    high = MHA_HEIGHT
    for j in xrange(MHA_HEIGHT):
      if t[j] != 0:
        low = j
        break
    for j in xrange(MHA_HEIGHT-1, -1, -1):
      if t[j] != 0:
        high = j
        break
    if szH < high - low + 1:
      szH = high - low + 1
    
    t = np.amax(np.amax(v[i], 0), 0)
    low = 0
    high = MHA_WIDTH
    for j in xrange(MHA_WIDTH):
      if t[j] != 0:
        low = j
        break
    for j in xrange(MHA_WIDTH-1, -1, -1):
      if t[j] != 0:
        high = j
        break
    if szW < high - low + 1:
      szW = high - low + 1
  
  print [szD, szH, szW]
  # largest tumor size [115, 166, 129]

def try_read():
  path = "/home/nhan/Desktop/x2goshared/BRATS2015/BRATS2015_Training/in"
  for root, dirs, files in os.walk(path):
    print str(len(files)) + " files"

  for i in range(10):
    t = read_array_from_file(root + "/" + files[i])
    print files[i] + ": ", t.shape

  return

if __name__ == "__main__":
  f = ["VSD.Brain.XX.O.MR_T1.54513.mha",
      "VSD.Brain.XX.O.MR_T1c.54514.mha",
      "VSD.Brain.XX.O.MR_T2.54515.mha",
      "VSD.Brain.XX.O.MR_Flair.54512.mha",
      "VSD.Brain_3more.XX.O.OT.54517.mha"]
  # v = read_mha(f)
  # f2 = "data_105"
  # write_array(v, f2)
  # t = read_array_from_file(f2)

  generate_binary_input(TRAIN_PATH)
  # generate_binary_input(TEST_PATH, False)
