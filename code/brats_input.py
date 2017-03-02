
"""Routine for decoding the BRATS2014 binary file format."""

import os

from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

NUM_FILES_PER_ENTRY = 5
MRI_DIMS = 3
MHA_HEIGHT = 155
MHA_WIDTH = 240
MHA_DEPTH = 240

import skimage.io as sio

def read_mha(filename):
  #mha_data = io.imread(filename, plugin='simpleitk')
  m = sio.imread(filename[0], plugin='simpleitk')
  t = np.array([m])
  for i in xrange(1,5):
    p = sio.imread(filename[i], plugin='simpleitk')
    t = np.append(t, [p], axis=0)
  return t

def write_array(t, fname):
  f = open(fname, mode='wb')
  t.tofile(f) # order='C'
  f.close()

def read_array_from_file(fname):
  f = open(fname, mode='rb')
  t = np.fromfile(f, dtype=np.int16)
  t = t.reshape((NUM_FILES_PER_ENTRY, \
                MHA_HEIGHT, \
                MHA_WIDTH, \
                MHA_DEPTH)) # order='C'
  return t

def view(v):
  plt.imshow(v[112,:,:], cmap='gray')
  plt.show()
  return
  
if __name__ == "__main__":
  f = ["VSD.Brain.XX.O.MR_T1.35536.mha", \
      "VSD.Brain.XX.O.MR_T1c.35535.mha", \
      "VSD.Brain.XX.O.MR_T2.35534.mha", \
      "VSD.Brain.XX.O.MR_Flair.35533.mha", \
      "VSD.Brain_3more.XX.O.OT.42283.mha"]
  v = read_mha(f)
  f2 = "data_105"
  write_array(v, f2)
  t = read_array_from_file(f2)

  print 3005752057

