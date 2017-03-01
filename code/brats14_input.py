
"""Routine for decoding the BRATS2014 binary file format."""

import os

from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf

from matplotlib import pyplot as plt

MRI_DIMS = 3
MHA_HEIGHT = 155
MHA_WIDTH = 240
MHA_DEPTH = 240

import skimage.io as io

def read_mha(filename):
  mha_data = io.imread(filename, plugin='simpleitk')
  return mha_data


def view(v):
  plt.imshow(v[112,:,:], cmap='gray')
  plt.show()
  return
  
if __name__ == "__main__":
  f = "VSD.Brain.XX.O.MR_T1.35536.mha"
  v = read_mha(f)
  view(v)

