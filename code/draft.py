  
  group1_ot = _conv3pool3(mris[:, 4, :, :, :, :], 'conv1_ot', 'pool1_ot',
                        [5, 5, 5, 1, 4], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])

  group1b_ot = _conv3pool3(group1_ot, 'conv1b_ot', 'pool1b_ot',
                        [3, 3, 3, 4, 8], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])

  group2_3_t1 = _conv3conv3pool3(group1b_t1, 'conv2_t1', 'conv3_t1', 'pool3_t1',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group2_3_t1c = _conv3conv3pool3(group1b_t1c, 'conv2_t1c', 'conv3_t1c', 'pool3_t1c',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group2_3_t2 = _conv3conv3pool3(group1b_t2, 'conv2_t2', 'conv3_t2', 'pool3_t2',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group2_3_fl = _conv3conv3pool3(group1b_fl, 'conv2_fl', 'conv3_fl', 'pool3_fl',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group2_3_ot = _conv3conv3pool3(group1b_ot, 'conv2_ot', 'conv3_ot', 'pool3_ot',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])


  group4_5_t1 = _conv3conv3pool3(group2_3_t1, 'conv4_t1', 'conv5_t1', 'pool5_t1',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group4_5_t1c = _conv3conv3pool3(group2_3_t1c, 'conv4_t1c', 'conv5_t1c', 'pool5_t1c',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group4_5_t2 = _conv3conv3pool3(group2_3_t2, 'conv4_t2', 'conv5_t2', 'pool5_t2',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group4_5_fl = _conv3conv3pool3(group2_3_fl, 'conv4_fl', 'conv5_fl', 'pool5_fl',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group4_5_ot = _conv3conv3pool3(group2_3_ot, 'conv4_ot', 'conv5_ot', 'pool5_ot',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])


  group6_7_t1 = _conv3conv3pool3(group4_5_t1, 'conv6_t1', 'conv7_t1', 'pool7_t1',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group6_7_t1c = _conv3conv3pool3(group4_5_t1c, 'conv6_t1c', 'conv7_t1c', 'pool7_t1c',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group6_7_t2 = _conv3conv3pool3(group4_5_t2, 'conv6_t2', 'conv7_t2', 'pool7_t2',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group6_7_fl = _conv3conv3pool3(group4_5_fl, 'conv6_fl', 'conv7_fl', 'pool7_fl',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group6_7_ot = _conv3conv3pool3(group4_5_ot, 'conv6_ot', 'conv7_ot', 'pool7_ot',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])


  group8_9_t1 = _conv3conv3pool3(group6_7_t1, 'conv8_t1', 'conv9_t1', 'pool9_t1',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group8_9_t1c = _conv3conv3pool3(group6_7_t1c, 'conv8_t1c', 'conv9_t1c', 'pool9_t1c',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group8_9_t2 = _conv3conv3pool3(group6_7_t2, 'conv8_t2', 'conv9_t2', 'pool9_t2',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group8_9_fl = _conv3conv3pool3(group6_7_fl, 'conv8_fl', 'conv9_fl', 'pool9_fl',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group8_9_ot = _conv3conv3pool3(group6_7_ot, 'conv8_ot', 'conv9_ot', 'pool9_ot',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])


  group10_11_t1 = _conv3conv3pool3(group8_9_t1, 'conv10_t1', 'conv11_t1', 'pool11_t1',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group10_11_t1c = _conv3conv3pool3(group8_9_t1c, 'conv10_t1c', 'conv11_t1c', 'pool11_t1c',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group10_11_t2 = _conv3conv3pool3(group8_9_t2, 'conv10_t2', 'conv11_t2', 'pool11_t2',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group10_11_fl = _conv3conv3pool3(group8_9_fl, 'conv10_fl', 'conv11_fl', 'pool11_fl',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])
  group10_11_ot = _conv3conv3pool3(group8_9_ot, 'conv10_ot', 'conv11_ot', 'pool11_ot',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 1, 1, 1, 1])


  group12_13_t1 = _conv3conv3pool3(group10_11_t1, 'conv12_t1', 'conv13_t1', 'pool13_t1',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group12_13_t1c = _conv3conv3pool3(group10_11_t1c, 'conv12_t1c', 'conv13_t1c', 'pool13_t1c',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group12_13_t2 = _conv3conv3pool3(group10_11_t2, 'conv12_t2', 'conv13_t2', 'pool13_t2',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group12_13_fl = _conv3conv3pool3(group10_11_fl, 'conv12_fl', 'conv13_fl', 'pool13_fl',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  group12_13_ot = _conv3conv3pool3(group10_11_ot, 'conv12_ot', 'conv13_ot', 'pool13_ot',
                              [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                              [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])
  
  
