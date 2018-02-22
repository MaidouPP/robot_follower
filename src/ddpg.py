#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time

dir_path = os.path.dirname(os.path.realpath(__file__))

# tensorflow log dir
TF_LOG_DIR = dir_path + '/../log'

# tensorflow output dir
TF_OUT_DIR = dir_path + '/../output'


class DDPG:

    def __init__(self):

        if not tf.gfile.Exists(TF_LOG_DIR):
            tf.gfile.MakeDirs(TF_LOG_DIR)
        if not tf.gfile.Exists(TF_OUT_DIR):
            tf.gfile.MakeDirs(TF_OUT_DIR)

        self.session = tf.session()

        self.memory
