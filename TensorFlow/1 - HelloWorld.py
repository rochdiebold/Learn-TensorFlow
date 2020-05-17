# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:52:52 2020

@author: Roch
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([28.0, 13.0, 4.0, 1.0, 4.0, 13.0, 28.0, 49.0], dtype=float)

model.fit(xs, ys, epochs=1000)

print(model.predict([100.0]))