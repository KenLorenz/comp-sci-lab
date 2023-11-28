import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf;

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

from keras.utils import to_categorical
from keras import Input

print('loading the model')
load = tf.keras.models.load_model('model/main.keras')