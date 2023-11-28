

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf;

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout

from keras.optimizers import SGD
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical #.np_utils
from keras import Input

print('-- phase 1: Reading dataset --')
# phase 1
csv_file= pd.read_csv('dataset/A_Z Handwritten Data.csv').astype('float32')
dataset = pd.DataFrame(csv_file)

print('-- phase 2: Preparing dataset --')
# phase 2
x = dataset.drop('0', axis = 1)
y = dataset['0']

print(f'x: {x.shape}')
print(f'y: {y.shape}\n\n')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28,1))
y_train = np.reshape(y_train.values, (y_train.shape[0], 1))

print('-- phase 3: Preparing model--')
# phase 3

print(f'{x_train.shape}')

# tf.random.set_seed(1234)
model = Sequential(
    [               
        tf.keras.Input(shape=(784,)),
        
        #Dense(30, activation='relu', name='L1'), # 28
        Dense(128, activation='relu', name='L2'), # 14
        Dense(64, activation='relu', name='L3'), # 7
        Dense(26, activation='softmax', name='L4') # 1
        
    ], name = "my_model" 
)


print('-- phase 4: Training model --')    
# phase 4

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

model.fit(
    x_train, y_train,
    epochs=1
)