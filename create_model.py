import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf;

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

# from keras import optimizers

from keras.utils import to_categorical #.np_utils
from keras import Input

print('-- phase 1: Reading dataset --')

csv_file= pd.read_csv('dataset/A_Z Handwritten Data.csv').astype('float32')
dataset = pd.DataFrame(csv_file)
print('\n Dataset acquired!\n')

print('-- phase 2: Preparing dataset --')

x = dataset.drop('0', axis = 1)
y = dataset['0']

print(f'x: {x.shape}')
print(f'y: {y.shape}\n\n')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

y_train = np.reshape(y_train.values, (y_train.shape[0], 1))

print('\n Dataset prepared! \n')

print('-- phase 3: Preparing model --')

print(f'{x_train.shape}')

model = Sequential(
    [               
        tf.keras.Input(shape=(784,)),
        
        Dense(128, activation='relu', name='L2'),
        Dense(64, activation='relu', name='L3'),
        Dense(26, activation='softmax', name='L4') # 26 due to y_train
        
    ], name = "my_model" 
)

print('\n Model Created! \n')

print('-- phase 4: Training model --')    


model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

model.fit(
    x_train, y_train,
    epochs=20
)


print('\n-- Training end, saving model --\n')
model.save('./model/main.keras')