# IF Console returns "KILLED", the process dies due to lack of proper specs.

<<<<<<< HEAD
import tensorflow as tf;

from dataset import load_x_train, load_y_train, limit_train_count
=======
import numpy as np
import pandas as pd

import tensorflow as tf;

from sklearn.model_selection import train_test_split
>>>>>>> origin/Harry's_branch

from keras.models import Sequential
from keras.layers import Dense

<<<<<<< HEAD
# from keras.callbacks import EarlyStopping

print('-- Loading dataset...')

x_train = load_x_train()
y_train = load_y_train()

print(f'\nx_train shape: {x_train.shape}')
print(f'\ny_train shape: {y_train.shape}')

print('\n-- Dataset prepared!')

print('\n-- Creating Model')
=======
from keras.utils import to_categorical
from keras import Input

print('-- phase 1: Reading dataset --')

csv_file= pd.read_csv('dataset/A_Z Handwritten Data.csv').astype('float32')
dataset = pd.DataFrame(csv_file)
print('\n Dataset acquired!\n')

print('-- phase 2: Preparing dataset --')

x = dataset.drop('0', axis = 1)
y = dataset['0']

print(f'x: {x.shape}')
print(f'y: {y.shape}')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

y_train = np.reshape(y_train.values, (y_train.shape[0], 1))

print('\n Dataset prepared! \n')

print('-- phase 3: Preparing model --')
>>>>>>> origin/Harry's_branch

model = Sequential(
    [               
        tf.keras.Input(shape=(784,)),
        
<<<<<<< HEAD
        Dense(126, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.2)),
        Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.2)),
        Dense(26, activation='softmax')
        
    ], name = "new_model" 
) # recommended fit = 200

print('\n-- Model Created!')

print('\n-- Training model\n')

trainCount = int(input('Initial Training Iterations (min=0, max=10000): '))

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, validation_split = 0.1, epochs=100)


print('\n-- Training end, saving model\n')
modelName = str(input('Enter a new model name: '))
model.save(f'./model/{modelName}.keras')
=======
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(26, activation='softmax') # 26 due to y_train
        
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
    epochs=1
)


print('\n-- Training end, saving model --\n')
model.save('./model/main.keras')
>>>>>>> origin/Harry's_branch
