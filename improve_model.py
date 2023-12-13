<<<<<<< HEAD
import tensorflow as tf;

from dataset import load_x_train, load_y_train, limit_train_count


print('-- Loading dataset...')

x_train = load_x_train()
y_train = load_y_train()

print(f'\nx_train shape: {x_train.shape}')
print(f'\ny_train shape: {y_train.shape}')

print('\n-- Dataset prepared!')

print('\n-- Training model\n') 


modelName = str(input('Enter an existing model name: '))
try:
    modelLoad = tf.keras.models.load_model(f'model/{modelName}.keras')
except:
    print('Model not found!')
    exit()

trainCount = int(input('Training Iterations (min=0, max=10000): '))

modelLoad.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy'],
)

modelLoad.fit(
    x_train, y_train, validation_split = 0.1,
    epochs=limit_train_count(trainCount,0,10000) # basically an input with min and max limit
)


print('\n-- Training end, updating model\n')
modelLoad.save(f'./model/{modelName}.keras')
=======
# IF Console returns "KILLED", the process dies due to lack of proper specs.

import numpy as np
import pandas as pd

import tensorflow as tf;

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

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

print('-- phase 3: Training model --')    

modelLoad = tf.keras.models.load_model('model/main.keras')

modelLoad.fit(
    x_train, y_train,
    epochs=50
)


print('\n-- Training end, updating model --\n')
modelLoad.save('./model/main.keras')
>>>>>>> origin/Harry's_branch
