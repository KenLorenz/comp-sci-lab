# IF Console returns "KILLED", the process dies due to lack of proper specs.

import tensorflow as tf;

from dataset import load_x_train, load_y_train, limit_train_count

from keras.models import Sequential
from keras.layers import Dense

print('-- Loading dataset...')

x_train = load_x_train()
y_train = load_y_train()

print(f'\nx_train shape: {x_train.shape}')
print(f'\ny_train shape: {y_train.shape}')

print('\n-- Dataset prepared!')

print('\n-- Creating Model')

model = Sequential(
    [               
        tf.keras.Input(shape=(784,)),
        
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(26, activation='softmax') # 26 due to y_train
        
    ], name = "my_model" 
)

print('\n-- Model Created!')

print('\n-- Training model\n')

trainCount = int(input('Initial Training Iterations (min=0, max=10000): '))

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

model.fit(
    x_train, y_train,
    epochs=limit_train_count(trainCount,0,10000) # basically an input with min and max limit
)


print('\n-- Training end, saving model\n')
modelName = str(input('Enter a new model name: '))
model.save(f'./model/{modelName}.keras')