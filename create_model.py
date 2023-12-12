# IF Console returns "KILLED", the process dies due to lack of proper specs.

import tensorflow as tf;

from dataset import load_x_train, load_y_train, limit_train_count

from keras.models import Sequential
from keras.layers import Dense

# from keras.callbacks import EarlyStopping

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
        
        Dense(126, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.2)),
        Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.2)),
        Dense(26, activation='softmax')
        
    ], name = "new_model" 
)

print('\n-- Model Created!')

print('\n-- Training model\n')

trainCount = int(input('Initial Training Iterations (min=0, max=10000): '))

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, validation_split = 0.1, epochs=100)


print('\n-- Training end, saving model\n')
modelName = str(input('Enter a new model name: '))
model.save(f'./model/{modelName}.keras')