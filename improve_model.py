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

modelLoad.fit(
    x_train, y_train,
    epochs=limit_train_count(trainCount,0,10000) # basically an input with min and max limit
)


print('\n-- Training end, updating model\n')
modelLoad.save(f'./model/{modelName}.keras')