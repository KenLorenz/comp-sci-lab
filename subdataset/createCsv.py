# This code only needs to be run one time.

import pandas as pd
from sklearn.model_selection import train_test_split

# ignore I / W messages.


print('Loading Main Dataset')
csv_file= pd.read_csv('dataset/A_Z Handwritten Data.csv').astype('float32')
dataset = pd.DataFrame(csv_file)

x = dataset.drop('0', axis = 1)
y = dataset['0']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

print('\nCreating Csv 1/4')
x_train.to_csv('subdataset/x_train.csv')
print('\nCreating Csv 2/4')
y_train.to_csv('subdataset/y_train.csv')
print('\nCreating Csv 3/4')
x_test.to_csv('subdataset/x_test.csv')
print('\nCreating Csv 4/4')
y_test.to_csv('subdataset/y_test.csv')
print('\n\nDone')
