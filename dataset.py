import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(): # returns in-order: x_train, x_test, y_train, y_test
    csv_file= pd.read_csv('main_dataset/A_Z Handwritten Data.csv').astype('float32')
        
    dataset = pd.DataFrame(csv_file)
    
    x = dataset.drop('0', axis = 1)
    y = dataset['0']
    
    return train_test_split(x, y, test_size = 0.2) # 20% train set


def dataset_to_csv(x1,y1,x2,y2): # creates csv for x_train, x_test, y_train, y_test
    
    x_train, x_test, y_train, y_test = split_dataset()
    
    print('\nCreating Csv 1/4')
    x_train.head(x1).to_csv('subdataset/x_train.csv', index=False, header=False)
    
    print('\nCreating Csv 2/4')
    y_train.head(y1).to_csv('subdataset/y_train.csv', index=False, header=False)
    
    print('\nCreating Csv 3/4')
    x_test.head(x2).to_csv('subdataset/x_test.csv', index=False, header=False)
    
    print('\nCreating Csv 4/4')
    y_test.head(y2).to_csv('subdataset/y_test.csv', index=False, header=False)
    
    print('\n-- Done --')

def dataset_to_csv_all(): # creates csv for x_train, x_test, y_train, y_test
    
    x_train, x_test, y_train, y_test = split_dataset()
    
    print('\nCreating Csv 1/4')
    x_train.to_csv('subdataset/x_train.csv', index=False, header=False)
    
    print('\nCreating Csv 2/4')
    y_train.to_csv('subdataset/y_train.csv', index=False, header=False)
    
    print('\nCreating Csv 3/4')
    x_test.to_csv('subdataset/x_test.csv', index=False, header=False)
    
    print('\nCreating Csv 4/4')
    y_test.to_csv('subdataset/y_test.csv', index=False, header=False)
    
    print('\n-- Done --')
    
def verify_all_datasets(): # loads all datasets, still kept just in-case.
    
    x_train = load_x_train()
    y_train = load_y_train()
    x_test= load_x_test()
    y_test = load_y_test()
    
    print(f'\nx_train shape: {x_train.shape}')
    print(f'\ny_train shape: {y_train.shape}')
    print(f'\nx_test shape: {x_test.shape}')
    print(f'\ny_test shape: {y_test.shape}')
    
def load_x_train():
    return pd.read_csv('subdataset/x_train.csv', header=None).astype('float32')

def load_y_train():
    return pd.read_csv('subdataset/y_train.csv', header=None).astype('float32')

def load_x_test():
    return pd.read_csv('subdataset/x_test.csv', header=None).astype('float32')

def load_y_test():
    return pd.read_csv('subdataset/y_test.csv', header=None).astype('float32')

def limit_train_count(trainCount, min, max):
    if(trainCount > max): return max
    elif(trainCount < min): return min
    return trainCount
