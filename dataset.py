import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
def split_dataset(): # returns in-order: x_train, x_test, y_train, y_test
    try:
        csv_file= pd.read_csv('dataset/A_Z Handwritten Data.csv').astype('float32')
        
        dataset = pd.DataFrame(csv_file)
        
        x = dataset.drop('0', axis = 1)
        y = dataset['0']
    except:
        print('\'dataset/A_Z Handwritten Data.csv\' directory and file not found.')
        exit()
    return train_test_split(x, y, test_size = 0.2) # 20% train set


def dataset_train_test_to_csv(): # creates csv for x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = split_dataset()
    
    print('\nCreating Csv 1/4')
    x_train.head(1000).to_csv('subdataset/x_train.csv', index=False, header=False)
    
    print('\nCreating Csv 2/4')
    y_train.head(1000).to_csv('subdataset/y_train.csv', index=False, header=False)
    
    print('\nCreating Csv 3/4')
    x_test.head(1000).to_csv('subdataset/x_test.csv', index=False, header=False)
    
    print('\nCreating Csv 4/4')
    y_test.head(1000).to_csv('subdataset/y_test.csv', index=False, header=False)
    
    print('\n-- Done --')

""" def verify_dataset():
    
    try:
        os.mkdir('subdataset')
        
        print('subdataset directory not found, creating...') # skipped if mkdir throws an error.
        verify_dataset()
        return
    except:    
        try:
            x_train = load_x_train() # will optimize
            y_train = load_y_train()
            x_test = load_x_test()
            y_test = load_y_test()
        except:
            print('\nmissing sub dataset, creating...')
            dataset_train_test_to_csv() """
        
def load_train_test_all(): # loads all datasets, still kept just in-case.
    
    x_train = load_x_train()
    y_train = load_y_train()
    x_test= load_x_test()
    y_test = load_y_test()
    
    print(f'\nx_train shape: {x_train.shape}')
    print(f'\ny_train shape: {y_train.shape}')
    print(f'\nx_test shape: {x_test.shape}')
    print(f'\ny_test shape: {y_test.shape}')
    
    return x_train, x_test, y_train, y_test


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
