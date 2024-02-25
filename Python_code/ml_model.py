import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import time
import pickle
import matplotlib.pyplot as plt
import scipy.io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('block_size')
args = parser.parse_args()
block_size = int(args.block_size)
dataset = scipy.io.loadmat('./dataset/dataset2.mat')
dataset = dataset['csi_buff']

labels = dataset[:,-1]
dataset = np.delete(dataset, -1, 1)
name = 'ml_model'
model = RandomForestClassifier(random_state = 12)
i = 0
while i < np.shape(dataset)[1]:
    
    confusion_matrixes = []
    time_to_fit = []
    splits = 10
    kf = KFold(n_splits = splits, random_state = None, shuffle = True)
    if i + block_size != np.shape(dataset)[1]:
        ds = dataset[:,i:i+block_size]
    else:
        ds = dataset[i:]

    for train_index, test_index in kf.split(ds):
        x_train, x_test = ds[train_index], ds[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        start = time.process_time()
        model.fit(x_train, y_train)
        end = time.process_time() - start
        time_to_fit.append(end)
        predictions = model.predict(x_test)
        cm = confusion_matrix(y_test, predictions, normalize='true')
        confusion_matrixes.append(cm)


    cm_avg = np.mean(confusion_matrixes, axis = 0)
    avg_time = np.mean(time_to_fit)

    print(avg_time)
    disp = ConfusionMatrixDisplay(np.around(cm_avg, decimals = 2), display_labels=['C', 'E', 'G', 'I', 'J', 'L', 'R', 'S'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('models_outputs/confusion_matrix_' + name +  '[' + str(int(i/2)) + ':' + str(int((i+block_size)/2)-1) + '].jpeg', pad_inches=10)
    i = i + block_size

pickle.dump(model, open('models_outputs/' + name + '.pkl', 'wb'))




