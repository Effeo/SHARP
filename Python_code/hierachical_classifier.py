import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hiclass import LocalClassifierPerNode
import scipy.io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time
import pickle

def demultiplex(list_to_convert):
    list_to_append = []
    for element in list_to_convert:
        if (element == np.array(["No empty", "Move", "Move in place", "Person sitting down/standing up"])).all():
            list_to_append.append(0)
        if (element == np.array(["Empty", "", "", ""])).all():
            list_to_append.append(1)
        if (element == np.array(["No empty", "Move", "Move in place", "Arm exercise"])).all():
            list_to_append.append(2)
        if (element == np.array(["No empty", "Move", "Move in place", "Jumping"])).all():
            list_to_append.append(3)
        if (element == np.array(["No empty", "No move", "Person sitting still", ""])).all():
            list_to_append.append(4)
        if (element == np.array(["No empty", "Move", "Move around", "Running"])).all():
            list_to_append.append(5)
        if (element == np.array(["No empty", "No move", "Person standing still", ""])).all():
            list_to_append.append(6)
        if (element == np.array(["No empty", "Move", "Move around", "Walking"])).all():
            list_to_append.append(7)
    list_to_append = np.array(list_to_append)
    return list_to_append

dataset = scipy.io.loadmat('./dataset/dataset2.mat')
dataset = dataset['csi_buff']

raw_labels = dataset[:,-1]
dataset = np.delete(dataset, -1, 1)
labels=[]

name = 'hierarchical_classifier'


for l in raw_labels:
    if l == 0:
        labels.append(["No empty", "Move", "Move in place", "Person sitting down/standing up"])
    elif l == 1:
        labels.append(["Empty", "", "", ""])
    elif l == 2:
        labels.append(["No empty", "Move", "Move in place", "Arm exercise"])
    elif l == 3:
        labels.append(["No empty", "Move", "Move in place", "Jumping"])
    elif l == 4:
        labels.append(["No empty", "No move", "Person sitting still", ""])
    elif l == 5:
        labels.append(["No empty", "Move", "Move around", "Running"])
    elif l == 6:
        labels.append(["No empty", "No move", "Person standing still", ""])
    elif l == 7:
        labels.append(["No empty", "Move", "Move around", "Walking"])


labels = np.array(labels, dtype=object) 

rf = RandomForestClassifier()
classifier = LocalClassifierPerNode(local_classifier=rf)
confusion_matrixes = []
time_to_fit = []
splits = 10
kf = KFold(n_splits = splits, random_state = None, shuffle = True)
for train_index, test_index in kf.split(dataset):
    x_train, x_test = dataset[train_index], dataset[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    start = time.process_time()
    classifier.fit(x_train, y_train)
    end = time.process_time() - start
    time_to_fit.append(end)
    predictions = classifier.predict(x_test)
    predictions = demultiplex(predictions)
    y_test = demultiplex(y_test)
    cm = confusion_matrix(y_test, predictions, normalize='true')
    confusion_matrixes.append(cm) 
avg_time = np.mean(time_to_fit)
cm_avg = np.mean(confusion_matrixes, axis = 0)
disp = ConfusionMatrixDisplay(np.around(cm_avg, decimals = 2), display_labels = ['C', 'E', 'G', 'I', 'J', 'L', 'R', 'S']) 
print(avg_time)
disp.plot(cmap=plt.cm.Blues)
plt.savefig('models_outputs/hierarchical_classifier_confusion_matrix.jpeg')
pickle.dump(classifier, open('models_outputs/' + name + '.pkl', 'wb'))


