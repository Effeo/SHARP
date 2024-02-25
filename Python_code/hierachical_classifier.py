import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hiclass import LocalClassifierPerNode
import scipy.io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time

def demultiplex(list_to_convert):
    list_to_append = []
    for element in list_to_convert:
        if (element == np.array(["No empty", "Move", "Move in place", "Person sitting down/standing up"])).all():
            list_to_append.append(0)
        elif (element == np.array(["Empty", "", "", ""])).all():
            list_to_append.append(1)
        elif (element == np.array(["No empty", "Move", "Move in place", "Arm exercise"])).all():
            list_to_append.append(2)
        elif (element == np.array(["No empty", "Move", "Move in place", "Jumping"])).all():
            list_to_append.append(3)
        elif (element == np.array(["No empty", "No move", "Person sitting still", ""])).all():
            list_to_append.append(4)
        elif (element == np.array(["No empty", "Move", "Move around", "Running"])).all():
            list_to_append.append(5)
        elif (element == np.array(["No empty", "No move", "Person standing still", ""])).all():
            list_to_append.append(6)
        elif (element == np.array(["No empty", "Move", "Move around", "Walking"])).all():
            list_to_append.append(7)
    list_to_append = np.array(list_to_append)
    return list_to_append

def demultiplex_level(list_to_convert, level):
    list_to_append = []
    for element in list_to_convert:
        if level == 0:
            if element[level] == "No empty":
                list_to_append.append(0)
            elif element[level] == "Empty":
                list_to_append.append(1)
        if level == 1:
            if element[level] == "No move":
                list_to_append.append(0)
            elif element[level] == "Move":
                list_to_append.append(1)
        if level == 2:
            if element[level] == "Move in place":
                list_to_append.append(0)
            elif element[level] == "Move around":
                list_to_append.append(1)
            elif element[level] == "Person sitting still":
                list_to_append.append(2)
            elif element[level] == "Person standing still":
                list_to_append.append(3)
        if level == 3:
            if element[level] == "Arm exercise":
                list_to_append.append(0)
            elif element[level] == "Person sitting down/standing up":
                list_to_append.append(1)
            elif element[level] == "Jumping":
                list_to_append.append(2)
            elif element[level] == "Walking":
                list_to_append.append(3)
            elif element[level] == "Running":
                list_to_append.append(4)
        if level == 'leaf':
            list_to_append = demultiplex(list_to_convert)
    return list_to_append

def get_metrics(predictions, level, conf_matrix, y_test):
    pred = predictions
    y = y_test
    if isinstance(level, int):
        if level > 0:
            pred = [x for i, x in enumerate(predictions) if x[level - 1] == y_test[i][level - 1]]
            y = [x for i, x in enumerate(y_test) if x[level - 1] == predictions[i][level - 1]]
    pred = demultiplex_level(pred, level)
    y = demultiplex_level(y, level)
    print(len(pred), len(y))
    cm = confusion_matrix(y, pred, normalize='true')
    conf_matrix.append(cm) 

def plot(conf_matrixes, labels, name):
    cm_avg = np.mean(conf_matrixes, axis = 0)
    disp = ConfusionMatrixDisplay(np.around(cm_avg, decimals = 2), display_labels = labels) 
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('models_outputs/'+ name)

dataset = scipy.io.loadmat('./dataset/dataset.mat')
dataset = dataset['csi_buff']

raw_labels = dataset[:,-1]
dataset = np.delete(dataset, -1, 1)
labels=[]


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
confusion_matrixes_level_0 = []
confusion_matrixes_level_1 = []
confusion_matrixes_level_2 = []
confusion_matrixes_level_3 = []

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
    get_metrics(predictions, 0, confusion_matrixes_level_0, y_test)
    get_metrics(predictions, 1, confusion_matrixes_level_1, y_test)
    get_metrics(predictions, 2, confusion_matrixes_level_2, y_test)
    get_metrics(predictions, 3, confusion_matrixes_level_3, y_test)
    get_metrics(predictions, 'leaf', confusion_matrixes, y_test)

avg_time = np.mean(time_to_fit)
print(avg_time)


plot(confusion_matrixes_level_0, ['NE', 'E'], 'hierarchical_classifier_confusion_matrix_level_0.jpeg')
plot(confusion_matrixes_level_1, ['NM', 'M'], 'hierarchical_classifier_confusion_matrix_level_1.jpeg')
plot(confusion_matrixes_level_2, ['MP', 'MA','SIT', 'STAND'], 'hierarchical_classifier_confusion_matrix_level_2.jpeg')
plot(confusion_matrixes_level_3, ['AE', 'PS', 'J', 'W', 'R'], 'hierarchical_classifier_confusion_matrix_level_3.jpeg')
plot(confusion_matrixes, ['C', 'E', 'G', 'I', 'J', 'L', 'R', 'S'], 'hierarchical_classifier_confusion_matrix_2.jpeg')


